import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import AdamW
import torch.nn.functional as F
from torch import masked_select
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, MBartForConditionalGeneration, MBartConfig
from transformers import AutoTokenizer, MBartTokenizer, MBart50TokenizerFast
from datasets import load_dataset, load_metric, interleave_datasets
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

class CKALoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, eps ):
        super().__init__()
        self.eps = eps
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1,dS).to(SH.device,torch.float64)
        TH = TH.view(-1,dT).to(SH.device,torch.float64)
        
        slen = SH.size(0)
                # Dropout on Hidden State Matching
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
                
        num = torch.norm(SH.t().matmul(TH),'fro')
        den1 = torch.norm(SH.t().matmul(SH),'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH),'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)



def Eval_Student(student,teacher,eval_dataset,test_dataset, batch_size = 32):

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    student.eval()
    teacher.eval()

    loss = [0]*(student.config.encoder_layers + student.config.decoder_layers + 4)
    nBatch = 0
    for batch in eval_dataloader:
        with torch.cuda.amp.autocast():
            batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
            decoder_input_ids = shift_tokens_right(batch['label_ids'], pad_token_id, pad_token_id)
            dec_mask = decoder_input_ids.ne(pad_token_id)
            torch.cuda.empty_cache()
            student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids'])  
            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids'])    
            #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)

            dV = student_out.logits.size(-1)
            logit_mask = batch['label_attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()
            SL = masked_select(student_out.logits,logit_mask).view(-1,dV)
            loss[0] += CELoss(SL,masked_select(batch['label_ids'],batch['label_attention_mask'].bool()))
            loss[1] += ((temperature)**2)*STLoss(F.log_softmax(SL/ temperature,dim=-1), F.softmax(masked_select(teacher_out.logits,logit_mask).view(-1,dV)/ temperature, dim=-1))
            #loss[2] += CELoss(rearrange(student_out.logits[:,:pseudo.size(1),:],'a b c -> (a b) c'),rearrange(pseudo, 'a b -> (a b)'))

            teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.encoder_hidden_states[-1]).bool()
            student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.encoder_hidden_states[-1]).bool()
            dT = teacher_out.encoder_hidden_states[-1].size(-1)
            dS = student_out.encoder_hidden_states[-1].size(-1)                
            
            loss[2] += CSLoss(masked_select(student_out.encoder_hidden_states[0],student_mask).view(-1,dS), \
                                    masked_select(teacher_out.encoder_hidden_states[0],teacher_mask).view(-1,dT))
                
            for i in range(student.config.encoder_layers):
                torch.cuda.empty_cache()
                loss[i+3] += CSLoss(masked_select(student_out.encoder_hidden_states[i+1],student_mask).view(-1,dS), \
                                            masked_select(teacher_out.encoder_hidden_states[i+1],teacher_mask).view(-1,dT))
                
            teacher_mask = dec_mask.unsqueeze(-1).expand_as(teacher_out.decoder_hidden_states[-1]).bool()
            student_mask = dec_mask.unsqueeze(-1).expand_as(student_out.decoder_hidden_states[-1]).bool()    
            dT = teacher_out.decoder_hidden_states[-1].size(-1)
            dS = student_out.decoder_hidden_states[-1].size(-1)

            nSE = student.config.encoder_layers
            loss[nSE+3] += CSLoss(masked_select(student_out.decoder_hidden_states[0],student_mask).view(-1,dS), \
                                masked_select(teacher_out.decoder_hidden_states[0],teacher_mask).view(-1,dT))
                    
            for i in range(student.config.decoder_layers):
                torch.cuda.empty_cache()
                loss[i+nSE+4] += CSLoss(masked_select(student_out.decoder_hidden_states[i+1],student_mask).view(-1,dS), \
                                        masked_select(teacher_out.decoder_hidden_states[i+1],teacher_mask).view(-1,dT))
        

            del teacher_mask, student_mask, student_out, teacher_out
            nBatch+=1
        
        if(nBatch%1000==0): break
    
    bleu = load("sacrebleu",experiment_id = "MBart200-%d-%d.txt" % (student.config.encoder_layers,student.config.d_model), trust_remote_code=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    for batch in test_dataloader:
        with torch.cuda.amp.autocast():
            batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
            predictions = student.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
            decoded_preds = tokenizer_enro.batch_decode(predictions, skip_special_tokens=True)
            labels = batch['label_ids'].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer_enro.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            bleu.add_batch(predictions = decoded_preds, references = decoded_labels)

    return [l.item()/nBatch for l in loss], bleu.compute()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"






teacher_id = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(teacher_id)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
teacher = MBartForConditionalGeneration.from_pretrained(teacher_id,output_hidden_states = True, output_past = False, use_cache = False)
for param in teacher.parameters(): param.requires_grad = False
student_dim = 640
student_layer = 12
config = MBartConfig(vocab_size = teacher.config.vocab_size, d_model = student_dim, \
                    encoder_layers = student_layer, decoder_layers = student_layer, \
                    encoder_ffn_dim = 4*student_dim, decoder_ffn_dim = 4*student_dim,
                    output_hidden_states = True, output_past = False, use_cache = False)
student = MBartForConditionalGeneration(config)
student.load_state_dict(torch.load("../../Checkpoints/BART/MBart-Encoder-12-640-CKA.pt"))

#student = Create_Student('...',student_dim,student_layer,teacher)

print("Teacher: ", teacher.config)
print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))
print("Student: ", student.config)
print(sum(p.numel() for p in student.parameters()))
teacher.to(device)
student.to(device)


############################ Concatenate XSUM, CNN, CC NEWS & NEWSROOM ##########################

datasets = []

def encode(examples):
    result = {}
    keys = list(examples['translation'].keys())
    temp = tokenizer(examples['translation'][keys[0]], truncation=True, padding='max_length', max_length = 128)
    result['input_ids'] = temp['input_ids']
    result['attention_mask'] = temp['attention_mask']
    temp = tokenizer(examples['translation'][keys[1]], truncation=True, padding='max_length', max_length = 128)
    result['label_ids'] = temp['input_ids']
    result['label_attention_mask'] = temp['attention_mask']
    return result

src_langs = ['ar','hi','zh','tr','en']
#langs = ['ar', 'pl', 'de', 'hi', 'it', 'en', 'he', 'nl', 'vi', 'tr', 'ru', 'ja', 'uk', 'es', 'cs', 'fa', 'ko', 'zh', 'pt', 'fi', 'id', 'sv']
langs = [ 'ar', 'cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'hi', 'it', 'ja', 'ko', \
         'lt', 'lv', 'ka', 'ne', 'nl', 'ro', 'ru', 'si', 'tr', 'vi', 'zh', 'az', 'bn', \
         'fa', 'he', 'id', 'ml', 'mr', 'pl',  'pt', 'sv', \
         'sw', 'ta',  'uk', 'ur', 'xh','sl', 'km']

for src in src_langs:
    for dest in langs:
        if src!=dest and [src,dest]!=['tr','xh'] and [src,dest]!=['en','fr'] and [src,dest]!=['en','ka'] and [src,dest]!=['en','km'] and [src,dest]!=['en','kk'] and [src,dest]!= ['en','es'] and [src,dest]!= ['hi','kk'] and [src,dest]!= ['hi','km'] and [src,dest]!= ['hi','ka'] and [src,dest]!= ['hi','az'] and [src,dest]!= ['hi','xh'] and [src,dest]!= ['zh','kk'] and [src,dest]!= ['zh','az']and [src,dest]!= ['zh','ne'] and [src,dest]!= ['zh','si'] and [src,dest]!= ['zh','mr'] and [src,dest]!= ['zh','ur'] and [src,dest]!= ['zh','xh'] and [src,dest]!= ['zh','ka'] and [src,dest]!= ['zh','km']:
            datasets.append(load_dataset("yhavinga/ccmatrix", src+'-'+dest,split = 'train', streaming=True).map(encode, remove_columns=["translation"]))
            print(src,dest)


dataset = interleave_datasets(datasets,stopping_strategy="all_exhausted").remove_columns(["id","score"]).with_format('torch')


dataset = dataset.shuffle(buffer_size=100_000)
eval_dataset = dataset.take(70000)
batch_size = 64
train_dataloader = DataLoader(dataset.skip(70000), batch_size = batch_size)


tokenizer_enro = MBart50TokenizerFast.from_pretrained(teacher_id, src_lang="en_XX", tgt_lang="ro_RO")
def encode_enro(examples):
    result = {}
    temp = tokenizer_enro(examples['translation']['en'], truncation=True, padding='max_length', max_length = 128)
    result['input_ids'] = temp['input_ids']
    result['attention_mask'] = temp['attention_mask']
    temp = tokenizer_enro(text_target = examples['translation']['ro'], truncation=True, padding='max_length', max_length = 128)
    result['label_ids'] = temp['input_ids']
    result['label_attention_mask'] = temp['attention_mask']
    return result

test_dataset = load_dataset('wmt16', 'ro-en', split= 'test').map(encode_enro, remove_columns = ['translation'])
test_dataset.set_format('torch')

############################ Training Starts ############################
no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 5e-4,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    }
]

torch.manual_seed(99)
optimizer = AdamW(optimizer_grouped_parameters , lr=3e-5, betas = (0.9,0.999), eps = 1e-7)
f = open("../../Checkpoints/LOGIT-KD-CKA-MBart200-%d-%d.txt" % (student_layer,student_dim), "w+", buffering= 50)
f1 = open("../../Checkpoints/LOGIT-KD-Eval-CKA-MBart200-%d-%d.txt" % (student_layer,student_dim), "w+", buffering= 1)
from transformers import get_scheduler
vaild_sample_interval = 5000
num_training_steps = 30 * vaild_sample_interval
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
STLoss = nn.KLDivLoss(reduction = 'batchmean')
CSLoss = CKALoss(eps = 1e-8)
pad_token_id = tokenizer.pad_token_id
CELoss = nn.CrossEntropyLoss(ignore_index = pad_token_id)

scaler = torch.cuda.amp.GradScaler()
teacher.eval()
temperature = 1.0
lambdaH = 1.0

    
nBatch = 0
for batch in train_dataloader:
    with torch.cuda.amp.autocast():
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
        decoder_input_ids = shift_tokens_right(batch['label_ids'], pad_token_id, pad_token_id)
        dec_mask = decoder_input_ids.ne(pad_token_id)
        teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids']) 
        loss = [0]*(student.config.encoder_layers + student.config.decoder_layers + 4)
            #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)
        student.train()
        with torch.enable_grad():
            student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids'])   
            torch.cuda.empty_cache()
            dV = student_out.logits.size(-1)
            logit_mask = batch['label_attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()
            SL = masked_select(student_out.logits,logit_mask).view(-1,dV)
            loss[0] = CELoss(SL,masked_select(batch['label_ids'],batch['label_attention_mask'].bool()))/lambdaH
            loss[1] = ((temperature)**2)* STLoss(F.log_softmax(SL/ temperature,dim=-1), F.softmax(masked_select(teacher_out.logits,logit_mask).view(-1,dV)/temperature, dim=-1))/lambdaH
                #loss[2] = CELoss(rearrange(student_out.logits[:,:pseudo.size(1),:],'a b c -> (a b) c'),rearrange(pseudo, 'a b -> (a b)'))

            teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.encoder_hidden_states[-1]).bool()
            student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.encoder_hidden_states[-1]).bool()
            dT = teacher_out.encoder_hidden_states[-1].size(-1)
            dS = student_out.encoder_hidden_states[-1].size(-1)                
            
            loss[2] = CSLoss(masked_select(student_out.encoder_hidden_states[0],student_mask).view(-1,dS), \
                                    masked_select(teacher_out.encoder_hidden_states[0],teacher_mask).view(-1,dT))
                
            for i in range(student.config.encoder_layers):
                torch.cuda.empty_cache()
                loss[i+3] = CSLoss(masked_select(student_out.encoder_hidden_states[i+1],student_mask).view(-1,dS), \
                                            masked_select(teacher_out.encoder_hidden_states[i+1],teacher_mask).view(-1,dT))
                
            teacher_mask = dec_mask.unsqueeze(-1).expand_as(teacher_out.decoder_hidden_states[-1]).bool()
            student_mask = dec_mask.unsqueeze(-1).expand_as(student_out.decoder_hidden_states[-1]).bool()    
            dT = teacher_out.decoder_hidden_states[-1].size(-1)
            dS = student_out.decoder_hidden_states[-1].size(-1)

            nSE = student.config.encoder_layers
            loss[nSE+3] = CSLoss(masked_select(student_out.decoder_hidden_states[0],student_mask).view(-1,dS), \
                                    masked_select(teacher_out.decoder_hidden_states[0],teacher_mask).view(-1,dT))
                    
            for i in range(student.config.decoder_layers):
                torch.cuda.empty_cache()
                loss[i+nSE+4] = CSLoss(masked_select(student_out.decoder_hidden_states[i+1],student_mask).view(-1,dS), \
                                            masked_select(teacher_out.decoder_hidden_states[i+1],teacher_mask).view(-1,dT))

            del teacher_mask, student_mask, student_out, teacher_out
            loss_sum = sum(loss)
            torch.cuda.empty_cache()
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
                
                

        f.write(str(['%.3f' % l.item() for l in loss])+'\n')
        nBatch+=1
        if(nBatch%vaild_sample_interval==0):
            torch.save(student.state_dict(),"../../Checkpoints/MBart200-%d-%d-All.pt" % (student_layer,student_dim))
            loss, result = Eval_Student(student,teacher,eval_dataset,test_dataset, batch_size)
            f1.write(str('%.3f ' % result['score'])  + str(['%.3f' % l for l in loss]) + '\n')
            f1.flush()
            
    


torch.save(student.state_dict(),"../../Checkpoints/MBart200-%d-%d-All.pt" % (student_layer,student_dim))

f.close()
f1.close()
