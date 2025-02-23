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
from transformers import AutoTokenizer, MBartTokenizer, MBart50TokenizerFast, MBart50Tokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
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






def Eval_Student(student,teacher,test_dataset, batch_size = 32):
    bleu = load("sacrebleu",experiment_id = "MBart-nH-%s-%d-%d.txt" % (src_lang,student.config.encoder_layers,student.config.d_model), trust_remote_code=True)

    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last = True)
    student.eval()
    teacher.eval()

    loss = [0]*2 #(student.config.encoder_layers + student.config.decoder_layers + 4)
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
            loss[1] += ((temperature)**2)*STLoss(F.log_softmax(masked_select(teacher_out.logits,logit_mask).view(-1,dV)/ temperature, dim=-1), F.softmax(SL/ temperature,dim=-1))
            #loss[2] += CELoss(rearrange(student_out.logits[:,:pseudo.size(1),:],'a b c -> (a b) c'),rearrange(pseudo, 'a b -> (a b)'))

            # teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.encoder_hidden_states[-1]).bool()
            # student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.encoder_hidden_states[-1]).bool()
            # dT = teacher_out.encoder_hidden_states[-1].size(-1)
            # dS = student_out.encoder_hidden_states[-1].size(-1)                
            
            # loss[2] += CSLoss(Eh(masked_select(student_out.encoder_hidden_states[0],student_mask).view(-1,dS)), \
            #                         masked_select(teacher_out.encoder_hidden_states[0],teacher_mask).view(-1,dT))
                
            # for i in range(student.config.encoder_layers):
            #     torch.cuda.empty_cache()
            #     loss[i+3] += CSLoss(Eh(masked_select(student_out.encoder_hidden_states[i+1],student_mask).view(-1,dS)), \
            #                                 masked_select(teacher_out.encoder_hidden_states[d*(i+1)],teacher_mask).view(-1,dT))
                
            # teacher_mask = dec_mask.unsqueeze(-1).expand_as(teacher_out.decoder_hidden_states[-1]).bool()
            # student_mask = dec_mask.unsqueeze(-1).expand_as(student_out.decoder_hidden_states[-1]).bool()    
            # dT = teacher_out.decoder_hidden_states[-1].size(-1)
            # dS = student_out.decoder_hidden_states[-1].size(-1)

            # nSE = student.config.encoder_layers
            # loss[nSE+3] += CSLoss(Eh(masked_select(student_out.decoder_hidden_states[0],student_mask).view(-1,dS)), \
            #                     masked_select(teacher_out.decoder_hidden_states[0],teacher_mask).view(-1,dT))
                    
            # for i in range(student.config.decoder_layers):
            #     torch.cuda.empty_cache()
            #     loss[i+nSE+4] += CSLoss(Eh(masked_select(student_out.decoder_hidden_states[i+1],student_mask).view(-1,dS)), \
            #                             masked_select(teacher_out.decoder_hidden_states[d*(i+1)],teacher_mask).view(-1,dT))
        
            predictions = student.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],num_beams=5, early_stopping = True, max_length = 127)
            decoded_preds = tokenizer.batch_decode(predictions.sequences, skip_special_tokens=True)
            labels = batch['label_ids'].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            bleu.add_batch(predictions = decoded_preds, references = decoded_labels)
            
            nBatch+=1
            
    
    return [l.item()/nBatch for l in loss], bleu.compute()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

src_lang = 'ro'
def encode(examples):
    result = {}
    temp = tokenizer(examples['translation']['en'], truncation=True, padding='max_length', max_length = 128)
    result['input_ids'] = temp['input_ids']
    result['attention_mask'] = temp['attention_mask']
    temp = tokenizer(examples['translation'][src_lang], truncation=True, padding='max_length', max_length = 128)
    result['label_ids'] = temp['input_ids']
    result['label_attention_mask'] = temp['attention_mask']
    return result




# teacher_id = "context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr"
teacher_id = "facebook/mbart-large-en-ro"

tokenizer = MBartTokenizer.from_pretrained(teacher_id)
teacher = MBartForConditionalGeneration.from_pretrained(teacher_id,output_hidden_states = True, output_past = False, use_cache = False)

print("Teacher: ", teacher.config)
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

############################ Concatenate XSUM, CNN, CC NEWS & NEWSROOM ##########################
torch.manual_seed(42)
indices = np.random.randint(0,40000000,size=(3000000,))
# dataset = load_dataset('iwslt2017', 'iwslt2017-fr-en')

# wmt_fr_en = load_dataset('wmt14','fr-en', split = 'train').select(indices)
# dataset['train'] = concatenate_datasets([dataset['train'],wmt_fr_en])
# tokenized_datasets.save_to_disk('/scratch/punim0478/sayantand/Dump/EN_FR')
# tokenized_datasets = load_from_disk('/scratch/punim0478/sayantand/Dump/EN_FR')
# print(tokenized_datasets)
# dataset = load_dataset('wmt16', src_lang+'-en', download_mode = "reuse_dataset_if_exists")
# tokenized_datasets = dataset.map(encode, remove_columns = ['translation'], num_proc = 32)
# tokenized_datasets.set_format('torch')
tokenized_datasets = load_from_disk('EN_RO')

batch_size = 32
train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
test_dataset = tokenized_datasets['test'] if src_lang == 'fr' else tokenized_datasets['validation']
valid_per_batch = 10 if src_lang == 'fr' else 5
num_epochs= 3 if src_lang == 'fr' else 5

vaild_sample_interval = len(train_dataloader)//valid_per_batch
print(vaild_sample_interval)

from transformers import get_scheduler

STLoss = nn.KLDivLoss(reduction = 'batchmean')
CSLoss = nn.MSELoss() #CKALoss(eps = 1e-8)
pad_token_id = tokenizer.pad_token_id
CELoss = nn.CrossEntropyLoss(ignore_index = pad_token_id)

scaler = torch.cuda.amp.GradScaler()
teacher.eval()
temperature = 1.0
lambdaH = 1.0


D = [384,512]
L = [6,6]

for i in range(len(D)):
    student_dim = D[i]
    student_layer = L[i]

    d = 2 if L[i]==6 else 1

    config = MBartConfig(vocab_size = 250054, d_model = student_dim, \
                        encoder_layers = student_layer, decoder_layers = student_layer, \
                        encoder_ffn_dim = 4*student_dim, decoder_ffn_dim = 4*student_dim,
                        num_beams=5, early_stopping = True, output_hidden_states = True, output_past = False, use_cache = True)
    student = MBartForConditionalGeneration(config)
    student.load_state_dict(torch.load("/data/projects/punim0478/sayantand/Checkpoints/BART/August_2024/MBart-TB-%d-%d-MC4.pt" % (student_layer,student_dim)))

    old_vocab_size = student.config.vocab_size
    temp_shared =  student.model.shared.weight.clone()
    temp_encoder = student.model.encoder.embed_tokens.weight.clone()
    temp_decoder = student.model.decoder.embed_tokens.weight.clone()
    temp_lmhead =  student.lm_head.weight.clone()

        ############### Resize Embedding to MBART25 and Copy the Old Embedding
        
    student.resize_token_embeddings(teacher.config.vocab_size)
    student.model.shared.weight.copy_(temp_shared[:teacher.config.vocab_size])
    student.model.encoder.embed_tokens.weight.copy_(temp_encoder[:teacher.config.vocab_size])
    student.model.decoder.embed_tokens.weight.copy_(temp_decoder[:teacher.config.vocab_size])
    student.lm_head.weight.copy_(temp_lmhead[:teacher.config.vocab_size])

    Eh = nn.Linear(student.config.d_model,teacher.config.d_model)
    Eh.to(device)

    print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))
    print("Student: ", student.config)
    print(sum(p.numel() for p in student.parameters()))
    teacher.to(device)
    student.to(device)

    ############################ Training Starts ############################
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 1e-2,
        },
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters , lr=1e-4, betas = (0.9,0.999), eps = 1e-8)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    optimizer1 = AdamW(Eh.parameters() , lr=1e-4, betas = (0.9,0.999), eps = 1e-6, weight_decay = 5e-4)
    lr_scheduler1 = get_scheduler(name="linear", optimizer=optimizer1, num_warmup_steps=0, num_training_steps = num_training_steps)

    f = open("../../Checkpoints/LOGIT-KD-nH-MBart-%s-%d-%d.txt" % (src_lang,student_layer,student_dim), "w+", buffering= 50)
    f1 = open("../../Checkpoints/LOGIT-KD-Eval-nH-MBart-%s-%d-%d.txt" % (src_lang,student_layer,student_dim), "w+", buffering= 1)


    for epoch in range(num_epochs):
        student.train()
        nBatch = 0
        for batch in train_dataloader:
            with torch.cuda.amp.autocast():
                torch.cuda.empty_cache()
                batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
                decoder_input_ids = shift_tokens_right(batch['label_ids'], pad_token_id, pad_token_id)
                dec_mask = decoder_input_ids.ne(pad_token_id)
                teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids']) 
                loss = [0]*2 #(student.config.encoder_layers + student.config.decoder_layers + 4)
                #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)

                with torch.enable_grad():
                    student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids'])  
                    torch.cuda.empty_cache()
                    dV = student_out.logits.size(-1)
                    logit_mask = batch['label_attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()
                    SL = masked_select(student_out.logits,logit_mask).view(-1,dV)
                    loss[0] = CELoss(SL,masked_select(batch['label_ids'],batch['label_attention_mask'].bool()))/lambdaH
                    loss[1] = ((temperature)**2)*STLoss(F.log_softmax(masked_select(teacher_out.logits,logit_mask).view(-1,dV)/ temperature, dim=-1), F.softmax(SL/ temperature,dim=-1))
                    #loss[2] = CELoss(rearrange(student_out.logits[:,:pseudo.size(1),:],'a b c -> (a b) c'),rearrange(pseudo, 'a b -> (a b)'))

                    # teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.encoder_hidden_states[-1]).bool()
                    # student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.encoder_hidden_states[-1]).bool()
                    # dT = teacher_out.encoder_hidden_states[-1].size(-1)
                    # dS = student_out.encoder_hidden_states[-1].size(-1)                
                
                    # loss[2] = CSLoss(Eh(masked_select(student_out.encoder_hidden_states[0],student_mask).view(-1,dS)), \
                    #                     masked_select(teacher_out.encoder_hidden_states[0],teacher_mask).view(-1,dT))
                    
                    # for i in range(student.config.encoder_layers):
                    #     torch.cuda.empty_cache()
                    #     loss[i+3] = CSLoss(Eh(masked_select(student_out.encoder_hidden_states[i+1],student_mask).view(-1,dS)), \
                    #                             masked_select(teacher_out.encoder_hidden_states[d*(i+1)],teacher_mask).view(-1,dT))
                    
                    # teacher_mask = dec_mask.unsqueeze(-1).expand_as(teacher_out.decoder_hidden_states[-1]).bool()
                    # student_mask = dec_mask.unsqueeze(-1).expand_as(student_out.decoder_hidden_states[-1]).bool()    
                    # dT = teacher_out.decoder_hidden_states[-1].size(-1)
                    # dS = student_out.decoder_hidden_states[-1].size(-1)

                    # nSE = student.config.encoder_layers
                    # loss[nSE+3] = CSLoss(Eh(masked_select(student_out.decoder_hidden_states[0],student_mask).view(-1,dS)), \
                    #                     masked_select(teacher_out.decoder_hidden_states[0],teacher_mask).view(-1,dT))
                        
                    # for i in range(student.config.decoder_layers):
                    #     torch.cuda.empty_cache()
                    #     loss[i+nSE+4] = CSLoss(Eh(masked_select(student_out.decoder_hidden_states[i+1],student_mask).view(-1,dS)), \
                    #                             masked_select(teacher_out.decoder_hidden_states[d*(i+1)],teacher_mask).view(-1,dT))
                    
                    scaler.scale(sum(loss)).backward()
                    scaler.step(optimizer)
                    #scaler.step(optimizer1)
                    scaler.update()
                    lr_scheduler.step()
                    #lr_scheduler1.step()
                    optimizer.zero_grad()
                    #optimizer1.zero_grad()
                    f.write(str(['%.3f' % l.item() for l in loss])+'\n')
                    
                    

            nBatch+=1
            if(nBatch%vaild_sample_interval==0):
                torch.save(student.state_dict(),"../../Checkpoints/MBart-%s-%d-%d-nH.pt" % (src_lang,student_layer,student_dim))
                loss, result = Eval_Student(student,teacher,test_dataset, 16)
                f1.write(str('%.3f ' % result['score'])  + str(['%.3f' % l for l in loss]) + '\n')
                f1.flush()
                
        


    torch.save(student.state_dict(),"../../Checkpoints/MBart-%s-%d-%d-nH.pt" % (src_lang,student_layer,student_dim))

    f.close()
    f1.close()
