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
import os, sys
    
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

def Create_Student(student_id,student_dim,student_layer,teacher):
    config = MBartConfig(vocab_size = teacher.config.vocab_size, d_model = student_dim, \
                    encoder_layers = student_layer, decoder_layers = student_layer, \
                    encoder_ffn_dim = 4*student_dim, decoder_ffn_dim = 4*student_dim,
                    output_hidden_states = True, output_past = False, use_cache = False)
    student = MBartForConditionalGeneration(config)
    print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))
    student_int_size = student.config.encoder_ffn_dim    
    student.model.shared.weight.copy_(teacher.model.shared.weight[:,:student_dim].clone())
    student.model.encoder.embed_tokens.weight.copy_(teacher.model.encoder.embed_tokens.weight[:,:student_dim].clone())
    student.model.decoder.embed_tokens.weight.copy_(teacher.model.decoder.embed_tokens.weight[:,:student_dim].clone())
        
    for i in range(student.config.encoder_layers):
        student.model.encoder.layers[i].self_attn.k_proj.weight.copy_(teacher.model.encoder.layers[i].self_attn.k_proj.weight[:student_dim,:student_dim].clone())
        student.model.encoder.layers[i].self_attn.k_proj.bias.copy_(teacher.model.encoder.layers[i].self_attn.k_proj.bias[:student_dim].clone())
        student.model.encoder.layers[i].self_attn.v_proj.weight.copy_(teacher.model.encoder.layers[i].self_attn.v_proj.weight[:student_dim,:student_dim].clone())
        student.model.encoder.layers[i].self_attn.v_proj.bias.copy_(teacher.model.encoder.layers[i].self_attn.v_proj.bias[:student_dim].clone())
        student.model.encoder.layers[i].self_attn.q_proj.weight.copy_(teacher.model.encoder.layers[i].self_attn.q_proj.weight[:student_dim,:student_dim].clone())
        student.model.encoder.layers[i].self_attn.q_proj.bias.copy_(teacher.model.encoder.layers[i].self_attn.q_proj.bias[:student_dim].clone())
        student.model.encoder.layers[i].self_attn.out_proj.weight.copy_(teacher.model.encoder.layers[i].self_attn.out_proj.weight[:student_dim,:student_dim].clone())
        student.model.encoder.layers[i].self_attn.out_proj.bias.copy_(teacher.model.encoder.layers[i].self_attn.out_proj.bias[:student_dim].clone())
    
        student.model.encoder.layers[i].self_attn_layer_norm.weight.copy_(teacher.model.encoder.layers[i].self_attn_layer_norm.weight[:student_dim].clone())
        student.model.encoder.layers[i].self_attn_layer_norm.bias.copy_(teacher.model.encoder.layers[i].self_attn_layer_norm.bias[:student_dim].clone())
    
        student.model.encoder.layers[i].fc1.weight.copy_(teacher.model.encoder.layers[i].fc1.weight[:student_int_size, :student_dim].clone())
        student.model.encoder.layers[i].fc1.bias.copy_(teacher.model.encoder.layers[i].fc1.bias[:student_int_size].clone())
        student.model.encoder.layers[i].fc2.weight.copy_(teacher.model.encoder.layers[i].fc2.weight[:student_dim, :student_int_size].clone())
        student.model.encoder.layers[i].fc2.bias.copy_(teacher.model.encoder.layers[i].fc2.bias[:student_dim,].clone())    
        student.model.encoder.layers[i].final_layer_norm.weight.copy_(teacher.model.encoder.layers[i].final_layer_norm.weight[:student_dim].clone())
        student.model.encoder.layers[i].final_layer_norm.bias.copy_(teacher.model.encoder.layers[i].final_layer_norm.bias[:student_dim].clone())
        
    student.model.encoder.layernorm_embedding.weight.copy_(teacher.model.encoder.layernorm_embedding.weight[:student_dim].clone())
    student.model.encoder.layernorm_embedding.bias.copy_(teacher.model.encoder.layernorm_embedding.bias[:student_dim].clone())
    
    for i in range(student.config.decoder_layers):
        student.model.decoder.layers[i].self_attn.k_proj.weight.copy_(teacher.model.decoder.layers[i].self_attn.k_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].self_attn.k_proj.bias.copy_(teacher.model.decoder.layers[i].self_attn.k_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].self_attn.v_proj.weight.copy_(teacher.model.decoder.layers[i].self_attn.v_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].self_attn.v_proj.bias.copy_(teacher.model.decoder.layers[i].self_attn.v_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].self_attn.q_proj.weight.copy_(teacher.model.decoder.layers[i].self_attn.q_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].self_attn.q_proj.bias.copy_(teacher.model.decoder.layers[i].self_attn.q_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].self_attn.out_proj.weight.copy_(teacher.model.decoder.layers[i].self_attn.out_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].self_attn.out_proj.bias.copy_(teacher.model.decoder.layers[i].self_attn.out_proj.bias[:student_dim].clone())
    
        student.model.decoder.layers[i].self_attn_layer_norm.weight.copy_(teacher.model.decoder.layers[i].self_attn_layer_norm.weight[:student_dim].clone())
        student.model.decoder.layers[i].self_attn_layer_norm.bias.copy_(teacher.model.decoder.layers[i].self_attn_layer_norm.bias[:student_dim].clone())
    
        student.model.decoder.layers[i].encoder_attn.k_proj.weight.copy_(teacher.model.decoder.layers[i].encoder_attn.k_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.k_proj.bias.copy_(teacher.model.decoder.layers[i].encoder_attn.k_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.v_proj.weight.copy_(teacher.model.decoder.layers[i].encoder_attn.v_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.v_proj.bias.copy_(teacher.model.decoder.layers[i].encoder_attn.v_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.q_proj.weight.copy_(teacher.model.decoder.layers[i].encoder_attn.q_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.q_proj.bias.copy_(teacher.model.decoder.layers[i].encoder_attn.q_proj.bias[:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.out_proj.weight.copy_(teacher.model.decoder.layers[i].encoder_attn.out_proj.weight[:student_dim,:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn.out_proj.bias.copy_(teacher.model.decoder.layers[i].encoder_attn.out_proj.bias[:student_dim].clone())
    
        student.model.decoder.layers[i].encoder_attn_layer_norm.weight.copy_(teacher.model.decoder.layers[i].encoder_attn_layer_norm.weight[:student_dim].clone())
        student.model.decoder.layers[i].encoder_attn_layer_norm.bias.copy_(teacher.model.decoder.layers[i].encoder_attn_layer_norm.bias[:student_dim].clone())
    
        student.model.decoder.layers[i].fc1.weight.copy_(teacher.model.decoder.layers[i].fc1.weight[:student_int_size, :student_dim].clone())
        student.model.decoder.layers[i].fc1.bias.copy_(teacher.model.decoder.layers[i].fc1.bias[:student_int_size].clone())
        student.model.decoder.layers[i].fc2.weight.copy_(teacher.model.decoder.layers[i].fc2.weight[:student_dim, :student_int_size].clone())
        student.model.decoder.layers[i].fc2.bias.copy_(teacher.model.decoder.layers[i].fc2.bias[:student_dim,].clone())
        student.model.decoder.layers[i].final_layer_norm.weight.copy_(teacher.model.decoder.layers[i].final_layer_norm.weight[:student_dim].clone())
        student.model.decoder.layers[i].final_layer_norm.bias.copy_(teacher.model.decoder.layers[i].final_layer_norm.bias[:student_dim].clone())
        
    student.model.decoder.layernorm_embedding.weight.copy_(teacher.model.encoder.layernorm_embedding.weight[:student_dim].clone())
    student.model.decoder.layernorm_embedding.bias.copy_(teacher.model.encoder.layernorm_embedding.bias[:student_dim].clone())
    
    student.lm_head.weight.copy_(teacher.lm_head.weight[:,:student_dim])


    return student




def Eval_Student(student,teacher,eval_dataset, batch_size = 32):

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    student.eval()
    teacher.eval()

    loss = [0]*(student.config.encoder_layers + student.config.decoder_layers + 4)
    nBatch = 0
    for batch in eval_dataloader:
        with torch.cuda.amp.autocast():
            batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
            decoder_input_ids = shift_tokens_right(batch['input_ids'], pad_token_id, pad_token_id)
            dec_mask = decoder_input_ids.ne(pad_token_id)
            torch.cuda.empty_cache()
            student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])  
            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])    
            #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)

            dV = student_out.logits.size(-1)
            logit_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()
            SL = masked_select(student_out.logits,logit_mask).view(-1,dV)
            loss[0] += CELoss(SL,masked_select(batch['input_ids'],batch['attention_mask'].bool()))
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

    return [l.item()/nBatch for l in loss]



os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



teacher_id = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(teacher_id)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
teacher = MBartForConditionalGeneration.from_pretrained(teacher_id,output_hidden_states = True, output_past = False, use_cache = False)
for param in teacher.parameters(): param.requires_grad = False
student_dim = int(sys.argv[1])
student_layer = 12
# config = MBartConfig(vocab_size = teacher.config.vocab_size, d_model = student_dim, \
#                     encoder_layers = student_layer, decoder_layers = student_layer, \
#                     encoder_ffn_dim = 4*student_dim, decoder_ffn_dim = 4*student_dim,
#                     output_hidden_states = True, output_past = False, use_cache = False)
# student = MBartForConditionalGeneration(config)
#student.load_state_dict(torch.load("../../Checkpoints/BART/MBart-Encoder-12-640-CKA.pt"))

student = Create_Student('...',student_dim,student_layer,teacher)

print("Teacher: ", teacher.config)
print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))
print("Student: ", student.config)
print(sum(p.numel() for p in student.parameters()))
teacher.to(device)
student.to(device)


############################ Concatenate XSUM, CNN, CC NEWS & NEWSROOM ##########################
langs = [ 'ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', \
         'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN', 'af_ZA', 'az_AZ', 'bn_IN', \
         'fa_IR', 'he_IL', 'id_ID', 'ka_GE', 'km_KH', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN', 'pl_PL', 'ps_AF', 'pt_XX', 'sv_SE', \
         'sw_KE', 'ta_IN', 'te_IN', 'th_TH', 'uk_UA', 'ur_PK', 'xh_ZA', 'gl_ES', 'sl_SI']


tokenizer = MBart50TokenizerFast.from_pretrained(teacher_id)
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = 512, return_tensors="pt")


datasets_train = []
datasets_valid = []

#dataset = load_dataset('mc4', 'en', streaming=True)



for lang in langs:
    key,_ = lang.split('_')
    if(key =='he'): key = 'iw'

    datasets_train.append(load_dataset('mc4', key, split = 'train',streaming=True))
    datasets_valid.append(load_dataset('mc4', key, split = 'validation',streaming=True))

dataset_train = interleave_datasets(datasets_train).remove_columns(['timestamp', 'url']).with_format('torch')
dataset_valid = interleave_datasets(datasets_valid).remove_columns(['timestamp', 'url']).with_format('torch')
    
tokenized_dataset_train = dataset_train.shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched = True)
tokenized_dataset_valid = dataset_valid.shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched = True)

batch_size = 8
train_dataloader = DataLoader(tokenized_dataset_train, batch_size = batch_size)


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
f = open("../../Checkpoints/LOGIT-KD-CKA-MBart-%d-%d-MC4.txt" % (student_layer,student_dim), "w+", buffering= 50)
f1 = open("../../Checkpoints/LOGIT-KD-Eval-CKA-MBart-%d-%d-MC4.txt" % (student_layer,student_dim), "w+", buffering= 1)
from transformers import get_scheduler, get_cosine_schedule_with_warmup
vaild_sample_interval = 5000
num_epochs = 50 
scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer,num_warmup_steps=0, num_training_steps=num_epochs*vaild_sample_interval)

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
        decoder_input_ids = shift_tokens_right(batch['input_ids'], pad_token_id, pad_token_id)
        dec_mask = decoder_input_ids.ne(pad_token_id)
        teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
        dV = teacher_out.logits.size(-1)

        logit_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.logits).bool()
        TL = masked_select(teacher_out.logits,logit_mask).view(-1,dV)

        loss = [0]*(student.config.encoder_layers + student.config.decoder_layers + 4)
            #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)
        student.train()
        with torch.enable_grad():
            student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])   
            torch.cuda.empty_cache()
            SL = masked_select(student_out.logits,logit_mask).view(-1,dV)
            loss[0] = CELoss(SL,masked_select(batch['input_ids'],batch['attention_mask'].bool()))/lambdaH
            loss[1] = ((temperature)**2)* STLoss(F.log_softmax(SL/ temperature,dim=-1), F.softmax(TL /temperature, dim=-1))/lambdaH
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
            scaler.scale(loss_sum).backward()

            nBatch+=1
            if(nBatch%4 ==0):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                f.write(str(['%.3f' % l.item() for l in loss])+'\n')
                
        if(nBatch%vaild_sample_interval==0):
            torch.save(student.state_dict(),"../../Checkpoints/MBart-CKA-%d-%d-MC4.pt" % (student_layer,student_dim))
            result = Eval_Student(student,teacher,tokenized_dataset_valid, batch_size)
            f1.write(str(['%.3f' % l for l in loss]) + '\n')
            f1.flush()
            
    


torch.save(student.state_dict(),"../../Checkpoints/MBart-CKA-%d-%d-MC4.pt" % (student_layer,student_dim))

f.close()
f1.close()
