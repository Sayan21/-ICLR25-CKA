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
import evaluate
from torch import masked_select
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, DistilBertForMaskedLM, BertConfig, DistilBertConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import os

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

def Eval_Student(student,teacher,test_dataset, temp = 2.0, batch_size = 32):
    STLoss = torch.nn.KLDivLoss(reduction = 'batchmean')
    CSLoss = CKALoss(eps = 1e-8)
    eval_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16)
    student.eval()
    teacher.eval()
    n_layers = student.config.num_hidden_layers
    loss = [0]*(n_layers+2)
    nBatch = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            torch.cuda.empty_cache()
            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
            student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
            torch.cuda.empty_cache()
            logit_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()  
            SL = masked_select(student_out.logits,logit_mask).view(-1,30522) 
            loss[0] += (temp**2)*STLoss(F.log_softmax(SL,dim=-1),F.softmax(masked_select(teacher_out.logits,logit_mask).view(-1,30522), dim=-1))
                
            torch.cuda.empty_cache()
            teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.hidden_states[-1]).bool()
            student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.hidden_states[-1]).bool()
            dT = teacher_out.hidden_states[-1].size(-1)
            dS = student_out.hidden_states[-1].size(-1)                

            loss[1] += CSLoss(masked_select(student_out.hidden_states[0],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[0],teacher_mask).view(-1,dT))

            for i in range(4):
                for j in range(4):
                    if(j==3):
                        loss[i*4+5] += CSLoss(masked_select(student_out.hidden_states[4*(i+1)],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[3*(i+1)],teacher_mask).view(-1,dT))
                    else:
                        loss[i*4+j+2] += CSLoss(masked_select(student_out.hidden_states[i*4+j+1],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[i*3+j+1],teacher_mask).view(-1,dT))
                

            
            nBatch+=1
            if(nBatch%2500==0): break
            
    
    return [l.item()/nBatch for l in loss]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



student_id = "distilbert-base-uncased"  #"huawei-noah/TinyBERT_General_4L_312D"
teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length",truncation=True)

batch_size = 32
dataset = load_dataset('wikipedia', '20220301.en',split = 'train')
print(dataset)
tokenized_datasets = dataset.map(tokenize_function, remove_columns = ["text", "url", "id", "title"], num_proc = 24)
tokenized_datasets.set_format("torch")
train_dataset, test_dataset = torch.utils.data.random_split(tokenized_datasets, [6400000, len(tokenized_datasets) - 6400000])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

      

teacher = BertForMaskedLM.from_pretrained(teacher_id,output_hidden_states = True)
for param in teacher.parameters(): param.requires_grad = False
student_dim = 640
n_layers = 16
student_int_size = 3072
config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=n_layers, \
                                    intermediate_size = student_int_size, num_attention_heads=n_layers,output_hidden_states = True)
student = BertForMaskedLM(config)

print("Number of Teacher Parameters: ", sum(p.numel() for p in teacher.parameters()))
print("Number of Student Parameters: ", sum(p.numel() for p in student.parameters()))

teacher.to(device)
student.to(device)

student.bert.embeddings.word_embeddings.weight.copy_(teacher.bert.embeddings.word_embeddings.weight[:,:student_dim].detach().clone())
student.bert.embeddings.position_embeddings.weight.copy_(teacher.bert.embeddings.position_embeddings.weight[:student_dim,:student_dim].detach().clone())
student.bert.embeddings.token_type_embeddings.weight.copy_(teacher.bert.embeddings.token_type_embeddings.weight[:,:student_dim].detach().clone())


for i in range(4):
    for j in range(4):
        if(j==3):
            [si,ti] = [4*(i+1)-1,3*(i+1)-1]
        else:
            [si,ti] = [i*4+j,i*3+j]
        student.bert.encoder.layer[si].attention.self.query.weight.copy_(teacher.bert.encoder.layer[ti].attention.self.query.weight[:student_dim,:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.self.query.bias.copy_(teacher.bert.encoder.layer[ti].attention.self.query.bias[:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.self.key.weight.copy_(teacher.bert.encoder.layer[ti].attention.self.key.weight[:student_dim,:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.self.key.bias.copy_(teacher.bert.encoder.layer[ti].attention.self.key.bias[:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.self.value.weight.copy_(teacher.bert.encoder.layer[ti].attention.self.value.weight[:student_dim,:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.self.value.bias.copy_(teacher.bert.encoder.layer[ti].attention.self.value.bias[:student_dim].detach().clone())
    
        
        student.bert.encoder.layer[si].attention.output.dense.weight.copy_(teacher.bert.encoder.layer[ti].attention.output.dense.weight[:student_dim,:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.output.dense.bias.copy_(teacher.bert.encoder.layer[ti].attention.output.dense.bias[:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[ti].attention.output.LayerNorm.weight[:student_dim].detach().clone())
        student.bert.encoder.layer[si].attention.output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[ti].attention.output.LayerNorm.bias[:student_dim].detach().clone())
    
        student.bert.encoder.layer[si].intermediate.dense.weight.copy_(teacher.bert.encoder.layer[ti].intermediate.dense.weight[:student_int_size,:student_dim].detach().clone())
        student.bert.encoder.layer[si].intermediate.dense.bias.copy_(teacher.bert.encoder.layer[ti].intermediate.dense.bias[:student_int_size].detach().clone())
    
        student.bert.encoder.layer[si].output.dense.weight.copy_(teacher.bert.encoder.layer[ti].output.dense.weight[:student_dim,:student_int_size].detach().clone())
        student.bert.encoder.layer[si].output.dense.bias.copy_(teacher.bert.encoder.layer[ti].output.dense.bias[:student_dim].detach().clone())
        student.bert.encoder.layer[si].output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[i].output.LayerNorm.weight[:student_dim].detach().clone())
        student.bert.encoder.layer[si].output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[ti].output.LayerNorm.bias[:student_dim].detach().clone())
        print(si,ti)
student.cls.predictions.transform.dense.weight.copy_(teacher.cls.predictions.transform.dense.weight[:student_dim,:student_dim].detach().clone())
student.cls.predictions.transform.dense.bias.copy_(teacher.cls.predictions.transform.dense.bias[:student_dim].detach().clone())

student.cls.predictions.decoder.weight.copy_(teacher.cls.predictions.decoder.weight[:,:student_dim].detach().clone())
student.cls.predictions.decoder.bias.copy_(teacher.cls.predictions.decoder.bias.detach().clone())

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay ) and p.requires_grad],
     "weight_decay": 5e-4,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    }
]

optimizer = AdamW(optimizer_grouped_parameters , lr=1e-4, betas = (0.9,0.999), eps = 1e-8)
f = open("../../Checkpoints/BERT/LOGIT-KD-CKA-EMB-16-%d-Wiki.txt" % student_dim, "w+", buffering= 50)
f1 = open("../../Checkpoints/BERT/LOGIT-KD-Eval-CKA-EMB-16-%d-Wiki.txt" % student_dim, "w+", buffering= 1)
from transformers import get_scheduler
num_epochs=30
num_batch_per_epoch = 10000
num_training_steps = num_epochs * num_batch_per_epoch
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps = num_training_steps)
CELoss = torch.nn.CrossEntropyLoss()
STLoss = torch.nn.KLDivLoss(reduction = 'batchmean')
CSLoss = CKALoss(eps = 1e-8)
scaler = torch.cuda.amp.GradScaler()
temp = 1.0


for epoch in range(num_epochs):
    torch.cuda.empty_cache()

    student.train()
    nBatch = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            loss = [0]*(n_layers+2)
            torch.cuda.empty_cache()
            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
            with torch.enable_grad():
                student_out = student(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
                torch.cuda.empty_cache()
                logit_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.logits).bool()  
                SL = masked_select(student_out.logits,logit_mask).view(-1,30522) 
                loss[0] = (temp**2)*STLoss(F.log_softmax(SL,dim=-1),F.softmax(masked_select(teacher_out.logits,logit_mask).view(-1,30522), dim=-1))
                
                torch.cuda.empty_cache()
                teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.hidden_states[-1]).bool()
                student_mask = batch['attention_mask'].unsqueeze(-1).expand_as(student_out.hidden_states[-1]).bool()
                dT = teacher_out.hidden_states[-1].size(-1)
                dS = student_out.hidden_states[-1].size(-1)                

                loss[1] = CSLoss(masked_select(student_out.hidden_states[0],student_mask).view(-1,dS), \
                               masked_select(teacher_out.hidden_states[0],teacher_mask).view(-1,dT))
                               
                for i in range(4):
                    for j in range(4):
                        if(j==3):
                            loss[i*4+5] += CSLoss(masked_select(student_out.hidden_states[4*(i+1)],student_mask).view(-1,dS), \
                                           masked_select(teacher_out.hidden_states[3*(i+1)],teacher_mask).view(-1,dT))
                        else:
                            loss[i*4+j+2] += CSLoss(masked_select(student_out.hidden_states[i*4+j+1],student_mask).view(-1,dS), \
                                           masked_select(teacher_out.hidden_states[i*3+j+1],teacher_mask).view(-1,dT))
                
                loss_sum = sum(loss)
                del logit_mask, teacher_mask, student_mask, teacher_out
                torch.cuda.empty_cache()
                scaler.scale(loss_sum).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            nBatch+=1
            f.write(str(['%.4f' % l.item() for l in loss])+'\n')

            if(nBatch%num_batch_per_epoch==0):
                loss = Eval_Student(student,teacher,test_dataset, temp, batch_size)
                f1.write(str(['%.4f' % l for l in loss]) +  '\n')
                f1.flush()
                torch.save(student.state_dict(),"../../Checkpoints/BERT/Bert-16-%d-CKA-Wiki.pt" % student_dim)
    
    f.flush()    
    torch.save(student.state_dict(),"../../Checkpoints/BERT/Bert-16-%d-CKA-Wiki.pt" % student_dim)


f.close()
f1.close()
