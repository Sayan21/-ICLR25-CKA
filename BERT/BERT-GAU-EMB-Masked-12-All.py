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
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from torch import masked_select
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, DistilBertForMaskedLM, BertConfig, DistilBertConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import os

class GAULoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self,path_mean,path_var):
        super().__init__()
        self.mean = torch.load(path_mean)
        T = torch.load(path_var)
        self.T_inv = T.detach().clone()
        for i in range(T.shape[0]):
            L = torch.linalg.cholesky(T[i]) 
            self.T_inv[i] = torch.cholesky_inverse(L) 

        self.loss = nn.MSELoss()

    def forward(self, SH, TH, index): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.to(SH.device,torch.float64)
        TH = TH.to(SH.device,torch.float64)

        Sm = SH.mean(0)
        Tm = TH.mean(0)

        Cov_ST = (SH-Sm).t()@(TH-Tm)
        
        Cov_cond = (SH-Sm).t()@(SH-Sm) - Cov_ST@self.T_inv[index]@Cov_ST.t()
                        
        return torch.sum(Cov_cond*Cov_cond)

def Eval_Student(student,teacher,test_dataset, temp, batch_size):

    eval_dataloader = DataLoader(test_dataset.shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched=True), batch_size=batch_size)
    student.eval()
    teacher.eval()
    n_layers = student.config.num_hidden_layers
    loss = [0]*(n_layers+2)
    nBatch = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}

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

            loss[1] += lambdaH*CSLoss(masked_select(student_out.hidden_states[0],student_mask).view(-1,dS), \
                        masked_select(teacher_out.hidden_states[0],teacher_mask).view(-1,dT),0)

            for i in range(n_layers):
                loss[i+2] += lambdaH*CSLoss(masked_select(student_out.hidden_states[i+1],student_mask).view(-1,dS), \
                        masked_select(teacher_out.hidden_states[i+1],teacher_mask).view(-1,dT),i+1)
                

            
            nBatch+=1

            if(nBatch%2000==0): break
            
    
    return [l.item()/nBatch for l in loss]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



student_id = "distilbert-base-uncased"  #"huawei-noah/TinyBERT_General_4L_312D"
teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
def encode(examples):
    return tokenizer(examples['text'], padding="max_length",truncation=True)

############################ Stream C4 Dataset ###########################
dataset = load_dataset('c4', 'en', streaming=True)
dataset = dataset.remove_columns(["timestamp", "url"])
dataset = dataset.with_format("torch")
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
batch_size = 32
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = 512)


train_dataloader = DataLoader(dataset['train'].shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched=True), batch_size = batch_size)
test_dataset = dataset['validation']


torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

     

teacher = BertForMaskedLM.from_pretrained(teacher_id,output_hidden_states = True)
print("Number of Teacher Parameters: ", sum(p.numel() for p in teacher.parameters()))
for param in teacher.parameters(): param.requires_grad = False
student_dim = 512
n_layers = 12
student_int_size = 3072
config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=n_layers, \
                                    intermediate_size = student_int_size, num_attention_heads=16,output_hidden_states = True)
student = BertForMaskedLM(config)
print("Number of Student Parameters: ", sum(p.numel() for p in student.parameters()))

student.bert.embeddings.word_embeddings.weight.copy_(teacher.bert.embeddings.word_embeddings.weight[:,:student_dim].detach().clone())
#student.bert.embeddings.position_embeddings.weight.copy_(teacher.bert.embeddings.position_embeddings.weight[:student_dim,:student_dim].detach().clone())
student.bert.embeddings.token_type_embeddings.weight.copy_(teacher.bert.embeddings.token_type_embeddings.weight[:,:student_dim].detach().clone())


for i in range(n_layers):
    student.bert.encoder.layer[i].attention.self.query.weight.copy_(teacher.bert.encoder.layer[i].attention.self.query.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.self.query.bias.copy_(teacher.bert.encoder.layer[i].attention.self.query.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.self.key.weight.copy_(teacher.bert.encoder.layer[i].attention.self.key.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.self.key.bias.copy_(teacher.bert.encoder.layer[i].attention.self.key.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.self.value.weight.copy_(teacher.bert.encoder.layer[i].attention.self.value.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.self.value.bias.copy_(teacher.bert.encoder.layer[i].attention.self.value.bias[:student_dim].detach().clone())

    
    student.bert.encoder.layer[i].attention.output.dense.weight.copy_(teacher.bert.encoder.layer[i].attention.output.dense.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.output.dense.bias.copy_(teacher.bert.encoder.layer[i].attention.output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[i].attention.output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[i].attention.output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[i].attention.output.LayerNorm.bias[:student_dim].detach().clone())

    student.bert.encoder.layer[i].intermediate.dense.weight.copy_(teacher.bert.encoder.layer[i].intermediate.dense.weight[:student_int_size,:student_dim].detach().clone())
    student.bert.encoder.layer[i].intermediate.dense.bias.copy_(teacher.bert.encoder.layer[i].intermediate.dense.bias[:student_int_size].detach().clone())

    student.bert.encoder.layer[i].output.dense.weight.copy_(teacher.bert.encoder.layer[i].output.dense.weight[:student_dim,:student_int_size].detach().clone())
    student.bert.encoder.layer[i].output.dense.bias.copy_(teacher.bert.encoder.layer[i].output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[i].output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[i].output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[i].output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[i].output.LayerNorm.bias[:student_dim].detach().clone())
    
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

optimizer = AdamW(optimizer_grouped_parameters , lr=1e-4, betas = (0.9,0.999), eps = 1e-6)
f = open("../../Checkpoints/LOGIT-KD-GAU-EMB-12-%d-C4.txt" % student_dim, "w+", buffering= 50)
f1 = open("../../Checkpoints/LOGIT-KD-Eval-GAU-EMB-12-%d-C4.txt" % student_dim, "w+", buffering= 1)
from transformers import get_scheduler
num_epochs=25
num_batch_per_epoch = 10000
num_training_steps = num_epochs * num_batch_per_epoch
scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer,num_warmup_steps=0, num_training_steps=num_training_steps)
CELoss = torch.nn.CrossEntropyLoss()
STLoss = torch.nn.KLDivLoss(reduction = 'batchmean')
CSLoss = GAULoss(path_mean="../../Checkpoints/BERT/Mean/Bert-Mean-C4.pt", \
                    path_var="../../Checkpoints/BERT/Mean/Bert-Var-C4.pt")
temp = 1.0
lambdaH = 1e-12
scaler = torch.cuda.amp.GradScaler()


teacher.to(device)
student.to(device)
teacher.eval()
nBatch = 0
for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
    student.train()
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

            loss[1] = lambdaH*CSLoss(masked_select(student_out.hidden_states[0],student_mask).view(-1,dS), \
                            masked_select(teacher_out.hidden_states[0],teacher_mask).view(-1,dT),0)
                               
            for i in range(n_layers):
                loss[i+2] = lambdaH*CSLoss(masked_select(student_out.hidden_states[i+1],student_mask).view(-1,dS), \
                                    masked_select(teacher_out.hidden_states[i+1],teacher_mask).view(-1,dT),i+1)
                
            loss_sum = sum(loss)
            torch.cuda.empty_cache()
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        nBatch+=1
        f.write(str(['%.4f' % l.item() for l in loss])+'\n')

        if(nBatch%num_batch_per_epoch==0):
            loss = Eval_Student(student,teacher, test_dataset, temp, batch_size)
            f1.write(str(['%.4f' % l for l in loss]) +  '\n')
            f1.flush()
            torch.save(student.state_dict(),"../../Checkpoints/Bert-GAU-12-%d-C4.pt" % student_dim)
    
f.flush()    
torch.save(student.state_dict(),"../../Checkpoints/Bert-GAU-12-%d-C4.pt" % student_dim)


f.close()
f1.close()
