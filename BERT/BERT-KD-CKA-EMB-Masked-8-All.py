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

def Eval_Student(student,teacher,test_dataset, temp = 1.0, batch_size = 32):

    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size)
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
            j=2
            for i in range(4):
                loss[j] += CSLoss(masked_select(student_out.hidden_states[2*i+1],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[3*i+2],teacher_mask).view(-1,dT))
                j+=1
                
            for i in range(4):
                loss[j] += CSLoss(masked_select(student_out.hidden_states[2*i+2],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[3*i+3],teacher_mask).view(-1,dT))
                j+=1
                

            
            nBatch+=1
            if(nBatch%2000==0): break
            
    
    return [l.item()/nBatch for l in loss]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



student_id = "distilbert-base-uncased"  #"huawei-noah/TinyBERT_General_4L_312D"
teacher_id = "bert-base-uncased"
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length",truncation=True)

# batch_size = 32
# dataset = load_dataset('wikipedia', '20220301.en',split = 'train')
# print(dataset)
# tokenized_datasets = dataset.map(tokenize_function, remove_columns = ["text", "url", "id", "title"], num_proc = 24)
# tokenized_datasets.set_format("torch")
# train_dataset, test_dataset = torch.utils.data.random_split(tokenized_datasets, [6400000, len(tokenized_datasets) - 6400000])
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

teacher = BertForMaskedLM.from_pretrained(teacher_id,output_hidden_states = True)
print("Number of Teacher Parameters: ", sum(p.numel() for p in teacher.parameters()))
for param in teacher.parameters(): param.requires_grad = False

student_dim = 512
n_layers = 8
student_int_size = 1536
config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=n_layers, num_attention_heads=n_layers,output_hidden_states = True)
student = BertForMaskedLM(config)
print("Number of Student Parameters: ", sum(p.numel() for p in student.parameters()))

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
teacher.to(device)
student.to(device)

student.bert.embeddings.word_embeddings.weight.copy_(teacher.bert.embeddings.word_embeddings.weight[:,:student_dim].detach().clone())
student.bert.embeddings.position_embeddings.weight.copy_(teacher.bert.embeddings.position_embeddings.weight[:student_dim,:student_dim].detach().clone())
student.bert.embeddings.token_type_embeddings.weight.copy_(teacher.bert.embeddings.token_type_embeddings.weight[:,:student_dim].detach().clone())


for i in range(3):
    student.bert.encoder.layer[2*i].attention.self.query.weight.copy_(teacher.bert.encoder.layer[3*i].attention.self.query.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.self.query.bias.copy_(teacher.bert.encoder.layer[3*i].attention.self.query.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.self.key.weight.copy_(teacher.bert.encoder.layer[3*i].attention.self.key.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.self.key.bias.copy_(teacher.bert.encoder.layer[3*i].attention.self.key.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.self.value.weight.copy_(teacher.bert.encoder.layer[3*i].attention.self.value.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.self.value.bias.copy_(teacher.bert.encoder.layer[3*i].attention.self.value.bias[:student_dim].detach().clone())

    
    student.bert.encoder.layer[2*i].attention.output.dense.weight.copy_(teacher.bert.encoder.layer[3*i].attention.output.dense.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.output.dense.bias.copy_(teacher.bert.encoder.layer[3*i].attention.output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[3*i].attention.output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].attention.output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[3*i].attention.output.LayerNorm.bias[:student_dim].detach().clone())

    student.bert.encoder.layer[2*i].intermediate.dense.weight.copy_(teacher.bert.encoder.layer[3*i].intermediate.dense.weight[:student_int_size,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].intermediate.dense.bias.copy_(teacher.bert.encoder.layer[3*i].intermediate.dense.bias[:student_int_size].detach().clone())

    student.bert.encoder.layer[2*i].output.dense.weight.copy_(teacher.bert.encoder.layer[3*i].output.dense.weight[:student_dim,:student_int_size].detach().clone())
    student.bert.encoder.layer[2*i].output.dense.bias.copy_(teacher.bert.encoder.layer[3*i].output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[3*i].output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i].output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[3*i].output.LayerNorm.bias[:student_dim].detach().clone())

for i in range(3):
    student.bert.encoder.layer[2*i+1].attention.self.query.weight.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.query.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.self.query.bias.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.query.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.self.key.weight.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.key.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.self.key.bias.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.key.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.self.value.weight.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.value.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.self.value.bias.copy_(teacher.bert.encoder.layer[3*i+1].attention.self.value.bias[:student_dim].detach().clone())

    
    student.bert.encoder.layer[2*i+1].attention.output.dense.weight.copy_(teacher.bert.encoder.layer[3*i+1].attention.output.dense.weight[:student_dim,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.output.dense.bias.copy_(teacher.bert.encoder.layer[3*i+1].attention.output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[3*i+1].attention.output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].attention.output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[3*i+1].attention.output.LayerNorm.bias[:student_dim].detach().clone())

    student.bert.encoder.layer[2*i+1].intermediate.dense.weight.copy_(teacher.bert.encoder.layer[3*i+1].intermediate.dense.weight[:student_int_size,:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].intermediate.dense.bias.copy_(teacher.bert.encoder.layer[3*i+1].intermediate.dense.bias[:student_int_size].detach().clone())

    student.bert.encoder.layer[2*i+1].output.dense.weight.copy_(teacher.bert.encoder.layer[3*i+1].output.dense.weight[:student_dim,:student_int_size].detach().clone())
    student.bert.encoder.layer[2*i+1].output.dense.bias.copy_(teacher.bert.encoder.layer[3*i+1].output.dense.bias[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].output.LayerNorm.weight.copy_(teacher.bert.encoder.layer[3*i+1].output.LayerNorm.weight[:student_dim].detach().clone())
    student.bert.encoder.layer[2*i+1].output.LayerNorm.bias.copy_(teacher.bert.encoder.layer[3*i+1].output.LayerNorm.bias[:student_dim].detach().clone())
    
student.cls.predictions.transform.dense.weight.copy_(teacher.cls.predictions.transform.dense.weight[:student_dim,:student_dim].detach().clone())
student.cls.predictions.transform.dense.bias.copy_(teacher.cls.predictions.transform.dense.bias[:student_dim].detach().clone())

student.cls.predictions.decoder.weight.copy_(teacher.cls.predictions.decoder.weight[:,:student_dim].detach().clone())
student.cls.predictions.decoder.bias.copy_(teacher.cls.predictions.decoder.bias.detach().clone())

############################ Stream C4 Dataset ###########################
dataset = load_dataset('c4', 'en', streaming=True)
dataset = dataset.remove_columns(["timestamp", "url"])
dataset = dataset.with_format("torch")
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
batch_size = 32
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = 512)


train_dataloader = DataLoader(dataset['train'].shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched=True), batch_size = batch_size)
test_dataset = dataset['validation'].map(encode, remove_columns=["text"], batched=True)





no_decay = ["bias", "LayerNorm.weight"]
high_decay = ["Embedding"]
all_mentioned_decay = no_decay + high_decay
optimizer_grouped_parameters = [
    {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in all_mentioned_decay ) and p.requires_grad],
     "weight_decay": 5e-4,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in high_decay) and p.requires_grad],
     "weight_decay": 5e-4,
    }
]

optimizer = AdamW(optimizer_grouped_parameters , lr=2e-4, betas = (0.9,0.999), eps = 1e-7)
f = open("../../Checkpoints/BERT/LOGIT-KD-CKA-EMB-8-%d-C4.txt" % student_dim, "w+", buffering= 50)
f1 = open("../../Checkpoints/BERT/LOGIT-KD-Eval-CKA-EMB-8-%d-C4.txt" % student_dim, "w+", buffering= 1)
from transformers import get_scheduler
vaild_sample_interval = 10000
num_epochs=30
num_training_steps = num_epochs * vaild_sample_interval
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
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

                loss[1] = CSLoss(masked_select(student_out.hidden_states[0],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[0],teacher_mask).view(-1,dT))
                j=2
                for i in range(4):
                    loss[j] = CSLoss(masked_select(student_out.hidden_states[2*i+1],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[3*i+2],teacher_mask).view(-1,dT))
                    j+=1
                
                for i in range(4):
                    loss[j] = CSLoss(masked_select(student_out.hidden_states[2*i+2],student_mask).view(-1,dS), masked_select(teacher_out.hidden_states[3*i+3],teacher_mask).view(-1,dT))
                    j+=1
                
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

            if(nBatch%vaild_sample_interval==0):
                loss = Eval_Student(student,teacher, test_dataset, temp, batch_size)
                f1.write(str(['%.4f' % l for l in loss]) +  '\n')
                f1.flush()
                torch.save(student.state_dict(),"../../Checkpoints/BERT/Bert-8-%d-CKA-C4.pt" % student_dim)
    
    torch.save(student.state_dict(),"../../Checkpoints/BERT/Bert-8-%d-CKA-C4.pt" % student_dim)


f.close()
f1.close()
