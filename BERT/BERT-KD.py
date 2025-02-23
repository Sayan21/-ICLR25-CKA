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
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, BertForSequenceClassification, BertConfig
from huggingface_hub import HfFolder
from transformers import AutoConfig
from transformers import get_scheduler
from datasets import load_metric
import numpy as np
from torch.optim import AdamW

student_id = "distilbert-base-uncased" #"huawei-noah/TinyBERT_General_4L_312D"
teacher_id = "bert-base-uncased" #"roberta-base""


from transformers import AutoTokenizer
from evaluate import load
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
student_tokenizer = AutoTokenizer.from_pretrained(teacher_id)

dataset_id="glue"
dataset_config="mnli"
from datasets import load_dataset

dataset = load_dataset(dataset_id,dataset_config)
repo_name = "bert-" +dataset_config+ "-distilled-cka"
metrics_map = {'cola':'matthews_correlation', 'qqp':'f1', 'mrpc':'f1', 'stsb':'spearmanr', 'mnli':'accuracy', 'sst2': 'accuracy', 'qnli': 'accuracy', 'rte':'accuracy'}

key = ["premise", "hypothesis"]
dataset = load_dataset(dataset_id,dataset_config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    if len(key)>1: return tokenizer(examples[key[0]],examples[key[1]], max_length = 512, padding="max_length",truncation=True)
    else: return tokenizer(examples[key[0]], padding="max_length",truncation=True)



print(dataset['train'])
tokenized_datasets = dataset.map(tokenize_function, remove_columns =['premise','hypothesis','idx'],  batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format('torch')

batch_size = 64
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle = True)
eval_dataloader = DataLoader(tokenized_datasets['validation_matched'], batch_size=batch_size, drop_last = True)

CELoss = nn.CrossEntropyLoss()

# define training args

# define data_collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define model
teacher = AutoModelForSequenceClassification.from_pretrained(
    teacher_id,
    num_labels = 3
)

student_dim = 512
n_layers = 8
student_int_size = 2048
config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=n_layers, \
                                    intermediate_size = student_int_size, num_attention_heads=n_layers, num_labels = 3)
student = BertForSequenceClassification(config)
# print(student_model.config)
student.load_state_dict(torch.load('../Checkpoints/Bert-8-512-CKA-All-Wiki.pt'),strict=False)
for param in teacher.parameters(): param.requires_grad= False

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

teacher.to(device)
student.to(device)
student.eval()
metric = load('glue',dataset_config)
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.cuda.amp.autocast():
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        student_out = student(batch['input_ids'],batch['attention_mask'])
        loss = CELoss(student_out.logits, batch["labels"].to(device))
        predictions = torch.argmax(student_out.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])


result = metric.compute()
print(loss.item(), result)

STLoss = torch.nn.KLDivLoss(reduction = 'batchmean')
CELoss = nn.CrossEntropyLoss()
glue_metric = load(dataset_id,dataset_config)
teacher.eval()
scaler = torch.cuda.amp.GradScaler()
num_epochs=5
num_training_steps = num_epochs * len(train_dataloader)

no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay ) and p.requires_grad],
     "weight_decay": 5e-4,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    }
]
optimizer = AdamW(optimizer_grouped_parameters , lr=5e-5, betas = (0.9,0.999), eps = 1e-8)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


for epoch in range(num_epochs):
    torch.cuda.empty_cache()

    student.train()
    for batch in train_dataloader:
        with torch.enable_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                torch.cuda.empty_cache()
                student_out = student(batch['input_ids'],batch['attention_mask'])
                teacher_out = teacher(batch['input_ids'],batch['attention_mask'])
                loss1 = CELoss(student_out.logits, batch["labels"])
                loss2 = STLoss(F.log_softmax(student_out.logits,dim=-1),F.softmax(teacher_out.logits, dim=-1))
                loss = loss1 + loss2
                torch.cuda.empty_cache()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

    print(loss.item())
    student.eval()
    metric = load('glue',dataset_config)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            student_out = student(batch['input_ids'],batch['attention_mask'])
            loss = CELoss(student_out.logits, batch["labels"])
            predictions = torch.argmax(student_out.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])


    result = metric.compute()
    print(loss.item(), result)
    torch.save(student.state_dict(),'../Checkpoints/Bert-8-512-CKA-%s.pt' % dataset_config)

torch.save(student.state_dict(),'../Checkpoints/Bert-8-512-CKA-%s.pt' % dataset_config)