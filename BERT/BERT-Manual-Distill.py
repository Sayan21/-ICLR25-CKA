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
from torch.optim import AdamW, Adam, SGD
import torch.nn.functional as F
import evaluate
from torch import masked_select
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, DistilBertForMaskedLM, BertConfig, DistilBertConfig
from transformers import AutoTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from evaluate import load
import os
from transformers import get_scheduler
from datasets import load_metric
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)

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



tokenized_datasets = dataset.map(tokenize_function, remove_columns =['premise','hypothesis','idx'],  batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format('torch')
batch_size = 32
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle = True)
eval_dataloader = DataLoader(tokenized_datasets['validation_matched'], batch_size=batch_size, drop_last = True)
CELoss = nn.CrossEntropyLoss()
STLoss = torch.nn.KLDivLoss(reduction = 'batchmean')

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

teacher = AutoModelForSequenceClassification.from_pretrained(teacher_id,num_labels = 3)
student_dim = 512
student_layer = 12
student_int_size = 3072
config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=student_layer, output_hidden_states = True, \
                                    intermediate_size = student_int_size, num_attention_heads=16, num_labels = 3)
student = BertForSequenceClassification(config)
#student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels = 3)
print(student.config)
student.load_state_dict(torch.load('../../Checkpoints/BERT/Bert-%d-%d-CKA-C4.pt' %(student_layer,student_dim)), strict = False)

# student = BertForSequenceClassification.from_pretrained("/data/projects/punim0478/sayantand/Checkpoints/BERT/minilm/12_384",num_labels = 3)

student.to(device)
student.eval()
teacher.to(device)
teacher.eval()
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



no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay ) and p.requires_grad],
     "weight_decay": 5e-4,
    },
    {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    }
]
torch.manual_seed(42)
num_epochs=10
optimizer = AdamW(optimizer_grouped_parameters , lr=3e-5, betas = (0.9,0.999), eps = 1e-8)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_dataloader))

scaler = torch.cuda.amp.GradScaler()

f1 = open("../../Checkpoints/LOGIT-KD-Eval-CKA-BERT-MNLI-%d-%d.txt" % (student_layer,student_dim), "w+", buffering= 1)


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
    f1.write(str('%.3f ' % loss.item()) + str(result) + '\n')
    torch.save(student.state_dict(),"../../Checkpoints/BERT-CKA-MNLI-%d-%d.pt" % (student_layer,student_dim))

torch.save(student.state_dict(),"../../Checkpoints/BERT-CKA-MNLI-%d-%d.pt" % (student_layer,student_dim))



