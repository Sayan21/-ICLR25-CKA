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
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, DistilBertForMaskedLM, BertConfig, DistilBertConfig
from transformers import AutoTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
from datasets import load_dataset, load_metric, concatenate_datasets
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os

dataset_id="glue"
dataset_config="mnli"


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], max_length = 512, padding="max_length",truncation=True)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
teacher_id = "bert-base-uncased";
tokenizer = AutoTokenizer.from_pretrained(teacher_id)

dataset = load_dataset(dataset_id,dataset_config,split='validation_matched')
tokenized_valid = dataset.map(encode, remove_columns = ['premise','hypothesis','idx'], num_proc = 48)
tokenized_valid.set_format('torch')
print(tokenized_valid)
batch_size = 8
eval_dataloader = DataLoader(tokenized_valid, batch_size=batch_size, drop_last = True)





torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

teacher = AutoModelForSequenceClassification.from_pretrained(teacher_id,num_labels = 3)
teacher.load_state_dict(torch.load('../../Checkpoints/BERT/Bert-base-MNLI-32.pt'), strict = False)

teacher.to(device)
teacher.eval()
metric = load('glue',dataset_config,experiment_id = "COLA.txt")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    student_out = teacher(batch['input_ids'],batch['attention_mask'])

end.record()
torch.cuda.synchronize()

print("Inference Time for Teacher: ", start.elapsed_time(end)/(len(eval_dataloader)*batch_size))

D = [384,384,768,512,512]
L = [6,12,6,8,12]
Di =[1536,1536,3072,3072,3072]
ident = 'CKA'
for i in range(len(D)):
    student_dim = D[i]
    student_layer = L[i]
    student_int_size = Di[i]
    nH = 16 if i==4 else L[i]
    config = BertConfig.from_pretrained(teacher_id,hidden_size=student_dim,num_hidden_layers=student_layer, output_hidden_states = True, \
                                    intermediate_size = student_int_size, num_attention_heads=nH, num_labels = 3)
    student = BertForSequenceClassification(config)
    student.load_state_dict(torch.load('../../Checkpoints/BERT/BERT-%s-MNLI-%d-%d.pt' %(ident,student_layer,student_dim)), strict = False)
    student.to(device)
    student.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        student_out = student(batch['input_ids'],batch['attention_mask'])

    end.record()
    torch.cuda.synchronize()

    print("Inference Time for %d %d : %0.3f" %(student_layer,student_dim, start.elapsed_time(end)/(len(eval_dataloader)*batch_size)))
