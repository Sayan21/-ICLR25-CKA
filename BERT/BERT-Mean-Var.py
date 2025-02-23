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
import evaluate, math
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from torch import masked_select
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BertForMaskedLM, DistilBertForMaskedLM, BertConfig, DistilBertConfig
from transformers import AutoTokenizer
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.wishart import Wishart
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from datasets import load_dataset
from torch.linalg import cholesky, solve_triangular
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def tLL(x,mu,inv_T,nu):
    p = inv_T.shape[0]
    assert p == inv_T.shape[1]
    
    
    res = -0.5*(nu+p)*torch.log(1 + (1/nu)*torch.diag((x-mu)@inv_T@(x-mu).t())).mean()
    res += math.lgamma((nu+p)/2) - math.lgamma(nu/2) - 0.5*p*math.log(nu)-0.5*p*math.log(np.pi)
    
    return res


def Eval_Student(teacher,Mean,T, kappa, nu,test_dataset,batch_size):

    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    teacher.eval()
    n_layers = teacher.config.num_hidden_layers

    loss = [0]*(n_layers+1)
    T_inv = T.detach().clone()
    for i in range(n_layers+1):
        L = torch.linalg.cholesky(T[i]/(nu-T[i].shape[-1]+1)) 
        T_inv[i] = torch.cholesky_inverse(L)

    nBatch = 0
    nToken = 0    

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
        teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
        dT = teacher_out.hidden_states[0].size(-1)
        teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.hidden_states[-1]).bool()

        nBatchToken =0 
        for i in range(n_layers+1):
            TH = masked_select(teacher_out.hidden_states[i],teacher_mask).view(-1,dT)
            nBatchToken = TH.shape[0]
            loss[i] += -nBatchToken*tLL(TH,Mean[i],T_inv[i],nu-dT+1)
            
                
        nBatch+=1
        nToken+=nBatchToken


        if(nBatch%250==0): 
            break

    return [l.item()/nToken for l in loss],T_inv



teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
def encode(examples):
    return tokenizer(examples['text'], padding="max_length",truncation=True)

############################ Stream C4 Dataset ###########################
dataset = load_dataset('c4', 'en', streaming=True)
dataset = dataset.remove_columns(["timestamp", "url"])
dataset = dataset.with_format("torch")
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
batch_size = 128
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = 512)


train_dataloader = DataLoader(dataset['train'].shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched=True), batch_size = batch_size)
test_dataset = dataset['validation'].shuffle(buffer_size=100_000).map(encode, remove_columns=["text"], batched=True)


torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

      

teacher = BertForMaskedLM.from_pretrained(teacher_id,output_hidden_states = True)
print("Number of Teacher Parameters: ", sum(p.numel() for p in teacher.parameters()))
for param in teacher.parameters(): param.requires_grad = False
nu = 0
kappa = 0

f = open("../../Checkpoints/BERT-Mean-Var-C4-%d.txt" % kappa, "w+", buffering= 50)

f1 = open("../../Checkpoints/Eval-BERT-Mean-Var-C4-%d.txt" % kappa, "w+", buffering= 1)
from transformers import get_scheduler
num_epochs=50
num_batch_per_epoch = 1000
num_training_steps = num_epochs * num_batch_per_epoch

temp = 1.0
scaler = torch.cuda.amp.GradScaler()

n_layers = teacher.config.num_hidden_layers
D = teacher.config.hidden_size
Mean = torch.zeros(n_layers+1,D).to(device,torch.float64)
T = torch.zeros(n_layers+1,D,D)

    
T=T.to(device,torch.float64)


teacher.to(device)
teacher.eval()
nBatch = 0
nStart = 0
################# Estimate Mean #####################

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}

    teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) 
    dT = teacher_out.hidden_states[0].size(-1)
    teacher_mask = batch['attention_mask'].unsqueeze(-1).expand_as(teacher_out.hidden_states[-1]).bool()


    
    loss = [0]*(n_layers+1)
    for i in range(n_layers+1):
        TH = masked_select(teacher_out.hidden_states[i],teacher_mask).view(-1,dT)
        xbar = TH.mean(0)
        n = TH.size(0)
        T[i] += TH.t()@TH
        Mean[i] = (kappa*Mean[i] + n*xbar)/(kappa+n)
        nu += n
        kappa += n


        #Sigma_1_i = Wishart(df = nu,precision_matrix = T[i]).sample()
        #mean_i = MultivariateNormal(loc =Mean[i], precision_matrix = kappa*Sigma_1_i).sample()
        #loss[i] = -MultivariateNormal(loc =Mean[i], covariance_matrix = T[i]/kappa).log_prob(TH).mean()




    
    nBatch+=1
    
    

    if(nBatch%num_batch_per_epoch==0):
        loss,T_inv = Eval_Student(teacher,Mean,T, kappa, nu,test_dataset,batch_size)
        torch.save(Mean,"../../Checkpoints/BERT/Mean/Bert-Mean-C4.pt")
        torch.save(T/(nu-D+1),"../../Checkpoints/BERT/Mean/Bert-Var-C4.pt")
        f1.write(str(['%.4f' % l for l in loss]) +  '\n')
        f1.flush()

