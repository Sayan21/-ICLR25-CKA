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
from datasets import load_dataset, load_metric, concatenate_datasets, interleave_datasets
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os
    

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def Eval_Student(teacher,Encoder_LM,test_dataset):
    evalloader = DataLoader(test_dataset, batch_size = 16)
    nBatch=0
    loss = 0
    for batch in evalloader:
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}

        with torch.cuda.amp.autocast():
            torch.cuda.empty_cache()

            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
            logits = Encoder_LM(teacher_out.encoder_hidden_states[-1])
            loss += CELoss(rearrange(logits,'a b c -> (a b) c'), rearrange(batch['input_ids'], 'a b -> (a b)'))
            nBatch+=1

        if(nBatch%2000==0):
            break
            
    
    return loss.item()/nBatch


teacher_id = "facebook/mbart-large-50-many-to-many-mmt"
teacher = MBartForConditionalGeneration.from_pretrained(teacher_id, output_hidden_states = True, use_cache = False)
for param in teacher.parameters(): param.requires_grad = False
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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



teacher.to(device)
Encoder_LM = nn.Linear(in_features=1024, out_features=250054, bias=False)
Encoder_LM.to(device)

############################ Training Starts ############################
scaler = torch.cuda.amp.GradScaler()
optimizer = AdamW(Encoder_LM.parameters() , lr=1e-4, betas = (0.9,0.999), eps = 1e-7, weight_decay = 1e-4)

f = open("../../Checkpoints/LOGIT-KD-MBart-LM.txt", "w+", buffering= 50)
f1 = open("../../Checkpoints/LOGIT-KD-Eval-MBart-LM.txt", "w+", buffering= 1)
from transformers import get_scheduler
num_epochs=30
batch_per_eval = 10000
num_training_steps = num_epochs * batch_per_eval
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

trainloader = DataLoader(tokenized_dataset_train, batch_size = 32)
CELoss = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)


nBatch = 0
for batch in trainloader:
    batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}

    with torch.cuda.amp.autocast():
        teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
    
        with torch.enable_grad():
            torch.cuda.empty_cache()

            logits = Encoder_LM(teacher_out.encoder_hidden_states[-1])
            # dV = logits.size(-1)
            # logit_mask = batch['attention_mask'].unsqueeze(-1).expand_as(logits).bool()
            # SL = torch.masked_select(logits,logit_mask).view(-1,dV)
            loss = CELoss(rearrange(logits,'a b c -> (a b) c'), rearrange(batch['input_ids'], 'a b -> (a b)'))
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            f.write(str('%.3f' % loss.item()) + '\n')

            nBatch+=1
        
    if(nBatch%batch_per_eval==0):
        torch.save(Encoder_LM.state_dict(),"../../Checkpoints/MBART_LM.pt" )
        torch.cuda.empty_cache()

        loss = Eval_Student(teacher,Encoder_LM,tokenized_dataset_valid)
        f1.write(str('%.3f' % loss) + '\n')
        f1.flush()

