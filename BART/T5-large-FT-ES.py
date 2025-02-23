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
from transformers import AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer, T5Tokenizer, T5TokenizerFast
from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels






def Eval(teacher,test_dataset, batch_size = 32):
    bleu = load("sacrebleu",experiment_id = "T5-large-%s.txt" % tgt_lang, trust_remote_code=True)

    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    teacher.eval()

    loss = 0
    nBatch = 0
    for batch in eval_dataloader:
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
        batch = {k: v.to(device).squeeze(1) for k, v in batch.items() if not isinstance(v,list)}
        teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids']) 
        logit_mask = batch['label_attention_mask'].unsqueeze(-1).expand_as(teacher_out.logits).bool()
        TL = masked_select(teacher_out.logits,logit_mask).view(-1,dV)
        loss += CELoss(TL,masked_select(batch['label_ids'],batch['label_attention_mask'].bool()))
        
            # predictions = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],num_beams=5, early_stopping = True, max_length = 127)
            # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # labels = batch['label_ids'].cpu().numpy()
            # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            # bleu.add_batch(predictions = decoded_preds, references = decoded_labels)
            
        nBatch+=1
        if(nBatch%5000==0): break    
   
    
    return loss.item()/nBatch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

tgt_lang = 'cot'
teacher_id = "google/t5-v1_1-large"
tokenizer = T5TokenizerFast.from_pretrained(teacher_id)



# print(tokenized_datasets)
# batch_size = 32
# train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
# test_dataset = tokenized_datasets['test']
# 
# print(vaild_sample_interval)



torch.manual_seed(42)
batch_size = 2
tokenized_datasets = load_from_disk('/scratch/punim0478/sayantand/Dump/CoT')
print(tokenized_datasets)
train_dataset, test_dataset = torch.utils.data.random_split(tokenized_datasets, [1800000, len(tokenized_datasets) - 1800000])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
vaild_sample_interval = len(train_dataloader)//20


torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
teacher = T5ForConditionalGeneration.from_pretrained(teacher_id)








############################ Training Starts ############################
no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {"params": [p for n, p in teacher.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 5e-3,
    },
    {"params": [p for n, p in teacher.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
     "weight_decay": 0.0,
    }
]
torch.manual_seed(99)
LR = 2
optimizer = AdamW(optimizer_grouped_parameters , lr=LR*1e-5, betas = (0.9,0.999), eps = 1e-6)
f = open("../../Checkpoints/LOGIT-KD-CKA-T5-%s-large-%d.txt" % (tgt_lang,LR), "w+", buffering= 20)
f1 = open("../../Checkpoints/LOGIT-KD-Eval-CKA-T5-%s-large-%d.txt" % (tgt_lang,LR), "w+", buffering= 1)
from transformers import get_scheduler
num_epochs=2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
STLoss = nn.KLDivLoss(reduction = 'batchmean')
pad_token_id = tokenizer.pad_token_id
CELoss = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()
teacher.to(device)
teacher.eval()


for epoch in range(num_epochs):
    teacher.train()
    nBatch = 0
    for batch in train_dataloader:
#        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
        torch.cuda.empty_cache()
        batch = {k: v.to(device).squeeze(1) for k, v in batch.items() if not isinstance(v,list)}
            #pseudo = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],max_new_tokens = 127)

        with torch.enable_grad():
            teacher_out = teacher(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], labels = batch['label_ids']) 
            dV = teacher_out.logits.size(-1)

            logit_mask = batch['label_attention_mask'].unsqueeze(-1).expand_as(teacher_out.logits).bool()
            TL = masked_select(teacher_out.logits,logit_mask).view(-1,dV)
            loss = CELoss(TL,masked_select(batch['label_ids'],batch['label_attention_mask'].bool()))
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()

            nBatch+=1
            if(nBatch%16==0):
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                f.write('%.5f\n' % loss.item())
        
        if(nBatch%vaild_sample_interval==0):
            torch.save(teacher.state_dict(),"../../Checkpoints/T5-large-%s-%d.pt" % (tgt_lang,LR))
            loss = Eval(teacher,test_dataset, 8)
            f1.write(str(' %.5f ' % loss) + '\n')
            f1.flush()
            
    


torch.save(teacher.state_dict(),"../../Checkpoints/T5-large-%s-%d.pt" % (tgt_lang,LR))

f.close()
f1.close()
