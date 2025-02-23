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
from transformers import AutoModelForSequenceClassification, BartForConditionalGeneration, BartConfig
from transformers import AutoTokenizer, BartTokenizer
from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os
    

rouge = load('rouge', experiment_id = "Bart-INF.txt", use_stemmer = True)


teacher_id = "facebook/bart-large-xsum"
tokenizer = BartTokenizer.from_pretrained(teacher_id)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenized_datasets = load_from_disk('XSUM')

def encode(examples):
    result = {}
    temp = tokenizer(examples['document'], truncation=True, padding='max_length', max_length = 1024)
    result['input_ids'] = temp['input_ids']
    result['attention_mask'] = temp['attention_mask']
    temp = tokenizer(examples['summary'], truncation=True, padding='max_length', max_length = 256)
    result['label_ids'] = temp['input_ids']
    result['label_attention_mask'] = temp['attention_mask']
    return result

# tokenized_valid = dataset.map(encode, remove_columns = ["document", "id", "summary"], num_proc = 48)
# tokenized_valid.set_format('torch')
# print(tokenized_valid)
batch_size = 16
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size, drop_last = True)



teacher = BartForConditionalGeneration.from_pretrained(teacher_id)
print("Teacher: ", teacher.config)

teacher.eval()
teacher.to(device)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()


for batch in eval_dataloader:
    with torch.cuda.amp.autocast():
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
        predictions = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], max_length = 127)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(batch['label_ids'].cpu().numpy() != -100, batch['label_ids'].cpu().numpy(), tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge.add_batch(predictions=decoded_preds, references=decoded_labels)

end.record()

torch.cuda.synchronize()

print("Inference Time for Teacher: ", start.elapsed_time(end)/(len(eval_dataloader)*batch_size))
print(rouge.compute())



################################################################################################

D = [640,768,768,768]
L = [3,3,6,12]
FFD =[4096,4096,3072,3072]

for i in range(len(D)):
    student_dim = D[i]
    student_layer = L[i]
    ffn_dim = FFD[i]
    config = BartConfig(vocab_size = 50264, d_model = student_dim, \
                        encoder_layers = student_layer, decoder_layers = student_layer, \
                        encoder_ffn_dim = ffn_dim, decoder_ffn_dim = ffn_dim)
    student = BartForConditionalGeneration(config)
    student.load_state_dict(torch.load("/data/projects/punim0478/sayantand/Checkpoints/Bart-%d-%d-XSUM-CKA.pt" % (student_layer,student_dim)))
    
    print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))

    student.eval()
    student.to(device)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for batch in eval_dataloader:
        with torch.cuda.amp.autocast():
            torch.cuda.empty_cache()
            batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
            predictions = student.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], max_length = 127)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(batch['label_ids'].cpu().numpy() != -100, batch['label_ids'].cpu().numpy(), tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge.add_batch(predictions=decoded_preds, references=decoded_labels)

    end.record()
        

    torch.cuda.synchronize()

    print("Inference Time for %d %d : %0.3f" %(student_layer,student_dim, start.elapsed_time(end)/(len(eval_dataloader)*batch_size)))
    print(rouge.compute())
