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
from datasets import load_dataset, load_metric, concatenate_datasets
from evaluate import load
from transformers.models.bart.modeling_bart import shift_tokens_right
from torcheval.metrics import BLEUScore
from einops import rearrange
import os
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

src_lang = 'fr'
def encode(examples):
    result = {}
    temp = tokenizer(examples['translation']['en'], truncation=True, padding='max_length', max_length = 128)
    result['input_ids'] = temp['input_ids']
    result['attention_mask'] = temp['attention_mask']
    temp = tokenizer(examples['translation'][src_lang], truncation=True, padding='max_length', max_length = 128)
    result['label_ids'] = temp['input_ids']
    result['label_attention_mask'] = temp['attention_mask']
    return result




torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang='en_XX', tgt_lang="fr_XX")

teacher_id = "context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr"
teacher = MBartForConditionalGeneration.from_pretrained(teacher_id)



############################ Concatenate XSUM, CNN, CC NEWS & NEWSROOM ##########################
dataset = load_dataset('wmt14', 'fr-en', split = 'validation',download_mode = "reuse_dataset_if_exists")
#dataset = load_dataset('iwslt2017', 'iwslt2017-fr-en', split = 'test')
tokenized_valid = dataset.map(encode, remove_columns = ['translation'], num_proc = 32)
tokenized_valid.set_format('torch')
teacher.eval()
teacher.to(device)
print(tokenized_valid)
batch_size = 8

eval_dataloader = DataLoader(tokenized_valid, batch_size=batch_size, drop_last = True)


print("Teacher: ", teacher.config)
bleu = load("sacrebleu",experiment_id = "MBart-wslt-fr-%d-%d.txt" % (teacher.config.encoder_layers,teacher.config.d_model), trust_remote_code=True)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()


for batch in eval_dataloader:
    with torch.cuda.amp.autocast():
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
        predictions = teacher.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], num_beams=5, early_stopping = True, max_length = 127)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = batch['label_ids'].cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        bleu.add_batch(predictions = decoded_preds, references = decoded_labels)
                

end.record()

torch.cuda.synchronize()

print(len(eval_dataloader))
print("Inference Time for Teacher: ", start.elapsed_time(end)/(len(eval_dataloader)*batch_size))
print(bleu.compute())




D = [384,512,512,640]
L = [6,6,12,12]

for i in range(4):
    student_dim = D[i]
    student_layer = L[i]
    config = MBartConfig(vocab_size = teacher.config.vocab_size, d_model = student_dim, \
                        encoder_layers = student_layer, decoder_layers = student_layer, \
                        encoder_ffn_dim = 4*student_dim, decoder_ffn_dim = 4*student_dim,
                        output_hidden_states = True, output_past = False, use_cache = True)
    student = MBartForConditionalGeneration(config)
    student.load_state_dict(torch.load("../../Checkpoints/BART/August_2024/MBart-fr-%d-%d-CKA.pt" % (student_layer,student_dim)))
    
    print("Student Parameters", sum(p.numel() for p in student.parameters() if p.requires_grad))

    student.eval()
    student.to(device)

    bleu = load("sacrebleu",experiment_id = "MBart-%d-%d.txt" % (student.config.encoder_layers,student.config.d_model), trust_remote_code=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for batch in eval_dataloader:
        with torch.cuda.amp.autocast():
            torch.cuda.empty_cache()
            batch = {k: v.to(device) for k, v in batch.items() if not isinstance(v,list)}
            predictions = student.generate(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'], num_beams=5, early_stopping = True, max_length = 127)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = batch['label_ids'].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            bleu.add_batch(predictions = decoded_preds, references = decoded_labels)
                

    end.record()

                
        

    torch.cuda.synchronize()

    print(len(eval_dataloader))
    print("Inference Time for %d %d : %0.3f" %(student_layer,student_dim, start.elapsed_time(end)/(len(eval_dataloader)*batch_size)))

    print(bleu.compute())