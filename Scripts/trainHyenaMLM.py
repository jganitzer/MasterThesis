import wandb
import torch
import pandas as pd

from utils.genomicDataset import GenomicDataset, train_test_dataset_split
from utils.tokenizer import CharacterTokenizer

from pathlib import Path

from datetime import datetime

from transformers import MegaConfig
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import math

from models.hyenaEncoderMLM import HyenaForMaskedLM

import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"

wandb_run_name = "_"

entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

checkpointRoot = '/data/checkpoints/'

file_path = dataRoot + heritability + '/train/snp_X_train_maf_token.txt'

#Config
#HyenaOperator
order = 2
activation = 'relu'

max_pos = 26284

lr = 0.001

batch_size = 16
num_epochs = 5#
weight_decay=0.01 

val_size = 0.01
train_ratio = 1.0 - val_size
input_dim = 8
num_layers = 6
num_heads = 1

patience = 15
lambda_ = 0.001

mlm_probability = 0.15

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
checkpoint_path_pretrained = checkpoint_path + 'pretrained/'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(checkpoint_path_pretrained).mkdir(parents=True, exist_ok=True)


run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity)

config = MegaConfig(
    vocab_size=10, 
    hidden_size=input_dim,
    num_attention_heads=num_heads,
    intermediate_size=4*input_dim, 
    num_hidden_layers=num_layers,
    output_attentions=False,
    return_dict=True,
    use_chunking=True,
    relative_positional_bias = "rotary",
    add_token_type_embeddings = False           
)

model = HyenaForMaskedLM(config, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, order = order, activation = activation)

tokenizer = CharacterTokenizer(
    characters=['0', '1', '2'],  # add SNP characters, 
    model_max_length= max_pos,
    add_special_tokens=True, 
    padding_side='right',
    return_special_tokens_mask=False    
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)

dataset = GenomicDataset(file_path, max_length = max_pos, tokenizer = tokenizer, use_padding = False)
train_dataset, eval_dataset = train_test_dataset_split(dataset, train_ratio=train_ratio, random_seed=42)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

training_args = TrainingArguments(
    output_dir=checkpoint_path,
    evaluation_strategy="epoch",
    learning_rate=lr,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    push_to_hub=False,
    do_train = True,
    do_eval = True,
    save_strategy = 'epoch',
    run_name = wandb_run_name,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset = eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics if training_args.do_eval else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None
)

# Training
if training_args.do_train:

    train_result = trainer.train()
    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(checkpoint_path)
    model.save_pretrained(checkpoint_path_pretrained)
    torch.save(model.state_dict(), checkpoint_path + "best_model_state_dict.pth")
    torch.save(model, checkpoint_path + "best_model.pth")
    
# Evaluation
if training_args.do_eval:

    metrics = trainer.evaluate()

    max_eval_samples = len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

run.finish()
