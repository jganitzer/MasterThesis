import wandb
import torch
import pandas as pd
from utils.genomicDataset import GenomicDataset3, train_test_split
from utils.tokenizer import CharacterTokenizer
from pathlib import Path
from datetime import datetime
from transformers import  DataCollatorWithPadding, EvalPrediction
from transformers import Trainer, TrainingArguments
import evaluate
from models.megaencoder import MegaChunkRegressor_MLP_catTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"
wandb_run_name = "_"
entity="_"
heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

checkpointRoot = '/data/checkpoints/'

y_path = "/train/snp_y_train_maf.parquet"
file_path = dataRoot + heritability + '/train/snp_X_train_maf_token.txt'
label_path = dataRoot + heritability + y_path

#Config

paddedLength = 26624
max_pos = 27000
chunk_size = 512

lr = 0.001 
batch_size = 8
num_epochs = 20
weight_decay=0.01 

val_size = 0.01 
train_ratio = 1.0 - val_size

input_dim = 8
num_layers = 2
num_heads = 1

hidden1 = 1000
hidden2 = 500 

lambda_ = 0.001 #

mlm_probability = 0.15

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
checkpoint_path_pretrained = checkpoint_path + 'pretrained/'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(checkpoint_path_pretrained).mkdir(parents=True, exist_ok=True)

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity)

model = MegaChunkRegressor_MLP_catTrainer(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, num_features = paddedLength, chunk_size=chunk_size).to(device)

tokenizer = CharacterTokenizer(
    characters=['0', '1', '2'],  # add SNP characters, 
    model_max_length= paddedLength,
    add_special_tokens=True, 
    padding_side='right',
    return_special_tokens_mask=True    
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, pad_to_multiple_of = chunk_size
)

dataset = GenomicDataset3(file_path = file_path, label_path = label_path, max_length = paddedLength, tokenizer = tokenizer, use_padding = True)
train_dataset, eval_dataset = train_test_split(dataset, train_ratio=train_ratio, random_seed=42)

metric_mse = evaluate.load("mse")
metric_mae = evaluate.load("mae")
metric_r2 = evaluate.load("r_squared")
metric_r = evaluate.load("pearsonr")

#https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result_mse = metric_mse.compute(predictions=preds, references=p.label_ids)["mse"]
    result_mae = metric_mae.compute(predictions=preds, references=p.label_ids)["mae"]
    result_r2 = metric_r2.compute(predictions=preds, references=p.label_ids)
    result_r = metric_r.compute(predictions=preds, references=p.label_ids, return_pvalue=True)

    pearsonr = result_r["pearsonr"]
    p_value = result_r["p-value"]
        
    return {"mse": result_mse, "mae": result_mae, "r_squared": result_r2, "pearsonr": pearsonr, "p_value": p_value}


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
    compute_metrics=compute_metrics
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

# Evaluation
if training_args.do_eval:

    metrics = trainer.evaluate()

    max_eval_samples = len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

run.finish()
