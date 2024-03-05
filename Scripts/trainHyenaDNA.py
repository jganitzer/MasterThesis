import wandb
import torch
import torch.nn as nn
import time
import pandas as pd
import torch.optim as optim
from models.hyenaDNA import HyenaDNAModel
from utils.torchDataloader import CatLoaderCLSTokenizer
from utils.helper import checkpoint
from pathlib import Path
from transformers import get_scheduler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime

import os

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"
wandb_run_name = "_"

resume_training = False

entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

X_path = "/train/snp_X_train_maf.parquet"
y_path = "/train/snp_y_train_maf.parquet"

file_path = dataRoot + heritability + '/train/snp_X_train_maf_token.txt'
label_path = dataRoot + heritability + y_path

val_size = 0.01
patience = 10

mode = 'sum' # "Mode must be ['last' | 'first' | 'pool' | 'sum']"
l_output = 0

# experiment settings:
num_epochs = 40 

use_padding = True
batch_size = 64
learning_rate = 0.001
rc_aug = True  
add_eos = False  
weight_decay = 0.1

use_head = True
n_classes = 1

hidden1 = 1000
hidden2 = 500
lambda_ = 0.001

num_features = 26284
catLoader = CatLoaderCLSTokenizer(file_path=file_path, label_path=label_path, batch_size = batch_size, val_size = val_size, model_max_length = num_features)

train_loader, test_loader = catLoader.getLoaders()

backbone_cfg = {
    "d_model": 8,
    "n_layer": 2,
    "d_inner": 32,
    "vocab_size": 10,
    "resid_dropout": 0.0,
    "embed_dropout": 0.1,
    "fused_mlp": False,
    "fused_dropout_add_ln": True,
    "residual_in_fp32": True,
    "pad_vocab_size_multiple": 8,
    "return_hidden_state": True,
    "layer": {
        "_name_": "hyena",
        "emb_dim": 5,
        "filter_order": 64, 
        "local_order": 3,
        "l_max": num_features,
        "modulate": True,
        "w": 10,
        "lr": learning_rate,
        "wd": 0.0,
        "lr_pos_emb": 0.0
    }
}


checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = backbone_cfg

additional_config = {
    "d_model": 8,
    "n_layer": 2,
    "d_inner": 32,
    "vocab_size": 10,
    "batch_size": batch_size,
    "val_size": val_size,
    "checkpoint_path": checkpoint_path,
    "num_epochs": num_epochs,
    "num_features": num_features,
    "hidden1": hidden1,
    "hidden2": hidden2,
    "L1_Lambda": lambda_,
}

config.update(additional_config)

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

pretrained_model_name = None 

# instantiate the model (pretrained here)
if pretrained_model_name in ['hyenadna-tiny-1k-seqlen']:
    # use the pretrained Huggingface wrapper instead
    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        download=True,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes,
    )

# from scratch
else:
    model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes, l_output=l_output, mode= mode)

# loss function
loss_fn = nn.MSELoss()

# create optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.to(device)

num_training_steps = num_epochs * len(train_loader)

scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps 
)

wandb.watch(model, loss_fn, log="all", log_freq=200, log_graph=True)


dtype = torch.cuda.FloatTensor

def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    
    total_loss = 0.
    total_mae = 0.  
    total_r2 = 0.  
    total_pearson = 0. 
    total_pvalue = 0. 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_,_ = model(data)
        
        output = output.squeeze()
        target = target.squeeze()
             
        loss = loss_fn(output, target)

        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in model.fc.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
       
        for name, param in model.linear_layer.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
        
        regularized_loss = loss + lambda_ * l1_regularization                       
        regularized_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * data.size(0)
        total_mae += mean_absolute_error(target.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
        total_r2 += r2_score(target.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
        _pearsonr, _pvalue = pearsonr(target.cpu().numpy(), output.detach().cpu().numpy())
        total_pearson += _pearsonr * data.size(0)
        total_pvalue += _pvalue * data.size(0)
                        
    train_loss = total_loss / len(train_loader.dataset)
    train_mae = total_mae / len(train_loader.dataset)
    train_rsquare = total_r2 / len(train_loader.dataset)
    train_pearson = total_pearson / len(train_loader.dataset)
    train_pvalue = total_pvalue / len(train_loader.dataset)

    return train_loss, train_mae, train_rsquare, train_pearson, train_pvalue


def test(model, device, test_loader, loss_fn):
    """Test loop."""
    model.eval()
    test_loss = 0
    
    total_loss = 0.
    total_mae = 0.
    total_r2 = 0. 
    total_pearson = 0. 
    total_pvalue = 0.  
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_,_ = model(data)
            
            output = output.squeeze()
            target = target.squeeze()
            
            loss = loss_fn(output.squeeze(), target.squeeze())
            
            total_loss += loss.item() * data.size(0)
            total_mae += mean_absolute_error(target.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
            total_r2 += r2_score(target.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
            _pearsonr, _pvalue = pearsonr(target.cpu().numpy(), output.detach().cpu().numpy())
            total_pearson += _pearsonr * data.size(0)
            total_pvalue += _pvalue * data.size(0)

    val_loss = total_loss / len(test_loader.dataset)
    val_mae = total_mae / len(test_loader.dataset)
    val_rsquare = total_r2 / len(test_loader.dataset)
    val_pearson = total_pearson / len(test_loader.dataset)
    val_pvalue = total_pvalue / len(test_loader.dataset)

    return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

def run_train():
        
    best_val_loss = float('inf')
    
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric='epoch')
    wandb.define_metric("val_loss", step_metric='epoch')
    wandb.define_metric("train_mae", step_metric='epoch')
    wandb.define_metric("val_mae", step_metric='epoch')
    wandb.define_metric("train_r2", step_metric='epoch')
    wandb.define_metric("val_r2", step_metric='epoch')
    wandb.define_metric("train_pearson", step_metric='epoch')
    wandb.define_metric("val_pearson", step_metric='epoch')
    wandb.define_metric("train_pvalue", step_metric='epoch')
    wandb.define_metric("val_pvalue", step_metric='epoch')
    wandb.define_metric("learning_rate", step_metric='epoch')


    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_mae, train_rsquare, train_pearson, train_pvalue = train(model, device, train_loader, optimizer, epoch, loss_fn)
        val_loss, val_mae, val_rsquare, val_pearson, val_pvalue = test(model, device, test_loader, loss_fn)
        elapsed = time.time() - epoch_start_time
        optimizer.step()
        
        print('-' * 180)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'train loss {train_loss:5.4f} | valid loss {val_loss:5.4f} | train MAE {train_mae:5.4f} | '
            f'valid MAE {val_mae:5.4f} | train R^2 {train_rsquare:5.4f} | valid R^2 {val_rsquare:5.4f} |'
            f'learning rate {scheduler.get_last_lr()[0]:5.7f} |'
            )

        # print pearson correlation and p-value
        print(f'| train pearson {train_pearson:5.4f} | valid pearson {val_pearson:5.4f} | train p-value {train_pvalue:5.4f} | valid p-value {val_pvalue:5.4f} |')

        print('-' * 180)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        wandb.log({"train_mae": train_mae, "val_mae": val_mae, "epoch": epoch})
        wandb.log({"train_r2": train_rsquare, "val_r2": val_rsquare, "epoch": epoch})
        wandb.log({"learning_rate": scheduler.get_last_lr()[0], "epoch": epoch})
        wandb.log({"train_pearson": train_pearson, "val_pearson": val_pearson, "epoch": epoch})
        wandb.log({"train_pvalue": train_pvalue, "val_pvalue": val_pvalue, "epoch": epoch})

        wandb.log({"epoch": epoch, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            checkpoint(model, checkpoint_path + "best_model.pth", epoch, val_loss, optimizer, scheduler)
            
        elif epoch - best_epoch > patience:
            print("Early stopped training at epoch %d" % epoch)
            break 

run_train()
run.finish()