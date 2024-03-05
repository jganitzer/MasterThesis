import wandb
import torch
import torch.nn as nn
import time
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from models.hyenaEncoder import HyenaRegressor_MLP_Reduced
from utils.torchDataloader import OneHotLoader
from utils.helper import checkpoint, load_checkpoint
#from torcheval.metrics import R2Score
from pathlib import Path
from transformers import get_scheduler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"
wandb_run_name = "_"

resume_training = False

entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

X_path = "/train/snp_X_train_maf_one_hot.parquet"
y_path = "/train/snp_y_train_maf_one_hot.parquet"

#Config

#HyenaOperator
order = 2 
activation = 'relu'

#paddedLength = 26283
#max_pos = 35000
max_pos = 27000


lr = 0.001 

batch_size = 64 
num_epochs = 40
val_size = 0.01

input_dim = 3
num_layers = 6
num_heads = 1

gamma = 0.95


patience = 10

hidden1 = 1000
hidden2 = 500
hidden3 = 200

lambda_ = 0.001

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "Preprocessing": "maf",
    "Hyena_Activation" : activation,
    "Hyena_Order" : order,
    "lr": lr,
    "Optimizer": "AdamW",
    "L1_Lambda": lambda_,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "val_size": val_size,
    "input_dim": input_dim,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "max_pos": max_pos,
    "FFNN_layers": 2,
    "hidden1": hidden1,
    "hidden2": hidden2,
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)

oneHotLoader = OneHotLoader(dataRoot + heritability, X_path, y_path, wandbDir=wandbDir, batch_size = batch_size, val_size = val_size)
#oneHotLoader = OneHotMaskLoader(dataRoot + heritability, X_path, y_path, wandbDir=wandbDir, batch_size = batch_size, val_size = 0.2, paddedLength = paddedLength)

train_loader, val_loader = oneHotLoader.getLoaders()

num_features = train_loader.dataset.tensors[0].shape[1]

#model = BigBirdRegressorHF(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos).to(device)
#model = HyenaRegressor_MLP(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation).to(device)
model = HyenaRegressor_MLP_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr)

num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = num_training_steps * 0.05

scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
criterion = nn.MSELoss()

if resume_training == True:
    model, optimizer, scheduler, start_epoch = load_checkpoint("/data/checkpoints/" + wandb_run_name + "/best_model.pth", model, optimizer, scheduler)
    optimizer.param_groups[0]['lr'] = lr
    
def train(model: nn.Module) -> float:
    model.train() 
    total_loss = 0.
    total_mae = 0.  
    total_r2 = 0.  
    total_pearson = 0.
    total_pvalue = 0. 
    
    dtype = torch.cuda.FloatTensor
    
    num_batches = len(train_loader)
    for batch, (data, targets) in enumerate(train_loader):
        with torch.cuda.amp.autocast():

            data = data.to(device).type(dtype)
            targets = targets.to(device).type(dtype)

            optimizer.zero_grad()
            
            output,_,_ = model(data)
            loss = criterion(output, targets)

            l1_regularization = torch.tensor(0., requires_grad=True)
            for name, param in model.fc.named_parameters():
                if 'bias' not in name:
                    l1_regularization = l1_regularization + torch.norm(param, p=1)
           
            for name, param in model.linear_layer.named_parameters():
                if 'bias' not in name:
                    l1_regularization = l1_regularization + torch.norm(param, p=1)
            
            regularized_loss = loss + lambda_ * l1_regularization                       
            regularized_loss.backward()
                        
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * data.size(0)
            total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
            total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
            _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())
            total_pearson += _pearsonr * data.size(0)
            total_pvalue += _pvalue * data.size(0)
    
        train_loss = total_loss / len(train_loader.dataset)
        train_mae = total_mae / len(train_loader.dataset)
        train_rsquare = total_r2 / len(train_loader.dataset)
        train_pearson = total_pearson / len(train_loader.dataset)
        train_pvalue = total_pvalue / len(train_loader.dataset)

    return train_loss, train_mae, train_rsquare, train_pearson, train_pvalue

def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    model.eval()  
    total_loss = 0
    total_mae = 0. 
    total_r2 = 0.  
    total_pearson = 0. 
    total_pvalue = 0.  
    
    dtype = torch.cuda.FloatTensor
    
    with torch.no_grad():
        for data, targets in eval_loader:
            with torch.cuda.amp.autocast():
                data = data.to(device).type(dtype)
                targets = targets.to(device).type(dtype)

                output,_,_ = model(data)
                loss = criterion(output, targets)

                # Update loss and metric variables
                total_loss += loss.item() * data.size(0)
                total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
                total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * data.size(0)
                _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())
                total_pearson += _pearsonr * data.size(0)
                total_pvalue += _pvalue * data.size(0)
            
    val_loss = total_loss / len(eval_loader.dataset)
    val_mae = total_mae / len(eval_loader.dataset)
    val_rsquare = total_r2 / len(eval_loader.dataset)
    val_pearson = total_pearson / len(eval_loader.dataset)
    val_pvalue = total_pvalue / len(eval_loader.dataset)

    return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

best_val_loss = float('inf')
window_size = 4096

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
    train_loss, train_mae, train_rsquare, train_pearson, train_pvalue = train(model)
    val_loss, val_mae, val_rsquare, val_pearson, val_pvalue = evaluate(model, val_loader)
    elapsed = time.time() - epoch_start_time
    print('-' * 180)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'train loss {train_loss:5.4f} | valid loss {val_loss:5.4f} | train MAE {train_mae:5.4f} | '
        f'valid MAE {val_mae:5.4f} | train R^2 {train_rsquare:5.4f} | valid R^2 {val_rsquare:5.4f} |'
        f'learning rate {scheduler.get_last_lr()[0]:5.7f} |'
        )

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

run.finish()
