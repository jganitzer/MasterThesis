import wandb
import torch
import torch.nn as nn
import time
import pandas as pd
import torch.optim as optim
from models.graphsage import SAGE_OneHot_MLP
from utils.helper import load_checkpoint, checkpoint
from pathlib import Path
from transformers import get_scheduler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime

from torch_geometric.loader import NeighborLoader
import copy

from torch_geometric.utils import to_undirected

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

lr = 0.001
end_lr = 0.0001 
const_epoch = 5

batch_size = 64 
num_epochs = 40
embed_dim = 256
neighbors = 6

val_size = 0.1
input_dim = 3
hidden1= 128
hidden2= 64

patience = 15
lambda_ = 0.001 

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "lr": lr,
    "Optimizer": "AdamW",
    "L1_Lambda": lambda_,
    "embed_dim": embed_dim,
    "neighbors": neighbors,
    "hops": 3,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "val_size": val_size,
    "input_dim": input_dim,
    "FFNN_layers": 2,
    "hidden1": hidden1,
    "hidden2": hidden2,
    "checkpoint_path": checkpoint_path,
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)

data = torch.load(dataRoot + heritability + "/graph_maf_float32_gen30_SireGeno.pt")

data.fineTrain_mask = data.train_mask | data.val_mask | data.test_mask

data.edge_index = to_undirected(data.edge_index)

kwargs = {'batch_size': batch_size, 'num_workers': 0, 'persistent_workers': False}


train_loader = NeighborLoader(copy.copy(data), num_neighbors = [neighbors] * 2, shuffle=True,
                                  input_nodes=data.fineTrain_mask, **kwargs)

val_loader = NeighborLoader(copy.copy(data), num_neighbors=[neighbors] * 2, shuffle=False, 
                                input_nodes=data.val_mask, **kwargs)
                                
subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

model = SAGE_OneHot_MLP(data.x.size(1) * data.x.size(2), embed_dim, 1,hidden1 = hidden1, hidden2 = hidden2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

num_train_samples = len(train_loader.dataset)
num_training_steps = (num_train_samples // batch_size) * num_epochs

scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps 
)

criterion = torch.nn.MSELoss()

if resume_training == True:
    model, optimizer, scheduler, start_epoch = load_checkpoint("/data/checkpoints/" + wandb_run_name + "/best_model.pth", model, optimizer, scheduler)
    optimizer.param_groups[0]['lr'] = lr
    
wandb.watch(model, criterion, log="all", log_freq=200, log_graph=True)

def train(model: nn.Module) -> float:
    model.train()  
    total_loss = 0.
    total_examples = 0
    total_mae = 0. 
    total_r2 = 0.
    total_pearson = 0. 
    total_pvalue = 0.  
    dtype = torch.cuda.FloatTensor

    num_batches = len(train_loader)
    for batch in train_loader:
        with torch.cuda.amp.autocast():
            
            optimizer.zero_grad()
            
            batch = batch.to(device)
            edge_index = batch.edge_index.to(device)
            
            targets = batch.y[:batch.batch_size]
            
            y_hat, embeddings = model(batch.x, batch.edge_index.to(device))
        
            loss = criterion(y_hat[:batch.batch_size], targets)

            l1_regularization = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_regularization = l1_regularization + torch.norm(param, p=1)
            
            regularized_loss = loss + lambda_ * l1_regularization                       
            
            regularized_loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += float(loss) * batch.batch_size
                       
            total_mae += mean_absolute_error(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy()) * batch.batch_size
            total_r2 += r2_score(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy()) * batch.batch_size
            _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy())
            total_pearson += _pearsonr * batch.batch_size
            total_pvalue += _pvalue * batch.batch_size
                     
            total_examples += batch.batch_size

    train_loss = total_loss / total_examples
    train_mae = total_mae / total_examples
    train_rsquare = total_r2 / total_examples
    train_pearson = total_pearson / total_examples
    train_pvalue = total_pvalue / total_examples
    
    return train_loss, train_mae, train_rsquare, train_pearson, train_pvalue       

@torch.no_grad()
def evaluate(model: nn.Module) -> float:
    model.eval() 
    total_loss = 0.
    total_mae = 0.
    total_r2 = 0.
    total_pearson = 0.
    total_pvalue = 0. 
    total_examples = 0
    
    dtype = torch.cuda.FloatTensor
    
    with torch.no_grad():
        for batch in val_loader:
            with torch.cuda.amp.autocast():
                batch = batch.to(device)
                edge_index = batch.edge_index.to(device)
                
                targets = batch.y[:batch.batch_size]
                
                y_hat, embeddings = model(batch.x, batch.edge_index.to(device))
            
                loss = criterion(y_hat[:batch.batch_size], targets)

                total_loss += float(loss) * batch.batch_size              

                total_mae += mean_absolute_error(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy()) * batch.batch_size
                total_r2 += r2_score(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy()) * batch.batch_size
                _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), y_hat[:batch.batch_size].detach().cpu().numpy())
                total_pearson += _pearsonr * batch.batch_size
                total_pvalue += _pvalue * batch.batch_size

                total_examples += batch.batch_size

        val_loss = total_loss / total_examples
        val_mae = total_mae / total_examples
        val_rsquare = total_r2 / total_examples
        val_pearson = total_pearson / total_examples
        val_pvalue = total_pvalue / total_examples

    #pbar.close()

    return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue
    
def evaluate2(model: nn.Module) -> float:
    model.eval() 
    total_loss = 0.
    total_mae = 0.
    total_r2 = 0.  
    total_pearson = 0. 
    total_pvalue = 0. 
        
    dtype = torch.cuda.FloatTensor

    y_hat, _ = model.inference(data.x, subgraph_loader, device)
    target = data.y.to(y_hat.device)

    total_loss = []
    total_mae = []
    total_r2 = []
    total_pearson = []
    total_pvalue = []

    mask = data.val_mask

    loss = criterion(y_hat[mask], target[mask])
    mae = mean_absolute_error(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    r2 = r2_score(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    _pearsonr, _pvalue = pearsonr(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
        
    total_loss.append(loss.item())
    total_mae.append(mae)
    total_r2.append(r2)
    total_pearson.append(_pearsonr)
    total_pvalue.append(_pvalue)

    return total_loss[0], total_mae[0], total_r2[0], total_pearson[0], total_pvalue[0]

def evaluate_gen30(model: nn.Module) -> float:
    model.eval()  
    total_loss = 0.
    total_mae = 0. 
    total_r2 = 0.  
    total_pearson = 0.
    total_pvalue = 0. 
    
    total_loss_gen30 = 0.
    total_mae_gen30 = 0.
    total_r2_gen30 = 0. 
    total_pearson_gen30 = 0.
    total_pvalue_gen30 = 0. 
        
    dtype = torch.cuda.FloatTensor

    y_hat, _ = model.inference(data.x, subgraph_loader, device)
    target = data.y.to(y_hat.device)

    total_loss = []
    total_mae = []
    total_r2 = []
    total_pearson = []
    total_pvalue = []
    
    total_loss_gen30 = []
    total_mae_gen30 = []
    total_r2_gen30 = []
    total_pearson_gen30 = []
    total_pvalue_gen30 = []

    mask = data.val_mask

    loss = criterion(y_hat[mask], target[mask])
    mae = mean_absolute_error(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    r2 = r2_score(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    _pearsonr, _pvalue = pearsonr(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
        
    total_loss.append(loss.item())
    total_mae.append(mae)
    total_r2.append(r2)
    total_pearson.append(_pearsonr)
    total_pvalue.append(_pvalue)
    
    mask = data.gen30_mask

    loss = criterion(y_hat[mask], target[mask])
    mae = mean_absolute_error(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    r2 = r2_score(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
    _pearsonr, _pvalue = pearsonr(target[mask].cpu().numpy(), y_hat[mask].detach().cpu().numpy())
        
    total_loss_gen30.append(loss.item())
    total_mae_gen30.append(mae)
    total_r2_gen30.append(r2)
    total_pearson_gen30.append(_pearsonr)
    total_pvalue_gen30.append(_pvalue)
    

    return total_loss[0], total_mae[0], total_r2[0], total_pearson[0], total_pvalue[0], total_loss_gen30[0], total_mae_gen30[0], total_r2_gen30[0], total_pearson_gen30[0], total_pvalue_gen30[0]


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

#GEN30
wandb.define_metric("gen30_loss", step_metric='epoch')
wandb.define_metric("gen30_mae", step_metric='epoch')
wandb.define_metric("gen30_r2", step_metric='epoch')
wandb.define_metric("gen30_pearson", step_metric='epoch')
wandb.define_metric("gen30_pvalue", step_metric='epoch')


for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    train_loss, train_mae, train_rsquare, train_pearson, train_pvalue = train(model)

    val_loss, val_mae, val_rsquare, val_pearson, val_pvalue, val_loss_gen30, val_mae_gen30, val_rsquare_gen30, val_pearson_gen30, val_pvalue_gen30 = evaluate_gen30(model)
    
    elapsed = time.time() - epoch_start_time
    print('-' * 180)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'train loss {train_loss:5.4f} | valid loss {val_loss:5.4f} | train MAE {train_mae:5.4f} | '
        f'valid MAE {val_mae:5.4f} | train R^2 {train_rsquare:5.4f} | valid R^2 {val_rsquare:5.4f} |'
        f'learning rate {scheduler.get_last_lr()[0]:5.7f} |'
        )
    print(f'| train pearson {train_pearson:5.4f} | valid pearson {val_pearson:5.4f} | train p-value {train_pvalue:5.4f} | valid p-value {val_pvalue:5.4f} |')
        
    #GEN30
    print(f'| gen30 loss {val_loss_gen30:5.4f} | gen30 MAE {val_mae_gen30:5.4f} | gen30 R^2 {val_rsquare_gen30:5.4f} | gen30 pearson {val_pearson_gen30:5.4f} | gen30 p-value {val_pvalue_gen30:5.4f} |')
        
    print('-' * 180)

    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
    wandb.log({"train_mae": train_mae, "val_mae": val_mae, "epoch": epoch})
    wandb.log({"train_r2": train_rsquare, "val_r2": val_rsquare, "epoch": epoch})
    wandb.log({"learning_rate": scheduler.get_last_lr()[0], "epoch": epoch})
    wandb.log({"train_pearson": train_pearson, "val_pearson": val_pearson, "epoch": epoch})
    wandb.log({"train_pvalue": train_pvalue, "val_pvalue": val_pvalue, "epoch": epoch})
    
    #GEN30
    wandb.log({"train_loss": train_loss, "gen30_loss": val_loss_gen30, "epoch": epoch})
    wandb.log({"train_mae": train_mae, "gen30_mae": val_mae_gen30, "epoch": epoch})
    wandb.log({"train_r2": train_rsquare, "gen30_r2": val_rsquare_gen30, "epoch": epoch})
    wandb.log({"train_pearson": train_pearson, "gen30_pearson": val_pearson_gen30, "epoch": epoch})
    wandb.log({"train_pvalue": train_pvalue, "gen30_pvalue": val_pvalue_gen30, "epoch": epoch})

    wandb.log({"epoch": epoch, "val_loss": val_loss, "epoch": epoch})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        checkpoint(model, checkpoint_path + "best_model.pth", epoch, val_loss, optimizer, scheduler)
        
    elif epoch - best_epoch > patience:
        print("Early stopped training at epoch %d" % epoch)
        break

run.finish()
