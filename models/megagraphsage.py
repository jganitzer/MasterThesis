import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from models.megaencoder import MegaChunkRegressor_Sage, MegaChunkRegressor_MLP, MegaRegressor_MLP, MegaChunkRegressor_MLP2, MegaChunkRegressor_MLP_cat
from models.graphsage import SAGE_OneHot2, SAGE_OneHot_MLP
from models.hyenaEncoder import HyenaRegressor_MLP, HyenaRegressor_MLP_cat_Reduced, HyenaRegressor_MLP_Reduced, HyenaRegressor_MLP_cat
from models.bertencoder import BertRegressor_MLP, BertRegressor_MLP2

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from models.hyenaEncoderMLM import HyenaForMaskedLM
from transformers import MegaModel, MegaConfig


class MegaGraphSage_MLP(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()

        self.mega_model = MegaChunkRegressor_Sage(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, chunk_size=chunk_size)
        self.sage_model = SAGE_OneHot2(in_channels, hidden_channels, out_channels)
            
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        self.num_features = num_features
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, attention_mask, batch_size):
    
        mega_pooled_output = self.mega_model(inputs_embeds, attention_mask)
        sage_embeddings = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.cat((mega_pooled_output, sage_embeddings[:batch_size]), dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        mask = data.val_mask
        target = data.y.to(device)

        xHatGraphSage = self.sage_model.inference(data.x, subgraph_loader, device)
            
        # Create a TensorDataset from the input tensors
        dataset = TensorDataset(xHatGraphSage[mask], data.x_pad[mask], target[mask], data.attention_mask[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0.  
        total_pearson = 0. 
        total_pvalue = 0. 
        total_pearson_numpy = 0.
        
        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets, attention_mask_val in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)
                    attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)

                    mega_pooled_output = self.mega_model(xMegaChunk, attention_mask_val)
                    concatenated_outputs = torch.cat((mega_pooled_output, xHatGraphSage), dim=1)

                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)
                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                    total_pearson_numpy += (np.corrcoef(y_hat[:batch.batch_size].detach().cpu().numpy(), targets.cpu().numpy())[0, 1]) * xMegaChunk.size(0)
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        val_pearson_numpy = total_pearson_numpy / len(dataloader.dataset)
        print("val_pearson_numpy: ", val_pearson_numpy)

        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue
        
        
class MegaGraphSage_3MLP(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()

        self.mega_model = MegaChunkRegressor_MLP(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, chunk_size=chunk_size)
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, attention_mask, batch_size):

        self.mega_model.train()
        self.sage_model.train()
        
        mega_pooled_output, _, _  = self.mega_model(inputs_embeds, attention_mask)
        sage_embeddings, _ = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.stack([mega_pooled_output, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        
        self.mega_model.eval()
        self.sage_model.eval()

        mask = data.val_mask
        target = data.y.to(device)
        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x_pad[mask], target[mask], data.attention_mask[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0. 
        total_pearson = 0.
        total_pvalue = 0. 
        
        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets, attention_mask_val in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)
                    attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)

                    mega_pooled_output, _, _ = self.mega_model(xMegaChunk, attention_mask_val)
                    concatenated_outputs = torch.stack([mega_pooled_output, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                                    
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)

        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

class MegaGraphSage_3MLP_Reduced(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()

        self.mega_model = MegaChunkRegressor_MLP2(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, chunk_size=chunk_size)
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, attention_mask, batch_size):

        self.mega_model.train()
        self.sage_model.train()
        
        mega_pooled_output, _, _  = self.mega_model(inputs_embeds, attention_mask)
        sage_embeddings, _ = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.stack([mega_pooled_output, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        
        self.mega_model.eval()
        self.sage_model.eval()

        mask = data.val_mask
        target = data.y.to(device)
        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x_pad[mask], target[mask], data.attention_mask[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0. 
        total_r2 = 0.  
        total_pearson = 0. 
        total_pvalue = 0.

        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets, attention_mask_val in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)
                    attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)

                    mega_pooled_output, _, _ = self.mega_model(xMegaChunk, attention_mask_val)
                    concatenated_outputs = torch.stack([mega_pooled_output, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

    @torch.no_grad()
    def inference_gen30(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        
        self.mega_model.eval()
        self.sage_model.eval()

        mask = data.gen30_mask
        target = data.y.to(device)
        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x_pad[mask], target[mask], data.attention_mask[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0.  
        total_pearson = 0.
        total_pvalue = 0.  
        
        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets, attention_mask_val in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)
                    attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)

                    mega_pooled_output, _, _ = self.mega_model(xMegaChunk, attention_mask_val)
                    concatenated_outputs = torch.stack([mega_pooled_output, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                                    
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue   

        
        
class HyenaGraphSage_3MLP(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()

        self.mega_model = HyenaRegressor_MLP(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation)
        
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):
           
        mega_pooled_output = self.mega_model(inputs_embeds)
        sage_embeddings, _ = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.stack([mega_pooled_output, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device, mask):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0. 
        total_r2 = 0.  
        total_pearson = 0.  
        total_pvalue = 0.  
        
        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)

                    mega_pooled_output = self.mega_model(xMegaChunk)
                    concatenated_outputs = torch.stack([mega_pooled_output, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

class HyenaGraphSage_3MLP_Reduced(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()

        self.mega_model = HyenaRegressor_MLP_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation)

        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
           
        hyena_logits,_ ,_ = self.mega_model(inputs_embeds.type(dtype1))
        sage_embeddings, _ = self.sage_model(x, edge_index)

        concatenated_outputs = torch.stack([hyena_logits, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        
        mask = data.val_mask
        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0.  
        total_pearson = 0.  
        total_pvalue = 0.  

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype1) 
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype1)

                    hyena_logits,_,_ = self.mega_model(xMegaChunk)
                    concatenated_outputs = torch.stack([hyena_logits, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

class HyenaGraphSage_3MLP_cat(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()

        self.mega_model = HyenaRegressor_MLP_cat(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation)

        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):

        self.mega_model.train()
        self.sage_model.train()

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
           
        hyena_logits,_ ,_ = self.mega_model(inputs_embeds.type(dtype2))
        sage_embeddings, _ = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.stack([hyena_logits, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device, mask):        
        target = data.y.to(device)
        sage_embeddings, _ = self.sage_model.inference(data.x_oneHot, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0. 
        total_pearson = 0.
        total_pvalue = 0. 
        
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype2)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype1)

                    hyena_logits,_,_ = self.mega_model(xMegaChunk)
                    concatenated_outputs = torch.stack([hyena_logits, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue


class HyenaGraphSage_3MLP_cat_Reduced(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()

        self.mega_model = HyenaRegressor_MLP_cat_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation)

        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
           
        hyena_logits,_ ,_ = self.mega_model(inputs_embeds.type(dtype2))
        sage_embeddings, _ = self.sage_model(x, edge_index)

        concatenated_outputs = torch.stack([hyena_logits, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device, mask):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        target = data.y.to(device)
        sage_embeddings, _ = self.sage_model.inference(data.x_oneHot, subgraph_loader, device)

        dataset = TensorDataset(sage_embeddings[mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0. 
        total_r2 = 0.
        total_pearson = 0.  
        total_pvalue = 0. 
                
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype2)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype1)

                    hyena_logits,_,_ = self.mega_model(xMegaChunk)
                    concatenated_outputs = torch.stack([hyena_logits, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue
  

class HyenaGraphSage_3MLP_cat_Reduced_Pretrained(nn.Module):
    def __init__(self, num_features, hyenaPreTrainedPath, sagePreTrainedPath, in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()

        self.mega_model = HyenaRegressor_MLP_cat_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation)       
        checkpoint = torch.load(hyenaPreTrainedPath)
        self.mega_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        checkpoint = torch.load(sagePreTrainedPath)
        self.sage_model.load_state_dict(checkpoint['model_state_dict'])
        
        for param in self.mega_model.parameters():
            param.requires_grad = False

        for param in self.sage_model.parameters():
            param.requires_grad = False
        
        self.fc = torch.nn.Linear(2, 1)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
           
        hyena_logits,_ ,_ = self.mega_model(inputs_embeds.type(dtype2))
        sage_embeddings, _ = self.sage_model(x, edge_index)

        concatenated_outputs = torch.stack([hyena_logits, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device, mask):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x_oneHot, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.
        total_r2 = 0.  
        total_pearson = 0. 
        total_pvalue = 0.
        
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype2)
                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype1)

                    hyena_logits,_,_ = self.mega_model(xMegaChunk)
                    concatenated_outputs = torch.stack([hyena_logits, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

class HyenaGraphSage_1MLP_cat_Reduced_Pretrained(nn.Module):
    def __init__(self, num_features, hyenaPreTrainedPath, sagePreTrainedPath, in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'relu'):
        super().__init__()
        
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
        
        pretrainedModel = HyenaForMaskedLM(config, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=26284, order = order, activation = activation)
        pretrainedModel.load_state_dict(torch.load(hyenaPreTrainedPath))
        
        self.transformer_encoder = pretrainedModel.mega
        
        
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)
        checkpoint = torch.load(sagePreTrainedPath)
        self.sage_model.load_state_dict(checkpoint['model_state_dict'])

        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

        for param in self.sage_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(num_features + hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        self.linear_layer = nn.Linear(input_dim, 1)

        self.linear_layer.apply(self.init_weights)
            
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
           
        output = self.transformer_encoder(inputs_embeds.type(dtype2))
        reduced_tensor = self.linear_layer(output).squeeze()
        
        sageLogits, sage_embeddings = self.sage_model(x, edge_index)
        
        concatenated_outputs = torch.cat((sage_embeddings[-1][:batch_size], reduced_tensor), dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device, mask):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        target = data.y.to(device)

        sageLogits, sage_embeddings = self.sage_model.inference(data.x_oneHot, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[-1][mask], data.x[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0. 
        total_r2 = 0.
        total_pearson = 0.  
        total_pvalue = 0. 
        
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype2)
                    xHatGraphSage = xHatGraphSage.to(device)#.type(dtype)
                    
                    targets = targets.to(device).type(dtype1)

                    output = self.transformer_encoder(xMegaChunk)
                    reduced_tensor = self.linear_layer(output).squeeze()
                    
                    concatenated_outputs = torch.cat((xHatGraphSage, reduced_tensor), dim=1)                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                    
                
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)
        
        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

  
class BertGraphSage_MLP(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 128, hidden2 = 64,  hidden3 = 32, input_dim=3, num_layers=2, num_heads=3, max_pos=256):
        super().__init__()
             
        self.bert_model = BertRegressor_MLP2(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, num_features = hidden_channels)     
        
        self.sage_model = SAGE_OneHot2(in_channels, hidden_channels, out_channels)
        
        self.num_features = num_features
        self.input_dim = input_dim

        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, batch_size):
        self.sage_model.eval()
        sage_embeddings = self.sage_model(x, edge_index)
    
        tensor_3d = (sage_embeddings[:batch_size]).unsqueeze(2)

        # Repeat the tensor along the last dimension to obtain a tensor of shape (64, 256, 3)
        tensor_3d_expanded = tensor_3d.expand(-1, -1,  self.input_dim)
        
        bert_output  = self.bert_model(tensor_3d_expanded)

        return bert_output
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        mask = data.val_mask
        target = data.y.to(device)

        sage_embeddings = self.sage_model.inference(data.x, subgraph_loader, device)
                    
        dataset = TensorDataset(sage_embeddings[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0.  
        total_pearson = 0.  
        total_pvalue = 0.  

        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, targets in dataloader:
                with torch.cuda.amp.autocast():

                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)

                    # Convert the 2D tensor to a 3D tensor
                    tensor_3d = xHatGraphSage.unsqueeze(2)

                    # Repeat the tensor along the last dimension to obtain a tensor of shape (64, 256, 3)
                    tensor_3d_expanded = tensor_3d.expand(-1, -1, self.input_dim)

                    output = self.bert_model(tensor_3d_expanded)
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xHatGraphSage.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xHatGraphSage.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xHatGraphSage.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xHatGraphSage.size(0)
                    total_pvalue += _pvalue * xHatGraphSage.size(0)
                                    
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)

        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue
        
class MegaGraphSage_MLP2(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 128, hidden2 = 64,  hidden3 = 32, input_dim=3, num_layers=2, num_heads=3, max_pos=35000):
        super().__init__()

        self.mega_model = MegaRegressor_MLP(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, num_features = hidden_channels)

        self.sage_model = SAGE_OneHot2(in_channels, hidden_channels, out_channels)
        
        self.num_features = num_features

        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, attention_mask, batch_size):
    
        sage_embeddings = self.sage_model(x, edge_index)
        
        bert_output, _  = self.mega_model((sage_embeddings[:batch_size]).unsqueeze(2))

        return bert_output
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:

        mask = data.val_mask
        target = data.y.to(device)

        sage_embeddings = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], target[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0.  
        total_pearson = 0. 
        total_pvalue = 0. 
        
        dtype = torch.cuda.FloatTensor
        
        with torch.no_grad():
            for xHatGraphSage, targets in dataloader:
                with torch.cuda.amp.autocast():

                    xHatGraphSage = xHatGraphSage.to(device)
                    
                    targets = targets.to(device).type(dtype)

                    output, _ = self.mega_model(xHatGraphSage.unsqueeze(2))
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xHatGraphSage.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xHatGraphSage.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xHatGraphSage.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xHatGraphSage.size(0)
                    total_pvalue += _pvalue * xHatGraphSage.size(0)
                                    
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)

        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue
        

class MegaGraphSage_3MLP_cat(nn.Module):
    def __init__(self, num_features,in_channels, hidden_channels = 256, out_channels = 1, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()

        #self.mega_model = MegaChunkRegressor_MLP2(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, chunk_size=chunk_size)
        self.mega_model = MegaChunkRegressor_MLP_cat(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, chunk_size=chunk_size)
        
        self.sage_model = SAGE_OneHot_MLP(in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64)      
        
        self.fc = torch.nn.Linear(2, 1)
       
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self,x, edge_index, inputs_embeds, attention_mask, batch_size):

        self.mega_model.train()
        self.sage_model.train()
        
        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        mega_pooled_output, _, _  = self.mega_model(inputs_embeds.type(dtype2), attention_mask)
        sage_embeddings, _ = self.sage_model(x, edge_index)

        concatenated_outputs = torch.stack([mega_pooled_output, sage_embeddings[:batch_size]], dim=1)

        logits = self.fc(concatenated_outputs).squeeze() 

        return logits
    
    @torch.no_grad()
    def inference(self, subgraph_loader, data, batch_size, criterion, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        
        self.mega_model.eval()
        self.sage_model.eval()

        mask = data.val_mask
        target = data.y.to(device)
        target = data.y.to(device)

        sage_embeddings, _ = self.sage_model.inference(data.x, subgraph_loader, device)
            
        dataset = TensorDataset(sage_embeddings[mask], data.x_pad[mask], target[mask], data.attention_mask[mask])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.
        total_mae = 0.  
        total_r2 = 0. 
        total_pearson = 0. 
        total_pvalue = 0. 

        dtype1 = torch.cuda.FloatTensor
        dtype2 = torch.cuda.IntTensor
        
        with torch.no_grad():
            for xHatGraphSage, xMegaChunk, targets, attention_mask_val in dataloader:
                with torch.cuda.amp.autocast():
                    xMegaChunk = xMegaChunk.to(device).type(dtype2)
                    xHatGraphSage = xHatGraphSage.to(device)#.type(dtype)
                    
                    targets = targets.to(device).type(dtype1)
                    attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)

                    mega_pooled_output, _, _ = self.mega_model(xMegaChunk, attention_mask_val)
                    concatenated_outputs = torch.stack([mega_pooled_output, xHatGraphSage], dim=1)
                    
                    output = self.fc(concatenated_outputs).squeeze() 
                    
                    loss = criterion(output, targets)

                    total_loss += loss.item() * xMegaChunk.size(0)
                    total_mae += mean_absolute_error(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    total_r2 += r2_score(targets.cpu().numpy(), output.detach().cpu().numpy()) * xMegaChunk.size(0)
                    _pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output.detach().cpu().numpy())

                    total_pearson += _pearsonr * xMegaChunk.size(0)
                    total_pvalue += _pvalue * xMegaChunk.size(0)
                                    
        val_loss = total_loss / len(dataloader.dataset)
        val_mae = total_mae / len(dataloader.dataset)
        val_rsquare = total_r2 / len(dataloader.dataset)
        val_pearson = total_pearson / len(dataloader.dataset)
        val_pvalue = total_pvalue / len(dataloader.dataset)

        return val_loss, val_mae, val_rsquare, val_pearson, val_pvalue

