import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, Linear
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import GATConv, GATv2Conv

class SAGE_OneHot(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        embeddings = []
        # flatten the one-hot encoding
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        x = self.linear(x)
        return x.squeeze(), embeddings

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        for i, conv in enumerate(self.convs):
            with torch.cuda.amp.autocast():
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
                x = self.linear(x_all).squeeze()

        return x, embeddings        
        
class SAGE_OneHot2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        embeddings = []
        # flatten the one-hot encoding
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        return x
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
    
        return x_all


class SAGE_OneHot_MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #if i < len(self.convs) - 1:
            x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    #if i < len(self.convs) - 1:
                    x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings
    
class SAGE_OneHot_MLP2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        for i, conv in enumerate(self.convs):
            with torch.cuda.amp.autocast():
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
                x = self.fc(x_all).squeeze()

        return x, embeddings


class SAGE_OneHot_MLP_Concat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        x_all = torch.cat(embeddings, dim=1)
        x = self.fc(x_all).squeeze()
        return x, embeddings 
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        for i, conv in enumerate(self.convs):
            with torch.cuda.amp.autocast():
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
                x = self.fc(x_all).squeeze()

        return x, embeddings

class SAGE_OneHot_hetero(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.linear = Linear(-1, 1)

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        x = self.linear(x)
        return x.squeeze(), embeddings 

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())

                x_all = torch.cat(xs, dim=0)
                embeddings.append(x_all)
                
                x = self.linear(x_all)

        return x_all.squeeze(), embeddings
        
class SAGE_OneHot_MLP_hetero(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
                
        self.fc = nn.Sequential(
            Linear(-1, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            Linear(-1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            Linear(-1, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #if i < len(self.convs) - 1:
            x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        x = self.fc(x)
        return x.squeeze(), embeddings 

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        #if i < len(self.convs) - 1:
                        x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())

                x_all = torch.cat(xs, dim=0)
                embeddings.append(x_all)
                
            x = self.fc(x_all)

        return x_all.squeeze(), embeddings

class SAGE_OneHot_MLP_hetero2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #if i < len(self.convs) - 1:
            x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        
        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    #if i < len(self.convs) - 1:
                    x = x.relu_()
                    xs.append(x[:batch['female'].batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = self.convs[0](x, edge_index)
        x = x.relu_()
        embeddings.append(x)
        x = self.convs[1](x, edge_index)
        x = x.relu_()
        embeddings.append(x)

        return x
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []

        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    with torch.cuda.amp.autocast():
                        x = x_all[batch.n_id.to(x_all.device)].to(device)
                        x = conv(x, batch.edge_index.to(device))
                        #if i < len(self.convs) - 1:
                        x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            #x = self.fc(x_all).squeeze()

        return x_all
        
class SAGE_OneHot_MLP3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64, hidden3 = 32):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)
        
        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings
        
class SAGE_OneHot_MLP4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64, hidden3 = 32, hidden4 = 32):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #if i < len(self.convs) - 1:
            x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    #if i < len(self.convs) - 1:
                    x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings
        
class GAT_OneHot_MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64, heads = 4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads = heads)) 
        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads = heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads = 1, concat=False))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        # flatten the one-hot encoding
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    if i < len(self.convs) - 1:
                        x = F.elu(x)
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings
        
        
class SAGE_OneHot_Linear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # aggr *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`. / project = True /normalize = True
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.fc = torch.nn.Linear(hidden_channels, 1)
                
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        # flatten the one-hot encoding
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #if i < len(self.convs) - 1:
            x = x.relu_()
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    #if i < len(self.convs) - 1:
                    x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings

class GAT_OneHot_Linear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden1 = 128, hidden2 = 64, heads = 4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads = heads)) 
        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads = heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads = 1, concat=False))
        
        self.fc = torch.nn.Linear(hidden_channels, 1)
        
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        

    def forward(self, x, edge_index):
        embeddings = []
        x = x.flatten(start_dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                #x = F.dropout(x, p=0.5, training=self.training)
            embeddings.append(x)

        x = self.fc(x).squeeze()
        return x, embeddings
        
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        embeddings = []
        x_all = x_all.flatten(start_dim=1)
        with torch.cuda.amp.autocast():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    if i < len(self.convs) - 1:
                        x = F.elu(x)
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0).to(device)
                embeddings.append(x_all)
            x = self.fc(x_all).squeeze()
    
        return x, embeddings