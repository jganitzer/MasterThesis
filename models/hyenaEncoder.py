import torch
import torch.nn as nn
from torch import Tensor
import math

import sys
sys.path.append('models/ssm/')

from models.ssm.hyena import HyenaOperator
from models.hyenaEncoderMLM import HyenaForMaskedLM

from transformers import MegaModel, MegaConfig #Pretrainind hack

# Define model architecture
class HyenaRegressor_MLP(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
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

    def forward(self, src: Tensor) -> Tensor:
        output = self.transformer_encoder(src)
        output, _ = torch.max(output, dim=-1)
        logits = self.fc(output).squeeze()

        return logits

class HyenaRegressor_MLP_Reduced(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.linear_layer = nn.Linear(input_dim, 1)

        self.fc.apply(self.init_weights)
        self.linear_layer.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        output = self.transformer_encoder(src)

        reduced_tensor = self.linear_layer(output).squeeze()

        logits = self.fc(reduced_tensor).squeeze() 

        return logits, reduced_tensor, output

class HyenaRegressor_MLP_cat_ReducedModules(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
       
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_features)])

        self.fc.apply(self.init_weights)
        self.linear_layers.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)

        # Apply separate linear layers to each token
        reduced_tensors = [linear_layer(token).squeeze() for linear_layer, token in zip(self.linear_layers, output.unbind(dim=1))]

        reduced_tensor = torch.stack(reduced_tensors, dim=1)

        logits = self.fc(reduced_tensor).squeeze()  # (batch_size, 1)

        return logits, reduced_tensor, output

class HyenaRegressor_MLP_cat_ReducedRelu(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
                
        self.linear_layer = nn.Linear(input_dim, 1)
        self.activation = nn.ReLU()

        self.fc.apply(self.init_weights)
        self.linear_layer.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)

        reduced_tensor = self.linear_layer(output).squeeze()
        reduced_tensor = self.activation(reduced_tensor)

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, output  

class HyenaRegressor_MLP_cat(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
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

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        
        #pooled_output = output.flatten(start_dim=1)
        pooled_output, _ = torch.max(output, dim=-1)
        #pooled_output = output.mean(dim=-1)
        #pooled_output = torch.sum(output, dim=-1)
        logits = self.fc(pooled_output).squeeze()

        return logits, pooled_output, output

class HyenaRegressor_MLP_cat_Reduced(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
       
        self.linear_layer = nn.Linear(input_dim, 1)

        self.fc.apply(self.init_weights)
        self.linear_layer.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)

        reduced_tensor = self.linear_layer(output).squeeze()

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, output


class HyenaRegressor_MLP_cat_Pretrained(nn.Module):
    def __init__(self, num_features, preTrainedPath, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
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
        
        pretrainedModel = HyenaForMaskedLM(config, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, order = order, activation = activation)
        pretrainedModel.load_state_dict(torch.load(preTrainedPath))
        
        self.transformer_encoder = pretrainedModel.mega
        
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
            
        self.fc = nn.Sequential(
            nn.Linear(num_features * input_dim, hidden1),
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

    def forward(self, src: Tensor) -> Tensor:

        output = self.transformer_encoder(src)
        
        pooled_output = output.flatten(start_dim=1)
        #pooled_output, _ = torch.max(output, dim=-1)
        #pooled_output = output.mean(dim=-1)
        #pooled_output = torch.sum(output, dim=-1)
        logits = self.fc(pooled_output).squeeze()

        return logits, pooled_output, output

class HyenaRegressor_MLP_cat_Reduced_Pretrained(nn.Module):
    def __init__(self, num_features, preTrainedPath, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=8, num_layers=2, num_heads=1, max_pos=35000, order = 2, activation = 'identity'):
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
        
        pretrainedModel = HyenaForMaskedLM(config, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, order = order, activation = activation)
        pretrainedModel.load_state_dict(torch.load(preTrainedPath))
        
        self.transformer_encoder = pretrainedModel.mega
   
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
       
        self.linear_layer = nn.Linear(input_dim, 1)

        self.fc.apply(self.init_weights)
        self.linear_layer.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, src: Tensor) -> Tensor:

        output = self.transformer_encoder(src)

        reduced_tensor = self.linear_layer(output).squeeze()

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, output
        
        input_dim=input_dim, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, preTrainedPath = preTrainedPath, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, order = order, activation = activation
class HyenaRegressor_MLP_cat_Reduced_Pretrained_Transfer(nn.Module):
    def __init__(self, num_features, preTrainedPath, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=8, num_layers=2, num_heads=1, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
                
        preTrainedPath2 = "/data/checkpoints/_/" + "best_model_state_dict.pth" 
        pretrainedModel = HyenaRegressor_MLP_cat_Reduced_Pretrained(num_features = num_features,preTrainedPath = preTrainedPath2, hidden1 = hidden1, hidden2 = hidden2,  hidden3 = hidden3, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, order = order, activation = activation)

        checkpoint = torch.load(preTrainedPath)
        pretrainedModel.load_state_dict(checkpoint['model_state_dict'])
        
        self.transformer_encoder = pretrainedModel

    def forward(self, src: Tensor) -> Tensor:

        logits, reduced_tensor, output = self.transformer_encoder(src)

        return logits, reduced_tensor, output        

class HyenaRegressor_MLP_cat_CLS_Pretrained(nn.Module):
    def __init__(self, num_features, preTrainedPath, hidden1 = 8, hidden2 = 4,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=1, max_pos=35000, order = 2, activation = 'identity'):
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
        
        pretrainedModel = HyenaForMaskedLM(config, input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, order = order, activation = activation)
        pretrainedModel.load_state_dict(torch.load(preTrainedPath))
        
        self.transformer_encoder = pretrainedModel.mega
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
                
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:

        output = self.transformer_encoder(src)
        
        cls_token = output[:, 0, :]
        
        logits = self.fc(cls_token).squeeze()

        return logits, cls_token, output

class HyenaRegressor_MLP_cat_CLS(nn.Module):
    def __init__(self, num_features, hidden1 = 8, hidden2 = 4,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        self.fc.apply(self.init_weights)
                
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        
        cls_token = output[:, 0, :]
        
        logits = self.fc(cls_token).squeeze()

        return logits, cls_token, output

class HyenaRegressor_MLP_cat_flat(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(num_features * input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        output_flat = output.flatten(start_dim=1)
        logits = self.fc(output_flat).squeeze()

        return logits, output_flat, output

class HyenaRegressor_MLP_cat_pool(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, order = 2, activation = 'identity'):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
        encoder_layers.self_attn = HyenaOperator(input_dim, l_max=max_pos, order = order, num_heads = num_heads, activation = activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(10, input_dim)
        self.d_model = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def init_weights(self) -> None:
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        pooled_output = output.mean(dim=1)

        logits = self.fc(pooled_output).squeeze()

        return logits, pooled_output, output
