import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig

class BigBirdRegressor(nn.Module):
    def __init__(self, input_dim=3, num_layers=2, num_heads=3, max_pos=36000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=input_dim, 
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        self.encoder = BigBirdModel(config)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds):
        out = self.encoder(inputs_embeds=inputs_embeds) 

        output = out['last_hidden_state']
        
        output = torch.mean(output, dim=1)

        output = self.dropout(output)
        logits = self.fc(output).squeeze()

        return logits
    
class BigBirdRegressorHF(nn.Module):
    def __init__(self, input_dim=3, num_layers=4, num_heads=3, max_pos=36000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        self.encoder = BigBirdModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds):
        out = self.encoder(inputs_embeds=inputs_embeds) 

        output = out['last_hidden_state']
        
        # Average pooling
        output = torch.mean(output, dim=1)
        output = self.dropout(output)
        output = self.dense(output)
        output = torch.tanh(output)
        output = self.dropout(output)      

        logits = self.out_proj(output).squeeze() 

        return logits

    
class BigBirdRegressor_cat(nn.Module):
    def __init__(self, input_dim=3, num_layers=2, num_heads=3, max_pos=36000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=input_dim,
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        self.encoder = BigBirdModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        out = self.encoder(input_ids)

        output = out['last_hidden_state']
        
        output = torch.mean(output, dim=1)

        output = self.dropout(output)
        logits = self.fc(output).squeeze()

        return logits
    
class BigBirdRegressor_MLP(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500, input_dim=3, num_layers=2, num_heads=3, max_pos=36000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=input_dim, 
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        self.encoder = BigBirdModel(config)
            
        self.fc = nn.Sequential(
            nn.Linear(input_dim * num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds):
        out = self.encoder(inputs_embeds=inputs_embeds)  

        output = out['last_hidden_state']
        
        output = output.flatten(start_dim=1)
        
        output = self.dropout(output)
        logits = self.fc(output).squeeze()

        return logits
    
class BigBirdRegressor_MLP_cat(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500, input_dim=256, num_layers=4, num_heads=4, max_pos=36000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        self.encoder = BigBirdModel(config)
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        out = self.encoder(input_ids)

        output = out['last_hidden_state']

        output = torch.mean(output, dim=2)
        
        output = self.dropout(output)
        logits = self.fc(output).squeeze() 

        return logits
        
class BigBirdRegressor_MLP_cat_Reduced(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        
        self.encoder = BigBirdModel(config)
        
        self.linear_layer = nn.Linear(input_dim, 1)
        
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
        self.linear_layer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, input_ids):
        out = self.encoder(input_ids)

        embedding = out['last_hidden_state']

        reduced_tensor = self.linear_layer(embedding).squeeze()

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, embedding
        
class BigBirdRegressor_MLP_Reduced(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=10,  
            hidden_size=input_dim,
            num_attention_heads=num_heads,
            intermediate_size=4*input_dim,
            max_position_embeddings=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True
        )
        
        self.encoder = BigBirdModel(config)
        
        self.linear_layer = nn.Linear(input_dim, 1)
            
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
        self.linear_layer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs_embeds):
        out = self.encoder(inputs_embeds=inputs_embeds)  

        embedding = out['last_hidden_state']

        reduced_tensor = self.linear_layer(embedding).squeeze()

        logits = self.fc(reduced_tensor).squeeze() 
        return logits, reduced_tensor, embedding