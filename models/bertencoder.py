import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch import Tensor
import math

class BertRegressor_MLP(nn.Module):
    def __init__(self, num_features, hidden1 = 128, hidden2 = 64, input_dim=3, num_heads=3, max_pos=256):
        super().__init__()
        config = BertConfig(
            vocab_size=input_dim,
            num_attention_heads=num_heads,
            max_positions=max_pos,
            output_attentions=False,
            return_dict=True,
            add_token_type_embeddings = False,
            hidden_act = "relu",              
        )
        self.encoder = BertModel(config)
        

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
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, inputs_embeds):

        out = self.encoder(inputs_embeds=inputs_embeds)  
        embedding = out['last_hidden_state']
        
        print("Bert_out_embedding")
        print(embedding)

        pooled_output, _ = torch.max(embedding, dim=-1)
        
        logits = self.fc(pooled_output).squeeze()  

        return logits

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BertRegressor_MLP2(nn.Module):
    def __init__(self, num_features, hidden1 = 128, hidden2 = 64,  hidden3 = 32, input_dim=3, num_layers=2, num_heads=3, max_pos=256):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, 0.1, max_pos)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, 4*input_dim)
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
            #m.bias.data.fill_(0.01)
            m.bias.data.zero_()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output, _ = torch.max(output, dim=-1)
        logits = self.fc(output).squeeze()

        return logits