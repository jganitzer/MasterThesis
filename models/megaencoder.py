import torch
import torch.nn as nn
from transformers import MegaModel, MegaConfig, AutoConfig, MegaForMaskedLM
from torch.nn import MSELoss
from typing import Optional

class MegaRegressor_MLP(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500, input_dim=3, num_layers=2, num_heads=3, max_pos=36000):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
            
        self.fc = nn.Sequential(
            nn.Linear(input_dim * num_features, hidden1),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden2, 1)
        )

    def forward(self, inputs_embeds):
        out = self.encoder(inputs_embeds=inputs_embeds) 

        output = out['last_hidden_state']
        
        # Flatten the input
        output = output.flatten(start_dim=1)
        
        #output = self.dropout(output)
        logits = self.fc(output).squeeze()  
        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits#, attn_weights

class MegaRegressor_MLP_cat(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500, input_dim=256, num_layers=4, num_heads=1, max_pos=36000, hidden_dim = 8):
        super().__init__()
        config = MegaConfig(
            vocab_size = 10, 
            hidden_size=hidden_dim,
            intermediate_size=4*hidden_dim, 
            max_positions=36000,
            num_hidden_layers=num_layers, #128
            output_attentions=False,
            return_dict=True,
            relative_positional_bias = "rotary"
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask)

        output = out['last_hidden_state']
        
        # Flatten the input
        #output = output.flatten(start_dim=1)
        
        output = torch.mean(output, dim=2)
        
        output = self.dropout(output)
        logits = self.fc(output).squeeze()  # (batch_size, 1)

        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits#, attn_weights

class MegaChunkRegressor_MLP(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple"
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
            
        self.fc = nn.Sequential(
            #nn.Linear(input_dim * num_features, hidden1), #for Flatten
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask)

        embedding = out['last_hidden_state']
        
        #pooled_output = embedding.flatten(start_dim=1)
        #pooled_output = torch.mean(embedding, dim=-1)
        #pooled_output, _ = torch.max(embedding, dim=-1)
        #pooled_output = torch.cumsum(embedding, dim=-1)[:, :, -1:]
        pooled_output = torch.sum(embedding, dim=-1)
        
        #output = self.dropout(output)
        logits = self.fc(pooled_output).squeeze()  # (batch_size, 1)

        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits, pooled_output, embedding #, attn_weights
        
        
class MegaChunkRegressor_MLP2(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        
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


    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask)  

        embedding = out['last_hidden_state']

        reduced_tensor = self.linear_layer(embedding).squeeze()

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, embedding

class MegaChunkRegressor_MLP3(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_features)])
            
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask) 

        embedding = out['last_hidden_state']

        stacked_tensor = torch.stack([self.linear_layers[i](embedding[:, i]) for i in range(embedding.size(1))], dim=2)
        #reduced_tensor = torch.cat([self.linear_layers[i](embedding[:, i * embedding.size(-1): (i + 1) * embedding.size(-1)]) for i in range(embedding.size(1))], dim=1)

        reduced_tensor = stacked_tensor.squeeze()

        logits = self.fc(reduced_tensor).squeeze()  # (batch_size, 1)

        return logits, reduced_tensor, embedding

class MegaChunkRegressor_MLP4(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=True,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
            
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask)  

        embedding = out['last_hidden_state']

        #pooled_output = torch.mean(embedding, dim=2)
        #pooled_output, _ = torch.max(embedding, dim=-1)
        #pooled_output = torch.cumsum(embedding, dim=-1)[:, :, -1:]
        
        # Attention weights
        attn_weights = out['attentions']
        print(attn_weights[-1].shape)
        print(embedding.shape)
        
        weighted_representation = torch.matmul(attn_weights[-1], embedding)
        
        print(weighted_representation.shape)
        
        #output = self.dropout(output)
        logits = self.fc(weighted_representation).squeeze()  # (batch_size, 1)


        return logits, attn_weights, embedding #, attn_weights

    
class MegaChunkRegressor_MLP_cat(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=8, num_layers=2, num_heads=1, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "rotary",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        
        self.fc = nn.Sequential(
            #nn.Linear(input_dim * num_features, hidden1), #for flatten
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask)

        embedding = out['last_hidden_state']
        
        # Flatten the input
        #pooled_output = output.flatten(start_dim=1)
        
        #output = torch.mean(output, dim=2)
        pooled_output, _ = torch.max(embedding, dim=-1)
        #pooled_output = torch.sum(embedding, dim=-1)
        #output = output.mean(dim=1)
        #output = self.dropout(output)
        logits = self.fc(pooled_output).squeeze()  # (batch_size, 1)

        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits, pooled_output, embedding
        
class MegaChunkRegressor_MLP_cat_Reduced(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10,
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "rotary", 
            add_token_type_embeddings = False           
        )
        
        self.encoder = MegaModel(config)
        
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

        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask)

        embedding = out['last_hidden_state']

        reduced_tensor = self.linear_layer(embedding).squeeze()

        logits = self.fc(reduced_tensor).squeeze()  

        return logits, reduced_tensor, embedding

class MegaChunkRegressor_MLP_cat_ReducedModules(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "rotary", 
            add_token_type_embeddings = False           
        )
        
        self.encoder = MegaModel(config)
        
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
        
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_features)])

        self.fc.apply(self.init_weights)
        self.linear_layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask)

        embedding = out['last_hidden_state']

        # Apply separate linear layers to each token
        reduced_tensors = [linear_layer(token).squeeze() for linear_layer, token in zip(self.linear_layers, embedding.unbind(dim=1))]

        reduced_tensor = torch.stack(reduced_tensors, dim=1)

        logits = self.fc(reduced_tensor).squeeze()  
        return logits, reduced_tensor, embedding

class MegaChunkRegressor_MLP_cat3(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, #8  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple", 
            add_token_type_embeddings = False           
        )
        
        self.encoder = MegaModel(config)
        
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_features)])
        
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
        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask)

        embedding = out['last_hidden_state']

        stacked_tensor = torch.stack([self.linear_layers[i](embedding[:, i]) for i in range(embedding.size(1))], dim=2)
        #reduced_tensor = torch.cat([linear_layers[i](embedding[:, i * embedding.size(-1): (i + 1) * embedding.size(-1)]) for i in range(embedding.size(1))], dim=1)

        reduced_tensor = stacked_tensor.squeeze()

        logits = self.fc(reduced_tensor).squeeze()  # (batch_size, 1)

        return logits, reduced_tensor, embedding


class MegaChunkRegressor_MLP_TopK(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024, topK = 500):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        self.topK = topK
            
        self.fc = nn.Sequential(
            nn.Linear(topK, hidden1),
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask) 

        embedding = out['last_hidden_state']
        
        # Flatten the input
        #output = embedding.flatten(start_dim=1)
        #pooled_output = torch.mean(embedding, dim=2)
        pooled_output, _ = torch.max(embedding, dim=-1)
        
        topKOut = torch.topk(pooled_output, self.topK, sorted = False).values
        
        #output = self.dropout(output)
        logits = self.fc(topKOut).squeeze()  # (batch_size, 1)

        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits, pooled_output, embedding #, attn_weights
     
     
class MegaChunkRegressor_Sage(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500,  hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple"
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)

    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask)  

        embedding = out['last_hidden_state']
        
        # Flatten the input
        #output = embedding.flatten(start_dim=1)
        #pooled_output = torch.mean(embedding, dim=2)
        pooled_output, _ = torch.max(embedding, dim=-1)
        
        return pooled_output


class MegaChunkRegressor_FF(nn.Module):
    def __init__(self, num_features, input_dim=3, num_layers=2, num_heads=1, max_pos=35000, chunk_size=1024):
        super().__init__()
        config = MegaConfig(
            vocab_size=input_dim,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers,
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple"
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, inputs_embeds, attention_mask):
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask = attention_mask)  
        embedding = out['last_hidden_state']
        
        pooled_output = embedding.mean(dim=1)

        logits = self.fc(pooled_output).squeeze()  

        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits, pooled_output, embedding #, attn_weights


class MegaChunkRegressor_MLP_cat_CLS(nn.Module):
    def __init__(self, num_features, hidden1 = 32, hidden2 = 16,  hidden3 = 8, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size = 10,  
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "rotary"
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config, add_pooling_layer=True)
        
        self.fc = nn.Linear(input_dim, 1)
        
        self.fc.apply(self.init_weights)
                
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        embedding = self.encoder(input_ids, attention_mask = attention_mask)
                   
        pooled_output = embedding['pooler_output']

        logits = self.fc(pooled_output).squeeze() 

        return logits, pooled_output, embedding


class MegaChunkRegressor_MLP_catTrainer(nn.Module):
    def __init__(self, num_features, hidden1 = 1000, hidden2 = 500, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple", 
            add_token_type_embeddings = False           
        )
        self.encoder = MegaModel(config)
        
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(
        self, 
        input_ids, 
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ):
        
        out = self.encoder(input_ids, attention_mask = attention_mask)

        output = out['last_hidden_state']

        output, _ = torch.max(output, dim=-1)

        logits = self.fc(output).squeeze() 
        
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        classifier_output = (logits,) + out[2:]

        return ((loss,) + classifier_output)

class MegaChunkRegressor_MLP_cat_PreTrained(nn.Module):
    def __init__(self, num_features,preTrainedPath, hidden1 = 1000, hidden2 = 500, hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaForMaskedLM.from_pretrained(preTrainedPath)
                    
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask, output_hidden_states = True)
        
        embedding = out['hidden_states'][-1]
        
        # Flatten the input
        #output = output.flatten(start_dim=1)
        
        #pooled_output = torch.mean(embedding, dim=2)
        pooled_output, _ = torch.max(embedding, dim=-1)
        
        #output = output.mean(dim=1)

        logits = self.fc(pooled_output).squeeze()  
        
        # Attention weights
        #attn_weights = out['attentions']  # [-1]  # (batch_size, num_heads, seq_length, seq_length)

        return logits, pooled_output, embedding

class MegaChunkRegressor_MLP_cat_PreTrained2(nn.Module):
    def __init__(self, num_features,preTrainedPath, hidden1 = 1000, hidden2 = 500, hidden3 = 200, input_dim=3, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "rotary",
            add_token_type_embeddings = False           
        )
        self.encoder = MegaForMaskedLM.from_pretrained(preTrainedPath)
                    
        for param in self.encoder.parameters():
            param.requires_grad = False
        
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
 
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask, output_hidden_states = True)
        
        embedding = out['hidden_states'][-1]

        reduced_tensor = self.linear_layer(embedding).squeeze()

        logits = self.fc(reduced_tensor).squeeze() 

        return logits, reduced_tensor, embedding

class MegaChunkRegressor_MLP_cat_PreTrained_CLS(nn.Module):
    def __init__(self, num_features,preTrainedPath, hidden1 = 1000, hidden2 = 500, hidden3 = 200, input_dim=8, num_layers=2, num_heads=3, max_pos=37000, chunk_size=512):
        super().__init__()
        config = MegaConfig(
            vocab_size=10, 
            hidden_size=input_dim,
            intermediate_size=4*input_dim, 
            max_positions=max_pos,
            num_hidden_layers=num_layers, 
            output_attentions=False,
            return_dict=True,
            use_chunking=True,
            chunk_size = chunk_size,
            relative_positional_bias = "simple", 
            add_token_type_embeddings = False           
        )
        self.encoder = MegaForMaskedLM.from_pretrained(preTrainedPath)
                    
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask = attention_mask, output_hidden_states = True)
        
        embedding = out['hidden_states'][-1]
                
        cls_token = embedding[:, 0, :]
        
        logits = self.fc(cls_token).squeeze()

        return logits, cls_token, embedding
