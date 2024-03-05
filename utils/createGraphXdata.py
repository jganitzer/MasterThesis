import pandas as pd
import numpy as np

import torch
from models.megaencoder import MegaChunkRegressor_MLP
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

dataRoot = "/data/_"
heritability = "/h40"

phenotype_path = dataRoot + heritability + heritability + "_simu.dat"
snp_path = dataRoot + heritability + heritability + "_simu.snp"
benchmark_path = dataRoot + heritability + heritability + "_simu.bv"
pedigree_path = dataRoot + heritability + heritability + "_simu.ped"
snpPos_path = dataRoot + heritability + heritability + "_simu_snp.txt"
qtl_path = dataRoot + heritability + heritability + "_simu_qtl.txt"

df_snp_pheno_ped = pd.read_parquet(dataRoot + heritability + '/snp_maf_one_hot.parquet', engine='pyarrow')

checkpointRoot = '_'
modelName = '_'

#Config
paddedLength = 26624
max_pos = 27000
chunk_size = 512

lr = 0.001 
batch_size = 64 

input_dim = 3
num_layers = 2
num_heads = 1

gamma = 0.85

hidden1 = 1000 
hidden2 = 500  
hidden3 = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MegaChunkRegressor_MLP(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=max_pos, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = paddedLength, chunk_size=chunk_size).to(device)

checkpoint = torch.load(checkpointRoot + modelName + '/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

y_train = df_snp_pheno_ped["phenotype"]
X_train_df = df_snp_pheno_ped.drop(["id", "phenotype"], axis=1)
df_ids = df_snp_pheno_ped[["id"]]

X_train = X_train_df.to_numpy()

# Reshape the data back to its original shape
X_train = X_train.reshape(X_train_df.shape[0], int(X_train_df.shape[1]/3), 3)

# Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
attention_mask_train = torch.ones(X_train.shape[0], X_train.shape[1]).to('cpu')

if(paddedLength != -1 and paddedLength > X_train.shape[1]):
    pad = paddedLength - X_train.shape[1]
    # Pad the data to the max length
    X_train = np.pad(X_train, ((0,0),(0,pad),(0,0)), 'constant', constant_values=0)
    attention_mask_train = F.pad(attention_mask_train, (0,pad), 'constant', 0)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cpu')

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

ids_tensor = torch.tensor(df_ids.values, dtype=torch.long).to('cpu')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train, ids_tensor)

test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True,num_workers = 2)


dtype = torch.cuda.FloatTensor
inference_df = pd.DataFrame(columns=['targets', 'output', 'id'])

embedding_columns = [f'pooled_output_{i}' for i in range(paddedLength)]

all_columns = ['targets', 'output', 'id'] + embedding_columns
inference_df = pd.DataFrame(columns=all_columns)

with torch.no_grad():
    for data, targets, attention_mask_val, ids in test_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device).type(dtype)
            targets = targets.to(device).type(dtype)
            attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)
            ids = ids.to(device).type(torch.cuda.LongTensor)

            output, pooled_output, embedding = model(data, attention_mask_val)
            targets_np = targets.cpu().numpy()
            output_np = output.cpu().numpy()
            pooled_output_np = pooled_output.cpu().numpy()
            embedding_np = embedding.cpu().numpy()
            ids_np = ids.squeeze().cpu().numpy()

            data_dict = {}
            for i in range(pooled_output_np.shape[1]):
                column_name = f"pooled_output_{i}"
                data_dict[column_name] = pooled_output_np[:, i]

            df_pooled_output = pd.DataFrame(data_dict)
  
            temp_df = pd.DataFrame({
                'targets': targets_np,
                'output': output_np,
                'id': ids_np,
            })
            
            combined_df = pd.concat([temp_df, df_pooled_output], axis=1)
            
            # Append the temporary dataframe to the main dataframe
            inference_df = pd.concat([inference_df, combined_df], ignore_index=True)
            
inference_df.to_parquet(dataRoot + heritability + '/df_mlp_mega_pooled_output_v01.parquet')
   

print("Finished 1")   
            
dtype = torch.cuda.FloatTensor
inference_df = pd.DataFrame(columns=['targets', 'output', 'id'])

# Generate the column names for embeddings
embedding_columns = [f'embeddings_{i}' for i in range(paddedLength * input_dim)]

# All column names including the standard columns and embeddings
all_columns = ['targets', 'output', 'id'] + embedding_columns

# Create an empty dataframe with the specified columns
inference_df = pd.DataFrame(columns=all_columns)

with torch.no_grad():
    for data, targets, attention_mask_val, ids in test_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device).type(dtype)
            targets = targets.to(device).type(dtype)
            attention_mask_val = attention_mask_val.to(device).type(torch.cuda.IntTensor)
            ids = ids.to(device).type(torch.cuda.LongTensor)

            output, pooled_output, embedding = model(data, attention_mask_val)
            
            targets_np = targets.cpu().numpy()
            output_np = output.cpu().numpy()
            pooled_output_np = pooled_output.cpu().numpy()
            
            embedding = embedding.flatten(start_dim=1)
            embedding_np = embedding.cpu().numpy()
            ids_np = ids.squeeze().cpu().numpy()

            data_dict = {}
            for i in range(embedding_np.shape[1]):
                column_name = f"embeddings_{i}"
                data_dict[column_name] = embedding_np[:, i]

            df_embedding_np = pd.DataFrame(data_dict)
            
            # Create a temporary dataframe with targets and output
            temp_df = pd.DataFrame({
                'targets': targets_np,
                'output': output_np,
                'id': ids_np,
            })
            
            combined_df = pd.concat([temp_df, df_embedding_np], axis=1)
            inference_df = pd.concat([inference_df, combined_df], ignore_index=True)

inference_df.to_parquet(dataRoot + heritability + '/_.parquet')        
                        
print("Finished")   
            