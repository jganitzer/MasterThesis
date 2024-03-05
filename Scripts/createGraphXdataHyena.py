import pandas as pd

import torch
from models.hyenaEncoder import HyenaRegressor_MLP_cat_Reduced

from torch.utils.data import DataLoader, TensorDataset


dataRoot = "/data"
heritability = "/h40"

phenotype_path = dataRoot + heritability + heritability + "_simu.dat"
snp_path = dataRoot + heritability + heritability + "_simu.snp"
benchmark_path = dataRoot + heritability + heritability + "_simu.bv"
pedigree_path = dataRoot + heritability + heritability + "_simu.ped"
snpPos_path = dataRoot + heritability + heritability + "_simu_snp.txt"
qtl_path = dataRoot + heritability + heritability + "_simu_qtl.txt"

df_snp_pheno_ped = pd.read_parquet(dataRoot + heritability + '/snp_maf_ped.parquet', engine='pyarrow')

checkpointRoot = '/data/checkpoints/'

modelName = ''

#Hyena Config

#HyenaOperator
order = 2
activation = 'relu' 

#paddedLength = 26283
#max_pos = 35000
max_pos = 27000

lr = 0.001 

batch_size = 128 
num_epochs = 40
val_size = 0.01 

input_dim = 8#3
num_layers = 6
num_heads = 1

gamma = 0.95

patience = 10

hidden1 = 1000
hidden2 = 500
hidden3 = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_features = 26282
model = HyenaRegressor_MLP_cat_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = hidden3, num_features = num_features, order = order, activation = activation).to(device)


checkpoint = torch.load(checkpointRoot + modelName + '/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

y_train = df_snp_pheno_ped["phenotype"]
X_train_df = df_snp_pheno_ped.drop(["id", "phenotype", "sire", "dam", "generation", "sex"], axis=1)
df_ids = df_snp_pheno_ped[["id"]]


X_train = X_train_df.to_numpy()

X_train_tensor = torch.tensor(X_train, dtype=torch.int8).to('cpu')

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

ids_tensor = torch.tensor(df_ids.values, dtype=torch.long).to('cpu')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, ids_tensor)

test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True,num_workers = 2)


dtype1 = torch.cuda.IntTensor
dtype2 = torch.cuda.FloatTensor


inference_df = pd.DataFrame(columns=['targets', 'output', 'id'])

# Generate the column names for embeddings
embedding_columns = [f'pooled_output_{i}' for i in range(num_features)]

all_columns = ['targets', 'output', 'id'] + embedding_columns

inference_df = pd.DataFrame(columns=all_columns)

with torch.no_grad():
    for data, targets, ids in test_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device).type(dtype1)
            targets = targets.to(device).type(dtype2)

            ids = ids.to(device).type(torch.cuda.LongTensor)

            output, pooled_output, embedding = model(data)
            
            # Convert tensors to numpy arrays
            targets_np = targets.cpu().numpy().astype('float32')
            output_np = output.cpu().numpy().astype('float32')
            pooled_output_np = pooled_output.cpu().numpy().astype('float32')
            #embedding_np = embedding.cpu().numpy().astype('float32')
            ids_np = ids.squeeze().cpu().numpy()

            data_dict = {}
            for i in range(pooled_output_np.shape[1]):
                column_name = f"pooled_output_{i}"
                data_dict[column_name] = pooled_output_np[:, i]

            df_pooled_output = pd.DataFrame(data_dict)
            
            # Create a temporary dataframe with targets and output
            temp_df = pd.DataFrame({
                'targets': targets_np,
                'output': output_np,
                'id': ids_np,
            })
            
            # Concatenate the dataframes horizontally
            combined_df = pd.concat([temp_df, df_pooled_output], axis=1)
            
            # Append the temporary dataframe to the main dataframe
            inference_df = pd.concat([inference_df, combined_df], ignore_index=True)
            
inference_df.to_parquet(dataRoot + heritability + modelName + '.parquet')
   
print("Finished")   
