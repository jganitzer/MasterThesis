import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from models.hyenaEncoder import HyenaRegressor_MLP_cat_Reduced
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import (
    LayerIntegratedGradients,
)

#Config
#HyenaOperator
order = 2 
activation = 'relu'
max_pos = 27000

batch_size = 64 

input_dim = 8
num_layers = 6
num_heads = 1

hidden1 = 1000
hidden2 = 500

criterion = nn.MSELoss()

wandbDir = "/data/_"
dataRoot = "/data/_"
eval_dataframes = '/eval_dataframes/hyena/'

heritability = "/h40"
phenotype_path = dataRoot + heritability + heritability + "_simu.dat"
snp_path = dataRoot + heritability + heritability + "_simu.snp"
benchmark_path = dataRoot + heritability + heritability + "_simu.bv"
pedigree_path = dataRoot + heritability + heritability + "_simu.ped"
snpPos_path = dataRoot + heritability + heritability + "_simu_snp.txt"
qtl_path = dataRoot + heritability + heritability + "_simu_qtl.txt"

checkpointRoot = '/data/checkpoints/'

#Hyena
heritability = "/h40"

#gen30
trainPath = '/snp_pheno_BPpos_maf.parquet'
testPath = '/snpTruePheno_gen30_BPpos_maf.parquet'

modelName = '_' 

device = torch.device('cpu')

def add_value(x, value):
    return x + value

class CatTestLoaderFine:
    def __init__(self, dataRoot, trainPath, testPath,wandbDir, batch_size = 16):
        self.dataRoot = dataRoot
        self.trainPath = trainPath
        self.testPath = testPath
        self.batch_size = batch_size
        self.wandbDir = wandbDir

    def getLoaders(self, addValue = 0):
        df_snp_pheno_train = pd.read_parquet(self.dataRoot + self.trainPath, engine='pyarrow')
        df_snp_pheno_test = pd.read_parquet(self.dataRoot + self.testPath, engine='pyarrow')

        X_train_df = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
        y_train = df_snp_pheno_train["phenotype"]
        
        X_val_df = df_snp_pheno_test.drop(columns=['phenotype', 'id'], axis=1)
        y_val = df_snp_pheno_test["phenotype"]
        
        id_val = df_snp_pheno_test["id"]
        id_train = df_snp_pheno_train["id"]
        
        X_val_df = X_val_df.apply(add_value, value=addValue)
        X_train_df = X_train_df.apply(add_value, value=addValue)
      
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int).to('cpu')
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).to('cpu')
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to('cpu')
        
        id_val_tensor = torch.tensor(id_val.values, dtype=torch.int).to('cpu')
        id_train_tensor = torch.tensor(id_train.values, dtype=torch.int).to('cpu')
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, id_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, id_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True, num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return val_loader, train_loader

def loadAndCheckModel():
    checkpoint = torch.load(checkpointRoot + modelName + '/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

def loadEvalData(modelName):
    model_data = pd.read_parquet(dataRoot + heritability + eval_dataframes + modelName + '.parquet', engine='pyarrow')
    output_data = model_data['output'].values.astype(float)
    target_data = model_data['targets'].values.astype(float)
    return model_data[['output', 'targets']],output_data, target_data, model_data

catLoader = CatTestLoaderFine(dataRoot + heritability,trainPath, testPath, wandbDir=wandbDir, batch_size = batch_size)
val_loader, train_loader = catLoader.getLoaders(addValue = 0)
num_features = val_loader.dataset.tensors[0].shape[1]

order = 2
activation = 'relu'
max_pos = 27000
batch_size = 64
input_dim = 8
num_layers = 6
num_heads = 1
hidden1 = 1000
hidden2 = 500

model = HyenaRegressor_MLP_cat_Reduced(input_dim=input_dim, num_layers=num_layers, num_heads=num_heads, max_pos=num_features, hidden1 = hidden1, hidden2 = hidden2, hidden3 = 200, num_features = num_features, order = order, activation = activation).to(device)

loadAndCheckModel()

inference_df, output_np, targets_np, model_data = loadEvalData(modelName)

X_train_tensor = train_loader.dataset.tensors[0]
X_val_tensor = val_loader.dataset.tensors[0]
y_val_tensor = val_loader.dataset.tensors[1]
id_val_tensor = val_loader.dataset.tensors[2]

# Define the number of desired samples
num_samples = 25

# Define the number of bins for the 'output' column
num_bins = 10

# Create bins and assign each row to a bin
model_data['output_bins'] = pd.cut(model_data['output'], bins=num_bins, labels=False)

# Stratified sampling
sampled_data = model_data.groupby('output_bins', group_keys=False).apply(lambda x: x.sample(int(num_samples/num_bins), random_state=42))

sampled_data.reset_index(drop=True, inplace=True)
sampled_data.drop(columns=['output_bins'], inplace=True)
selected_ids = sampled_data['id'].tolist()

selected_ids_set = set(selected_ids)

selected_values = []
selected_ids = []
selected_targets = []

for idx, id_val in enumerate(id_val_tensor):
    if id_val.item() in selected_ids_set:
        selected_values.append(X_val_tensor[idx])
        selected_ids.append(id_val_tensor[idx])
        selected_targets.append(y_val_tensor[idx])

selected_values_tensor = torch.stack(selected_values)
selected_ids_tensor = torch.stack(selected_ids)
selected_targets_tensor = torch.stack(selected_targets)

class WrapperModel(nn.Module):
    def __init__(self, pretrained_model):
        super(WrapperModel, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, x):
        output1, _, _ = self.pretrained_model(x)

        return output1
    
wrapper_model = WrapperModel(model)

baseTensor = torch.randint(0, 3, size=selected_values_tensor.shape)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
    
lig = LayerIntegratedGradients(wrapper_model, wrapper_model.pretrained_model.encoder)
attributions, delta = lig.attribute(inputs=selected_values_tensor.to(device),baselines = baseTensor.to(device), return_convergence_delta=True, n_steps = 100)

delta_np = delta.cpu().numpy()

attributions_sum = summarize_attributions(attributions)

attributions_mean_all = attributions_sum.cpu().mean(0).detach().numpy()
attributions_all_norm = attributions_mean_all / np.linalg.norm(attributions_mean_all, ord=1)

np.save(dataRoot + heritability + eval_dataframes + 'LayerIntegratedGradients_attributions_all_norm_v04_' + modelName + '.npy', attributions_all_norm)
np.save(dataRoot + heritability + eval_dataframes + 'LayerIntegratedGradients_delta_v04_' + modelName + '.npy', delta_np)

print("Delta:", delta.mean())