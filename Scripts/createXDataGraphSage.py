import pandas as pd
import torch
from models.graphsage import SAGE_OneHot_MLP
from torch_geometric.loader import NeighborLoader
import copy


dataRoot = "/data"
heritability = "/h40"

data = torch.load(dataRoot + heritability + "/graph_maf_float32.pt")

checkpointRoot = '/data/checkpoints/'

modelName = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Config
hidden1= 128
hidden2= 64
embed_dim = 256
model = SAGE_OneHot_MLP(data.x.size(1) * data.x.size(2), embed_dim, 1,hidden1 = hidden1, hidden2 = hidden2).to(device)


checkpoint = torch.load(checkpointRoot + modelName + '/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

batch_size = 64
kwargs = {'batch_size': batch_size, 'num_workers': 0, 'persistent_workers': False}
subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

y_hat, embeddings_all = model.inference(data.x, subgraph_loader, device)
targets = data.y.to(y_hat.device)

embeddings = embeddings_all[-1]

dtype = torch.cuda.FloatTensor
inference_df = pd.DataFrame(columns=['targets', 'output', 'id'])

embedding_columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]

print(embeddings.shape[1])

all_columns = ['targets', 'output', 'id'] + embedding_columns

inference_df = pd.DataFrame(columns=all_columns)

targets_np = targets.cpu().numpy().astype('float32')
output_np = y_hat.cpu().numpy().astype('float32')
embeddings_np = embeddings.cpu().numpy().astype('float32')

ids = data.node_ids
ids_np = ids.squeeze().cpu().numpy()

data_dict = {}
for i in range(embeddings_np.shape[1]):
    column_name = f"embedding_{i}"
    data_dict[column_name] = embeddings_np[:, i]

df_pooled_output = pd.DataFrame(data_dict)

temp_df = pd.DataFrame({
    'targets': targets_np,
    'output': output_np,
    'id': ids_np,
})

combined_df = pd.concat([temp_df, df_pooled_output], axis=1)

inference_df = pd.concat([inference_df, combined_df], ignore_index=True)
         
inference_df.to_parquet(dataRoot + heritability + modelName +'.parquet')
   
print("Finished")   
 