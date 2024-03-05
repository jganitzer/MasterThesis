import wandb
import pandas as pd
from utils.torchDataloader import EmbeddingLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.linear_model import LinearRegression
import pickle

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"

wandb_run_name = "_"

entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

X_path = "/train/snp_X_train_072_pca_ink2.parquet"
y_path = "/train/snp_y_train_072_pca_ink2.parquet"

#Config
val_size = 0.2
batch_size = 32
patience = 10
#percentage_to_keep = 0.95

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "Preprocessing": "maf",
    "batch_size": batch_size,
    "val_size": val_size
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)
embeddingLoader = EmbeddingLoader(dataRoot + heritability, X_path, y_path, wandbDir=wandbDir, batch_size = batch_size, val_size = val_size)

train_loader, val_loader = embeddingLoader.getLoaders()

(X_train, y_train) = train_loader.dataset.tensors
(X_val, targets) = val_loader.dataset.tensors

print(X_train.shape)

model = LinearRegression()

print("got Model")

model.fit(X_train.numpy(), y_train.numpy())

print("finished model fit")

regScore = model.score(X_val.numpy(), targets)

output = model.predict(X_val)

print("predicted output")

loss = mean_squared_error(targets.cpu().numpy(), output)
mae = mean_absolute_error(targets.cpu().numpy(), output)
r2 = r2_score(targets.cpu().numpy(), output)
_pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output)

# Print the results
print("regScore:", regScore)
print("Loss_mse:", loss)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Pearson correlation coefficient:", _pearsonr)
print("p-value:", _pvalue)

wandb.log({"regScore": regScore})
wandb.log({"MSE": loss})
wandb.log({"MAE": mae})
wandb.log({"rsquared": r2})
wandb.log({"pearsonr": _pearsonr})
wandb.log({"p_value": _pvalue})

run.summary["regScore"] = regScore
run.summary["MSE"] = loss
run.summary["MAE"] = mae
run.summary["rsquared"] = r2
run.summary["pearsonr"] = _pearsonr
run.summary["p_value"] = _pvalue

# All regression plots
wandb.sklearn.plot_regressor(model, X_train.numpy(), X_val.numpy(),  y_train.numpy(), targets.numpy(),  model_name=wandb_run_name)
wandb.sklearn.plot_feature_importances(model)

pkl_filename = checkpoint_path + "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

run.finish()
