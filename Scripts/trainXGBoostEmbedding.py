import wandb
import pandas as pd
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
from datetime import datetime
import xgboost as xgb
from wandb.xgboost import WandbCallback

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"
wandb_run_name = "_"
entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/_"

embeddingPath = '_.parquet'
embeddingPathGen30 = '_.parquet'

#Config
lr = 0.1 

n_estimators=5000
max_depth=3

val_size = 0.2
batch_size = 32
patience = 10

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "Preprocessing": "maf",
    "embeddingPath": embeddingPath,
    "embeddingPathGen30": embeddingPathGen30,
    "lr": lr,
    "batch_size": batch_size,
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "val_size": val_size,
    "patience": patience,
    "additional_settings": "fineTrain",
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)

df_snp_pheno_train = pd.read_parquet(dataRoot + heritability + embeddingPath, engine='pyarrow')
df_snp_pheno_test = pd.read_parquet(dataRoot + heritability + embeddingPathGen30, engine='pyarrow')

X_train = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
X_test = df_snp_pheno_test.drop(columns=['phenotype', 'id'], axis=1)
y_train = df_snp_pheno_train["phenotype"]
y_test = df_snp_pheno_test["phenotype"]


X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float).to('cpu')
X_val= torch.tensor(X_test.to_numpy(), dtype=torch.float).to('cpu')

y_val = torch.tensor(y_test.values, dtype=torch.float).to('cpu')
y_train = torch.tensor(y_train.values, dtype=torch.float).to('cpu')


model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, tree_method='gpu_hist', random_state=42, callbacks=[WandbCallback(log_model=True)], early_stopping_rounds=patience)

eval_set = [(X_train.numpy(), y_train.numpy()), (X_val.numpy(), y_val.numpy())]

model.fit(X_train.numpy(), y_train.numpy(), eval_set=eval_set, verbose=True, eval_metric="rmse")

print("finished model fit")

output = model.predict(X_val)

print("predicted output")

#loss = criterion(output, y_val)
loss = mean_squared_error(y_val.cpu().numpy(), output)
mae = mean_absolute_error(y_val.cpu().numpy(), output)
r2 = r2_score(y_val.cpu().numpy(), output)
_pearsonr, _pvalue = pearsonr(y_val.cpu().numpy(), output)

print("Loss_mse:", loss)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Pearson correlation coefficient:", _pearsonr)
print("p-value:", _pvalue)

results = model.evals_result()
print(results)

epochs = len(results['validation_0']['rmse'])

wandb.define_metric("epoch")
wandb.define_metric("train_loss", step_metric='epoch')
wandb.define_metric("val_loss", step_metric='epoch')

wandb.log({"val_loss": results['validation_0']['rmse'], "epoch": epochs})

run.summary["MSE"] = loss
run.summary["MAE"] = mae
run.summary["rsquared"] = r2
run.summary["pearsonr"] = _pearsonr
run.summary["p_value"] = _pvalue

wandb.log({"MSE": loss})
wandb.log({"MAE": mae})
wandb.log({"rsquared": r2})
wandb.log({"pearsonr": _pearsonr})
wandb.log({"p_value": _pvalue})

model.save_model(checkpoint_path + "model.json")

run.finish()
