import wandb
import pandas as pd
import pandas as pd
from utils.torchDataloader import CatLoaderFine
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import gc
from datetime import datetime
import xgboost as xgb
from wandb.xgboost import WandbCallback
from sklearn.metrics import mean_squared_error

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_project_name = "_"
wandb_run_name = "_"
entity="_"

heritability = "/h40"
dataRoot = "/data/_"
wandbDir = "/data/-"

trainPath = '/snp_pheno_BPpos_maf_one_hot.parquet'
testPath = '/snpTruePheno_gen30_BPpos_maf_one_hot.parquet'

#Config
lr = 0.1
reg_alpha = 0.001 
reg_lambda = 0.01

n_estimators=5000
max_depth=3

val_size = 0.2
batch_size = 32
patience = 500
percentage_to_keep = 0.94

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "Preprocessing": "maf",
    "trainPath" : trainPath,
    "testPath" : testPath,
    "lr": lr,
    "batch_size": batch_size,
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "val_size": val_size,
    "patience": patience,
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)

oneHotLoader = CatLoaderFine(dataRoot + heritability, trainPath, testPath, wandbDir=wandbDir, batch_size = batch_size)

train_loader, val_loader = oneHotLoader.getLoaders()

del oneHotLoader
gc.collect()

(X_train, y_train) = train_loader.dataset.tensors

del train_loader
gc.collect()

(X_val, targets) = val_loader.dataset.tensors

del val_loader
gc.collect()

model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, tree_method='gpu_hist', enable_categorical=True, random_state=42, early_stopping_rounds=patience, callbacks=[WandbCallback(log_model=True)], eval_metric = mean_squared_error) # eval_metric="rmse", 


print("got Model")

model.fit(X_train.numpy(),  y_train.numpy(), eval_set=[(X_val.numpy(), targets.numpy())], verbose=True)

print("finished model fit")

output = model.predict(X_val)

print("predicted output")

loss = mean_squared_error(targets.cpu().numpy(), output)
mae = mean_absolute_error(targets.cpu().numpy(), output)
r2 = r2_score(targets.cpu().numpy(), output)
_pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output)

# Print the results
print("Loss_mse:", loss)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Pearson correlation coefficient:", _pearsonr)
print("p-value:", _pvalue)

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

temp_df = pd.DataFrame({
    'targets': targets.cpu().numpy(),
    'output': output
})

eval_dataframes = '/eval_dataframes/xgboost/'
heritability = "/h40"
dataRoot = "/data"

temp_df.to_parquet(dataRoot + heritability + eval_dataframes + wandb_run_name + '.parquet')


run.finish()
