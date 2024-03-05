import wandb
import pandas as pd
from utils.torchDataloader import OneHotLoader
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

X_path = "/train/snp_X_train_maf_one_hot.parquet"
y_path = "/train/snp_y_train_maf_one_hot.parquet"

#Config
lr = 0.1

n_estimators=5000
max_depth=3

val_size = 0.2
batch_size = 32
patience = 10
percentage_to_keep = 0.95

checkpoint_path="/data/checkpoints/" + wandb_run_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

config = {
    "Preprocessing": "maf",
    "lr": lr,
    "batch_size": batch_size,
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "val_size": val_size,
    "patience": patience,
}

run = wandb.init(project=wandb_project_name, name=wandb_run_name, entity=entity, config=config)


oneHotLoader = OneHotLoader(dataRoot + heritability, X_path, y_path, wandbDir=wandbDir, batch_size = batch_size, val_size = val_size)

train_loader, val_loader = oneHotLoader.getLoaders()

(X_train, y_train) = train_loader.dataset.tensors
(X_val, targets) = val_loader.dataset.tensors

model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, tree_method='gpu_hist', random_state=42, callbacks=[WandbCallback(log_model=True)], early_stopping_rounds=patience)

print("got Model")

eval_set = [(X_val.flatten(start_dim=1).numpy(), targets.numpy())]

model.fit(X_train.flatten(start_dim=1).numpy(), y_train.numpy(), eval_set=eval_set, verbose=True, eval_metric="rmse")

print("finished model fit")

output = model.predict(X_val.flatten(start_dim=1))

print("predicted output")

loss = mean_squared_error(targets.cpu().numpy(), output)
mae = mean_absolute_error(targets.cpu().numpy(), output)
r2 = r2_score(targets.cpu().numpy(), output)
_pearsonr, _pvalue = pearsonr(targets.cpu().numpy(), output)

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

# All regression plots
wandb.sklearn.plot_regressor(model, X_train.flatten(start_dim=1).numpy(), X_val.flatten(start_dim=1).numpy(), y_train.numpy(), targets.numpy(), model_name=wandb_run_name)

model.save_model(checkpoint_path + "model.json")

run.finish()
