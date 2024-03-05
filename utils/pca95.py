from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import gc
import datetime
import pandas as pd

pca95 = PCA(n_components=0.95, copy=False)

dataRoot = "/data"
heritability = "/h40"
test_size = 0.15

X_train = pd.read_parquet(dataRoot + heritability + '/train/snp_X_train_noStaticSNP.parquet', engine='pyarrow')
y_train = pd.read_parquet(dataRoot + heritability + '/train/snp_y_train_noStaticSNP.parquet', engine='pyarrow')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train["phenotype"], test_size=test_size, random_state=42)


# y_train to dataframe
df_y_train = pd.DataFrame()
df_y_val = pd.DataFrame()
df_y_train=df_y_train.assign(phenotype=y_train.values)
df_y_val=df_y_val.assign(phenotype=y_val.values)

df_y_train.to_parquet(dataRoot + heritability + '/train/snp_y_train_0765_pca95.parquet')
df_y_val.to_parquet(dataRoot + heritability + '/val/snp_y_val_0135_pca95.parquet')

del y_train
del y_val
del df_y_train
del df_y_val
gc.collect()

df_train_ids = X_train[["id"]]
X_train.drop("id", axis=1, inplace=True)

X_train = pca95.fit_transform(X_train)


df_train_new = pd.DataFrame(X_train, columns=['PC' + str(i) for i in range(1, X_train.shape[1] + 1)])

del X_train
gc.collect()

df_train_new = df_train_new.assign(id=df_train_ids['id'].values)

del df_train_ids
gc.collect()

df_train_new.to_parquet(dataRoot + heritability + "/train/snp_X_train_0765_pca95.parquet")

del df_train_new
gc.collect()


df_val_ids = X_val[["id"]]
X_val.drop("id", axis=1, inplace=True)
X_val = pca95.transform(X_val)

df_val_new = pd.DataFrame(X_val, columns=['PC' + str(i) for i in range(1, X_val.shape[1] + 1)])

del X_val
gc.collect()

df_val_new = df_val_new.assign(id=df_val_ids['id'].values)

del df_val_ids
gc.collect()

df_val_new.to_parquet(dataRoot + heritability + "/val/snp_X_val_0135_pca95.parquet")

del df_val_new
gc.collect()

X_test = pd.read_parquet(dataRoot + heritability + '/test/snp_X_test_01_noStaticSNP.parquet', engine='pyarrow')
df_test_ids = X_test[["id"]]
X_test.drop("id", axis=1, inplace=True)
X_test = pca95.transform(X_test)

df_test_new = pd.DataFrame(X_test, columns=['PC' + str(i) for i in range(1, X_test.shape[1] + 1)])

del X_test
gc.collect()

df_test_new = df_test_new.assign(id=df_test_ids['id'].values)

del df_test_ids
gc.collect()

df_test_new.to_parquet(dataRoot + heritability + "/test/snp_X_test_01_pca95.parquet")

del df_test_new
gc.collect()

print ("Covariance: " + pca95.get_covariance())

print("Explained variance Ratio: ")
print(pca95.explained_variance_ratio_.cumsum())

print("Finished at " + str(datetime.datetime.now()))