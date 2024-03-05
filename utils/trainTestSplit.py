import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

dataRoot = "/data/_"
heritability = "/h40"

df_snp_pheno = pd.read_parquet(dataRoot + heritability + '/_.parquet', engine='pyarrow')

X_train, X_test, y_train, y_test = train_test_split(df_snp_pheno.drop(["phenotype"], axis=1), df_snp_pheno["phenotype"], test_size=0.1, random_state=42)

X_train.to_parquet(dataRoot + heritability + '/train/_.parquet')
X_test.to_parquet(dataRoot + heritability + '/test/_.parquet')

df_y_train = pd.DataFrame()
df_y_test = pd.DataFrame()
df_y_train=df_y_train.assign(phenotype=y_train.values)
df_y_test=df_y_test.assign(phenotype=y_test.values)

df_y_train.to_parquet(dataRoot + heritability + '/train/_.parquet')
df_y_test.to_parquet(dataRoot + heritability + '/test/_.parquet')

print("Finished at " + str(datetime.datetime.now()))

