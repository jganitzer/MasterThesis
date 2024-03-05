import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split

dataRoot = "/data/data"
heritability = "/h40"

def alleleFrequencyEncoding(df):
    for col in df.columns:
        # Count the number of 0s and 1s in the column
        D = (df[col] == 0).sum()
        H = (df[col] == 1).sum()
        N = df.shape[0]
        
        # Calculate p and q
        p = ((2*D) + H) / (2*N)
        q = 1 - p
        
        # Replace values in the column
        df[col] = df[col].apply(lambda x: p*p if x == 0 else (2*p*q if x == 1 else q*q))
    
    return df

df_snp_pheno = pd.read_parquet(dataRoot + heritability + '/snp_pheno_BPpos_benchmark.parquet', engine='pyarrow')

df_snp_pheno.drop(columns=['true', 'sstep', 'konv'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df_snp_pheno.drop(["phenotype"], axis=1), df_snp_pheno["phenotype"], test_size=0.1, random_state=42)

del df_snp_pheno

X_train.drop("id", axis=1, inplace=True)
X_test.drop("id", axis=1, inplace=True)

X_train_alleleEncoded = alleleFrequencyEncoding(X_train)

del X_train

X_train_alleleEncoded.to_parquet(dataRoot + heritability + "/train/snp_allelFreq_train_09.parquet")

del X_train_alleleEncoded

X_test_alleleEncoded = alleleFrequencyEncoding(X_test)

del X_test

X_test_alleleEncoded.to_parquet(dataRoot + heritability + "/test/snp_allelFreq_test_01.parquet")

print("Finished at " + str(datetime.datetime.now()))