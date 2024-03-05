import numpy as np
import pandas as pd
import datetime

dataRoot = "/data/_"
heritability = "/h40"

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),3), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for idx, snp in np.ndenumerate(sequence):
        if snp == 0:
            zeros_array[idx][0] = 1
        elif snp == 1:
            zeros_array[idx][1] = 1
        elif snp == 2:
            zeros_array[idx][2] = 1
        else:
            raise RuntimeError("Unsupported value: " + str(snp))

df_snp_pheno = pd.read_parquet(dataRoot + heritability + '/_.parquet', engine='pyarrow')

df_ids = df_snp_pheno[["id"]]
y = df_snp_pheno["phenotype"]

df_snp_pheno.drop("id", axis=1, inplace=True)
df_snp_pheno.drop("phenotype", axis=1, inplace=True)

column_names = []
for column in df_snp_pheno.columns:
    for i in range(3):
        column_names.append(column + "_" + str(i))

df_snp_pheno_one_hot = np.array([one_hot_encode_along_channel_axis(seq) for seq in df_snp_pheno.values])
del df_snp_pheno
df_snp_pheno_one_hot_flatten = df_snp_pheno_one_hot.reshape(df_snp_pheno_one_hot.shape[0], -1)
del df_snp_pheno_one_hot
df_X_train_one_hot = pd.DataFrame(df_snp_pheno_one_hot_flatten, columns=column_names)
del df_snp_pheno_one_hot_flatten
df_X_train_one_hot = df_X_train_one_hot.assign(phenotype=y.values, id=df_ids['id'].values)
df_X_train_one_hot = df_X_train_one_hot.assign(id=df_ids['id'].values)
df_X_train_one_hot.to_parquet(dataRoot + heritability + "/_.parquet")
del df_X_train_one_hot


print("Finished at " + str(datetime.datetime.now()))