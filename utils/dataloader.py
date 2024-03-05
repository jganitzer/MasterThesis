import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader:
    def __init__(self, phenotype_path, snp_path, one_hot_encode = False, dataPercentage = 1):
        self.phenotype_path = phenotype_path
        self.snp_path = snp_path
        self.one_hot_encode = one_hot_encode
        self.dataPercentage = dataPercentage

    def load_data(self):
        df_pheno = pd.read_csv(self.phenotype_path, sep=" ", header=None, names=["id", "fixed_effect", "phenotype"])
        df_snp = pd.read_csv(self.snp_path, sep=" ", header=None)

        df_snp.columns = ["id"] + ["snp_" + str(i) for i in range(1, len(df_snp.columns))]
        df_snp_pheno = df_snp.merge(df_pheno, on="id").drop("fixed_effect", axis=1)

        # take random dataPercentage of the data
        df_snp_pheno = df_snp_pheno.sample(frac=self.dataPercentage, random_state=42)

        return df_snp_pheno
    
    def train_test_split_data(self, _test_size=0.2, _need_val=False, _val_size=0.25):
        df_snp_pheno = self.load_data()

        X = df_snp_pheno.drop(["phenotype"], axis=1)
        y = df_snp_pheno["phenotype"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, random_state=42)

        if _need_val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=_val_size, random_state=42)

        X_train = X_train.drop("id", axis=1)
        X_test = X_test.drop("id", axis=1)
       
        if _need_val:
            X_val = X_val.drop("id", axis=1)

        if self.one_hot_encode:
            X_train = np.array([one_hot_encode_along_channel_axis(seq) for seq in X_train.values])
            X_test = np.array([one_hot_encode_along_channel_axis(seq) for seq in X_test.values])

            if _need_val:
                X_val = np.array([one_hot_encode_along_channel_axis(seq) for seq in X_val.values])

        if _need_val:
            return X_train, X_test, y_train, y_test, X_val, y_val#
        else:
            return X_train, X_test, y_train, y_test

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