import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Dataset, Data
import random

dataRoot = "/data/jganitzer/masterarbeit/data"
heritability = "/h40"
phenotype_path = dataRoot + heritability + heritability + "_simu.dat"
snp_path = dataRoot + heritability + heritability + "_simu.snp"
benchmark_path = dataRoot + heritability + heritability + "_simu.bv"
pedigree_path = dataRoot + heritability + heritability + "_simu.ped"
snpPos_path = dataRoot + heritability + heritability + "_simu_snp.txt"
qtl_path = dataRoot + heritability + heritability + "_simu_qtl.txt"


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
        elif snp == -7:
            zeros_array[idx][0] = 0
            zeros_array[idx][1] = 0
            zeros_array[idx][2] = 0
        else:
            raise RuntimeError("Unsupported value: " + str(snp))


# load data
data = torch.load(dataRoot + heritability + "/graph_maf_float32_gen30_noSireGeno_cat.pt")

foo = data.x - 7
numpy_array = foo.numpy()
df = pd.DataFrame(numpy_array)

x_oneHot = np.array([one_hot_encode_along_channel_axis(seq) for seq in df.values])

tensor = torch.tensor(x_oneHot)

data.x_oneHot = tensor

torch.save(data, dataRoot + heritability + "/graph_maf_float32_gen30_noSireGeno_cat_oneHot.pt")

print(data.x_oneHot.shape)

print("Finished at " + str(datetime.datetime.now()))