import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Dataset, Data
import random

dataRoot = "/data/_"
heritability = "/h40"
phenotype_path = dataRoot + heritability + heritability + "_simu.dat"
snp_path = dataRoot + heritability + heritability + "_simu.snp"
benchmark_path = dataRoot + heritability + heritability + "_simu.bv"
pedigree_path = dataRoot + heritability + heritability + "_simu.ped"
snpPos_path = dataRoot + heritability + heritability + "_simu_snp.txt"
qtl_path = dataRoot + heritability + heritability + "_simu_qtl.txt"

# load data
data = torch.load(dataRoot + heritability + "/graph_maf_temp.pt")

def split_data(_data, val_size=0.2, test_size=0.1, random_seed=42):
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Check if data.y contains NaN values
    y_has_nans = torch.isnan(_data.y)

    # Check if data.x has only zeros
    x_has_only_zeros = torch.sum(_data.x, dim=(1, 2)) == 0

    # Combine the conditions using logical OR
    has_nans = y_has_nans | x_has_only_zeros

    # Negate the boolean tensor to get the desired result
    has_no_nans = ~has_nans

    # Identify the indices where 'has_no_nans' is True
    true_indices = torch.where(has_no_nans)[0]

    # Determine the number of indices to change
    num_indices = len(true_indices)
    num_test_indices = int(num_indices * test_size)
    num_val_indices = int(num_indices * val_size)

    # Generate random indices for test, validation, and train sets
    indices = torch.randperm(num_indices)
    test_indices = true_indices[indices[:num_test_indices]]
    val_indices = true_indices[indices[num_test_indices:num_test_indices + num_val_indices]]
    train_indices = true_indices[indices[num_test_indices + num_val_indices:]]

    # Create the masks
    train_mask = torch.zeros_like(has_no_nans, dtype=torch.bool)
    val_mask = torch.zeros_like(has_no_nans, dtype=torch.bool)
    test_mask = torch.zeros_like(has_no_nans, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


train_mask, val_mask, test_mask = split_data(data, val_size=0.2, test_size=0.1, random_seed=42)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

torch.save(data, dataRoot + heritability + "/graph_maf2.pt")

print("Finished at " + str(datetime.datetime.now()))