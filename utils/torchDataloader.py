import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from utils.genomicDataset import train_test_dataset_split, GenomicDataset3, GenomicDataset2, GenomicDataset4
from utils.tokenizer import CharacterTokenizer

class OneHotLoader:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir
        
    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        X_train = X_train_df.to_numpy()
        X_val = X_val_df.to_numpy()

        # Reshape the data back to its original shape
        X_train = X_train.reshape(X_train_df.shape[0], int(X_train_df.shape[1]/3), 3)
        X_val = X_val.reshape(X_val_df.shape[0], int(X_val_df.shape[1]/3), 3)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cpu')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
    
# define a function to add a value to each element
def add_value(x, value):
    return x + value

class CatLoader:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir

    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        # apply the function to each element of the DataFrame
        #X_train_df = X_train_df.apply(add_value, value=4)
        #X_val_df = X_val_df.apply(add_value, value=4)
      
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
    
class OneHotMaskLoader:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir
        self.paddedLength = paddedLength
        
    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        X_train = X_train_df.to_numpy()
        X_val = X_val_df.to_numpy()

        # Reshape the data back to its original shape
        X_train = X_train.reshape(X_train_df.shape[0], int(X_train_df.shape[1]/3), 3)
        X_val = X_val.reshape(X_val_df.shape[0], int(X_val_df.shape[1]/3), 3)
        
        # Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
        attention_mask_train = torch.ones(X_train.shape[0], X_train.shape[1]).to('cpu')
        attention_mask_val = torch.ones(X_val.shape[0], X_val.shape[1]).to('cpu')

        if(self.paddedLength != -1 and self.paddedLength > X_train.shape[1]):
            pad = self.paddedLength - X_train.shape[1]
            # Pad the data to the max length
            X_train = np.pad(X_train, ((0,0),(0,pad),(0,0)), 'constant', constant_values=0)
            X_val = np.pad(X_val, ((0,0),(0,pad),(0,0)), 'constant', constant_values=0)
            attention_mask_train = F.pad(attention_mask_train, (0,pad), 'constant', 0)
            attention_mask_val = F.pad(attention_mask_val, (0,pad), 'constant', 0)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cpu')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, attention_mask_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
    

class CatMaskLoader:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir
        self.paddedLength = paddedLength

    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        X_train_df = X_train_df.apply(add_value, value=1)
        X_val_df = X_val_df.apply(add_value, value=1)
        
        # Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
        attention_mask_train = torch.ones(X_train_df.shape[0], X_train_df.shape[1]).to('cpu')
        attention_mask_val = torch.ones(X_val_df.shape[0], X_val_df.shape[1]).to('cpu')

        if self.paddedLength != -1 and self.paddedLength > X_train_df.shape[1]:
            pad = self.paddedLength - X_train_df.shape[1]
            
            new_columns = list(X_train_df.columns) + ['pad_' + str(i) for i in range(pad)]
            
            # Pad the data to the max length
            X_train_df = pd.DataFrame(np.pad(X_train_df.values, ((0, 0), (0, pad)), 'constant', constant_values=0), columns = new_columns)
            X_val_df = pd.DataFrame(np.pad(X_val_df.values, ((0, 0), (0, pad)), 'constant', constant_values=0), columns = new_columns)

            attention_mask_train = F.pad(attention_mask_train, (0, pad), 'constant', 0)
            attention_mask_val = F.pad(attention_mask_val, (0, pad), 'constant', 0)
        
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int8).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int8).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, attention_mask_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader


class CatMaskLoaderCLS:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir
        self.paddedLength = paddedLength
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''

        self.CLS_TOKEN_VALUE = 0
        self.SEP_TOKEN_VALUE = 1

    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        # apply the function to each element of the DataFrame
        X_train_df = X_train_df.apply(add_value, value=7)
        X_val_df = X_val_df.apply(add_value, value=7)
       
        
        # Add CLS and SEP tokens to train and val sequences
        X_train_df.insert(0, 'CLS', self.CLS_TOKEN_VALUE)
        X_val_df.insert(0, 'CLS', self.CLS_TOKEN_VALUE)
        
        X_train_df['SEP'] = self.SEP_TOKEN_VALUE
        X_val_df['SEP'] = self.SEP_TOKEN_VALUE
       
        # Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
        attention_mask_train = torch.ones(X_train_df.shape[0], X_train_df.shape[1]).to('cpu')
        attention_mask_val = torch.ones(X_val_df.shape[0], X_val_df.shape[1]).to('cpu') 
     
        if self.paddedLength != -1 and self.paddedLength > X_train_df.shape[1]:
            pad = self.paddedLength - X_train_df.shape[1]
            
            # Add column names to the padded data frames
            new_columns = list(X_train_df.columns) + ['pad_' + str(i) for i in range(pad)]
            
            # Pad the data to the max length
            X_train_df = pd.DataFrame(np.pad(X_train_df.values, ((0, 0), (0, pad)), 'constant', constant_values=4), columns = new_columns)
            X_val_df = pd.DataFrame(np.pad(X_val_df.values, ((0, 0), (0, pad)), 'constant', constant_values=4), columns = new_columns)

            #attention_mask_train = F.pad(attention_mask_train, (0, pad), 'constant', 0)
            #attention_mask_val = F.pad(attention_mask_val, (0, pad), 'constant', 0)     

            attention_mask_train = F.pad(attention_mask_train, (0, pad), 'constant', 0)
            attention_mask_val = F.pad(attention_mask_val, (0, pad), 'constant', 0)          
              
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int8).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int8).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, attention_mask_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
        
        
class CatMaskLoaderCLS2:
    def __init__(self, file_path, label_path, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.paddedLength = paddedLength
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):

        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add SNP characters, 
            model_max_length= self.paddedLength,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=True    
        )

        dataset = GenomicDataset3(file_path = self.file_path, label_path = self.label_path, max_length = self.paddedLength, tokenizer = tokenizer, use_padding = True)

        train_ratio = 1- self.val_size
        train_dataset, val_dataset = train_test_dataset_split(dataset, train_ratio=train_ratio, random_seed=42)            

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader

class CatMaskLoaderCLSFine:
    def __init__(self, train_file_path, train_label_path, test_file_path, test_label_path, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.train_file_path = train_file_path
        self.train_label_path = train_label_path
        self.test_file_path = test_file_path
        self.test_label_path = test_label_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.paddedLength = paddedLength
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):

        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add SNP characters, 
            model_max_length= self.paddedLength,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=True    
        )

        train_dataset = GenomicDataset3(file_path = self.train_file_path, label_path = self.train_label_path, max_length = self.paddedLength, tokenizer = tokenizer, use_padding = True)
        val_dataset = GenomicDataset3(file_path = self.test_file_path, label_path = self.test_label_path, max_length = self.paddedLength, tokenizer = tokenizer, use_padding = True)        

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader

class CatLoaderCLSFine:
    def __init__(self, train_file_path, train_label_path, test_file_path, test_label_path, batch_size = 16, val_size = 0.2, model_max_length = -1):
        self.train_file_path = train_file_path
        self.train_label_path = train_label_path
        self.test_file_path = test_file_path
        self.test_label_path = test_label_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.model_max_length = model_max_length
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):
        
        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add sNP characters, 
            model_max_length= self.model_max_length,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=False    
        )

        train_dataset = GenomicDataset2(file_path = self.train_file_path, label_path = self.train_label_path, max_length = self.model_max_length, tokenizer = tokenizer, use_padding = False)
        val_dataset = GenomicDataset2(file_path = self.test_file_path, label_path = self.test_label_path, max_length = self.model_max_length, tokenizer = tokenizer, use_padding = False)        

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader


class CatLoaderCLSTokenizer:
    def __init__(self, file_path, label_path, batch_size = 16, val_size = 0.2, model_max_length = -1):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.model_max_length = model_max_length
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):

        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add SNP characters, 
            model_max_length= self.model_max_length,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=True    
        )

        dataset = GenomicDataset2(file_path = self.file_path, label_path = self.label_path, max_length = self.model_max_length, tokenizer = tokenizer, use_padding = False, add_special_tokens = True)

        train_ratio = 1- self.val_size
        train_dataset, val_dataset = train_test_dataset_split(dataset, train_ratio=train_ratio, random_seed=42)            

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader

class CatLoaderCLSTokenizerFine:
    def __init__(self, file_path_train, label_path_train,file_path_test, label_path_test, batch_size = 16, val_size = 0.2, model_max_length = -1):
        self.file_path_train = file_path_train
        self.label_path_train = label_path_train
        self.file_path_test = file_path_test
        self.label_path_test = label_path_test
        self.batch_size = batch_size
        self.val_size = val_size
        self.model_max_length = model_max_length
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):

        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add SNP characters, 
            model_max_length= self.model_max_length,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=True    
        )

        dataset_train = GenomicDataset2(file_path = self.file_path_train, label_path = self.label_path_train, max_length = self.model_max_length, tokenizer = tokenizer, use_padding = False, add_special_tokens = True)
        dataset_test = GenomicDataset2(file_path = self.file_path_test, label_path = self.label_path_test, max_length = self.model_max_length, tokenizer = tokenizer, use_padding = False, add_special_tokens = True)
          
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader

class EmbeddingLoader:
    def __init__(self, dataRoot, X_path, y_path,wandbDir, batch_size = 16, val_size = 0.2):
        self.dataRoot = dataRoot
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.wandbDir = wandbDir

    def getLoaders(self):
        X = pd.read_parquet(self.dataRoot + self.X_path, engine='pyarrow')
        y = pd.read_parquet(self.dataRoot + self.y_path, engine='pyarrow')["phenotype"]

        X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)

        X_train_df.drop("id", axis=1, inplace=True)
        X_val_df.drop("id", axis=1, inplace=True)
        
        # apply the function to each element of the DataFrame
        #X_train_df = X_train_df.apply(add_value, value=4)
        #X_val_df = X_val_df.apply(add_value, value=4)
      
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.float).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.float).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
        
class CatLoaderCLSTokenizerPad:
    def __init__(self, file_path, label_path, batch_size = 16, val_size = 0.2, paddedLength = -1):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.paddedLength = paddedLength
        '''
        "[CLS]": 0
        "[SEP]": 1
        "[BOS]": 2
        "[MASK]": 3
        "[PAD]": 4
        "[RESERVED]": 5
        "[UNK]": 6
        an id (starting at 7) will be assigned to each character.
        model_max_length (int): Model maximum sequence length.
        '''


    def getLoaders(self):

        tokenizer = CharacterTokenizer(
            characters=['0', '1', '2'],  # add SNP characters, 
            model_max_length= self.paddedLength,
            add_special_tokens=True, 
            padding_side='right',
            return_special_tokens_mask=True    
        )

        dataset = GenomicDataset4(file_path = self.file_path, label_path = self.label_path, max_length = self.paddedLength, tokenizer = tokenizer, use_padding = True)

        train_ratio = 1- self.val_size
        train_dataset, val_dataset = train_test_dataset_split(dataset, train_ratio=train_ratio, random_seed=42)            

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)

        return train_loader, val_loader
        
class OneHotMaskLoaderFine:
    def __init__(self, dataRoot, trainPath, testPath,wandbDir, batch_size = 16, paddedLength = -1):
        self.dataRoot = dataRoot
        self.trainPath = trainPath
        self.testPath = testPath
        self.batch_size = batch_size
        self.wandbDir = wandbDir
        self.paddedLength = paddedLength
        
    def getLoaders(self):
        
        df_snp_pheno_train = pd.read_parquet(self.dataRoot + self.trainPath, engine='pyarrow')
        df_snp_pheno_test = pd.read_parquet(self.dataRoot + self.testPath, engine='pyarrow')

        X_train_df = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
        X_val_df = df_snp_pheno_test.drop(columns=['phenotype', 'id'], axis=1)
                
        y_train = df_snp_pheno_train["phenotype"]
        y_val = df_snp_pheno_test["phenotype"]
                      
        X_train = X_train_df.to_numpy()
        X_val = X_val_df.to_numpy()

        # Reshape the data back to its original shape
        X_train = X_train.reshape(X_train_df.shape[0], int(X_train_df.shape[1]/3), 3)
        X_val = X_val.reshape(X_val_df.shape[0], int(X_val_df.shape[1]/3), 3)
        
        # Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
        attention_mask_train = torch.ones(X_train.shape[0], X_train.shape[1]).to('cpu')
        attention_mask_val = torch.ones(X_val.shape[0], X_val.shape[1]).to('cpu')

        if(self.paddedLength != -1 and self.paddedLength > X_train.shape[1]):
            pad = self.paddedLength - X_train.shape[1]
            # Pad the data to the max length
            X_train = np.pad(X_train, ((0,0),(0,pad),(0,0)), 'constant', constant_values=0)
            X_val = np.pad(X_val, ((0,0),(0,pad),(0,0)), 'constant', constant_values=0)
            attention_mask_train = F.pad(attention_mask_train, (0,pad), 'constant', 0)
            attention_mask_val = F.pad(attention_mask_val, (0,pad), 'constant', 0)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cpu')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, attention_mask_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
        
class CatLoaderFine:
    def __init__(self, dataRoot, trainPath, testPath,wandbDir, batch_size = 16):
        self.dataRoot = dataRoot
        self.trainPath = trainPath
        self.testPath = testPath
        self.batch_size = batch_size
        self.wandbDir = wandbDir

    def getLoaders(self):
        df_snp_pheno_train = pd.read_parquet(self.dataRoot + self.trainPath, engine='pyarrow')
        df_snp_pheno_test = pd.read_parquet(self.dataRoot + self.testPath, engine='pyarrow')

        X_train_df = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
        X_val_df = df_snp_pheno_test.drop(columns=['phenotype', 'id'], axis=1)
        y_train = df_snp_pheno_train["phenotype"]
        y_val = df_snp_pheno_test["phenotype"]
        
        # apply the function to each element of the DataFrame
        #X_train_df = X_train_df.apply(add_value, value=7)
        #X_val_df = X_val_df.apply(add_value, value=7)
      
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader

class OneHotLoaderFine:
    def __init__(self, dataRoot,trainPath, testPath,wandbDir, batch_size = 16):
        self.dataRoot = dataRoot
        self.trainPath = trainPath
        self.testPath = testPath
        self.batch_size = batch_size
        self.wandbDir = wandbDir
        
    def getLoaders(self):

        df_snp_pheno_train = pd.read_parquet(self.dataRoot + self.trainPath, engine='pyarrow')
        df_snp_pheno_test = pd.read_parquet(self.dataRoot + self.testPath, engine='pyarrow')

        X_train_df = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
        X_val_df = df_snp_pheno_test.drop(columns=['phenotype', 'id', 'sire', 'dam'], axis=1)
        y_train = df_snp_pheno_train["phenotype"]
        y_val = df_snp_pheno_test["phenotype"]
        
        X_train = X_train_df.to_numpy()
        X_val = X_val_df.to_numpy()

        # Reshape the data back to its original shape
        X_train = X_train.reshape(X_train_df.shape[0], int(X_train_df.shape[1]/3), 3)
        X_val = X_val.reshape(X_val_df.shape[0], int(X_val_df.shape[1]/3), 3)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cpu')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True,num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader
        
class CatMaskLoaderfine:
    def __init__(self, dataRoot, trainPath, testPath,wandbDir, batch_size = 16,  paddedLength = -1):
        self.dataRoot = dataRoot
        self.trainPath = trainPath
        self.testPath = testPath
        self.batch_size = batch_size
        self.wandbDir = wandbDir
        self.paddedLength = paddedLength

    def getLoaders(self):

        df_snp_pheno_train = pd.read_parquet(self.dataRoot + self.trainPath, engine='pyarrow')
        df_snp_pheno_test = pd.read_parquet(self.dataRoot + self.testPath, engine='pyarrow')

        X_train_df = df_snp_pheno_train.drop(columns=['phenotype', 'id'], axis=1)
        X_val_df = df_snp_pheno_test.drop(columns=['phenotype', 'id'], axis=1)
        y_train = df_snp_pheno_train["phenotype"]
        y_val = df_snp_pheno_test["phenotype"]
        
        X_train_df = X_train_df.apply(add_value, value=7)
        X_val_df = X_val_df.apply(add_value, value=7)
        
        # Create attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) for train and val
        attention_mask_train = torch.ones(X_train_df.shape[0], X_train_df.shape[1]).to('cpu')
        attention_mask_val = torch.ones(X_val_df.shape[0], X_val_df.shape[1]).to('cpu')

        if self.paddedLength != -1 and self.paddedLength > X_train_df.shape[1]:
            pad = self.paddedLength - X_train_df.shape[1]
            
            new_columns = list(X_train_df.columns) + ['pad_' + str(i) for i in range(pad)]
            
            # Pad the data to the max length
            X_train_df = pd.DataFrame(np.pad(X_train_df.values, ((0, 0), (0, pad)), 'constant', constant_values=3), columns = new_columns)
            X_val_df = pd.DataFrame(np.pad(X_val_df.values, ((0, 0), (0, pad)), 'constant', constant_values=3), columns = new_columns)

            attention_mask_train = F.pad(attention_mask_train, (0, pad), 'constant', 0)
            attention_mask_val = F.pad(attention_mask_val, (0, pad), 'constant', 0)
        
        X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.int8).to('cpu')
        X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.int8).to('cpu')

        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to('cpu')
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to('cpu')

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, attention_mask_train)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, attention_mask_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True, num_workers = 2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True, num_workers = 2)

        return train_loader, val_loader