import pandas as pd
import torch
import d2l
from .snli_dataset import SNLIDataset

class DataService():
    def __init__(self): 
       pass 

    #loads csv into pandas dataframe.
    def load_data(self, train_path, test_path):    
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train["label"] = train["label"].astype(int)
        test["label"] = test["label"].astype(int)
        return train, test

    #load pandas dataframe to dataset.
    def create_snli_dataset(self, train, test, num_steps = 50, batch_size = 256, num_workers = 4):        
        train_set = SNLIDataset(train, num_steps)
        test_set = SNLIDataset(test, num_steps, train_set.vocab)
        vocab = train_set.vocab
        train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
        return train_iter, test_iter, vocab