import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')
import utility  as utl

class MyDataset(Dataset): 
  def __init__(self,data, df, use_faiss =True, max_neutral=20):       
    self.df =df.sample(frac=1)
    self.data =data
    self.use_faiss =use_faiss
    self.max_neutral= max_neutral
    utl.load_glove()

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Is should have used docid but I am using index in json. Check how ref files are structured
    docidx = self.df.iloc[idx]['idx']
    ndakey = self.df.iloc[idx]['nda_key']
    df1 = utl.get_data(self.data, docidx, ndakey,self.use_faiss, self.max_neutral)
    label = torch.tensor(df1.label.values)
    premise = torch.stack(df1.premise.apply(utl.emd).tolist())
    hypothesis =  torch.stack(df1.hypotheis.apply(utl.emd).tolist())
    sample = {"Label": label.to(device), "Premise": premise.to(device) , "Hypothesis" : hypothesis.to(device)}
    return sample

# Custome collate function
def coll(sample1):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if len(sample1)<2:
    return sample1
  else:
    label0 = sample1[0]['Label'].to(device)
    premise0 = sample1[0]['Premise'].to(device)
    hypothesis0 = sample1[0]['Hypothesis'].to(device)
    #bertsent0 = sample1[0]["Bert_Sent"].to(device)
    for i in range(1, len(sample1)):
      labeli = sample1[i]['Label'].to(device)
      premisei = sample1[i]['Premise'].to(device)
      hypothesisi = sample1[i]['Hypothesis'].to(device)
      #bertsenti = sample1[i]["Bert_Sent"].to(device)
      label0 =torch.cat((label0,labeli))
      premise0 = torch.cat((premise0, premisei))
      hypothesis0 = torch.cat((hypothesis0,hypothesisi))
      #bertsent0 = torch.cat((bertsent0, bertsenti))
    #sample1 = {"Label": label0, "Premise": premise0 , "Hypothesis" : hypothesis0, "Bert_Sent": bertsent0}
    sample1 = {"Label": label0, "Premise": premise0 , "Hypothesis" : hypothesis0}
    return sample1
