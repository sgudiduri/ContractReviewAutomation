import re
import json
import torch
import pickle
import numpy as np
import pandas as pd
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from nltk.tokenize import RegexpTokenizer

"""
Load train, dev, test data
"""
def load():

  global Pg
  Pg =  getPositionEncoding(100, d=100)
  Pg  = torch.FloatTensor(Pg)
  """
  Train
  """
  f = open('data/train.json') 
  data_train = json.load(f)

  kes=['nda-11', 'nda-16', 'nda-15', 'nda-10', 
       'nda-2', 'nda-1', 'nda-19', 'nda-12', 
       'nda-20', 'nda-3', 'nda-18', 'nda-7', 
       'nda-17', 'nda-8', 'nda-13', 'nda-5', 
       'nda-4']

  rg = len(data_train['documents'])
  reference_list=[]
  idx=0
  for idx in range(rg):
    docid = data_train['documents'][idx]['id']
    for ndas_key in kes:
      reference_list.append([docid, ndas_key,idx])
    idx=idx+1
  ref_train = pd.DataFrame(reference_list ,columns = ['docid', 'nda_key', 'idx'])
  f.close()

  """
  Test
  """
  f = open('data/test.json')  
  data_test = json.load(f)

  rg = len(data_test['documents'])
  reference_list=[]
  idxt=0
  for idxt in range(rg):
    #Document level
    docid = data_test['documents'][idxt]['id']

    for ndas_key in kes:
      reference_list.append([docid, ndas_key, idxt])
    idxt=idxt+1
  ref_test = pd.DataFrame(reference_list ,columns = ['docid', 'nda_key', 'idx'])

  """
  Validation
  """
  f = open('data/dev.json')  
  data_valid = json.load(f)

  rg = len(data_valid['documents'])
  reference_list=[]
  idxt=0
  for idxt in range(rg):
    #Document level
    docid = data_valid['documents'][idxt]['id']

    for ndas_key in kes:
      reference_list.append([docid, ndas_key, idxt])
    idxt=idxt+1
  ref_valid = pd.DataFrame(reference_list ,columns = ['docid', 'nda_key', 'idx'])

  return data_train, ref_train, data_valid, ref_valid, data_test, ref_test
 
def load_glove(d=100):

  global global_vectors 
  global_vectors = GloVe(name='6B', dim=d)
  return global_vectors

def get_data_non_fias(data, idx, ke, max_neutral=10):
  dataM=[]
  #Document level
  docid = data['documents'][idx]['id']
  
  string = data['documents'][idx]['text']
  file_name = data['documents'][idx]['file_name']
  spans = data['documents'][idx]['spans']
  spanall=[]
  for span in spans:           
      spanval = string[span[0]: span[1]]
      spanall.append(spanval)
  
  ndas = data['documents'][idx]['annotation_sets'][0]['annotations'] 
  # Keys level
    
  hypothesis = data['labels'][ke]['hypothesis']  

  # Key level   

  choice =ndas[ke] ['choice']
  spansC = ndas[ke] ['spans']    
  for si in range(len(spanall)):
    span_nbr =si
    if si in spansC:
      val=choice
    else:
      val="Neutral"
    premise = spanall[si]
    itm = [docid, file_name , ke, hypothesis, span_nbr, premise,val ]
    dataM.append(itm)
  df = pd.DataFrame(dataM ,columns = ['docid', 'file_name', 'hypotheis_key', 'hypotheis', 'span_nbr', 'premise', 'choice'   ])
  #"[CLS] " and " [SEP] "" [SEP]" 
  df['premise'] = df['premise'].apply(preprocess)
  df['hypotheis']= df['hypotheis'].apply(preprocess) 
  df['label'] = df['choice'].map(lambda s: 0 if s=='Entailment' else (1 if s== 'Contradiction' else 2 ))
  df['entl'] =  df['choice'].map(lambda s : 1 if s== 'Entailment' else 0)
  df['cont'] =  df['choice'].map(lambda s : 1 if s== 'Contradiction' else 0)
  df['neut'] =  df['choice'].map(lambda s : 1 if s== 'Neutral' else 0)
  
  df1=df[df.choice !='Neutral']
  df2=df[df.choice =='Neutral']
  n = min(max_neutral, len(df2)-1)
  df2= df2.sample(n = n)
  df = pd.concat([df1, df2], sort=False)
  df = df[df.span_nbr !=-1 ][df.premise != ''] [df.hypotheis != '']
  #df['bert_sent']= df.apply(lambda s: [s.premise,s.hypotheis], axis=1)
  return df


def get_data(data, idx, ke, use_faiss=True, max_neutral=20):  
  '''
  #Hack for testing so that epoch's dont fail.
  if not use_faiss:
    return get_data_non_fias(data, idx, ke, max_neutral=10)
  '''

  dataM=[]
  idxke= str(idx)+'|'+ke
  with open('samples.pickle', 'rb') as f:
    flt = pickle.load(f)
  
  docid = data['documents'][idx]['id']
  
  string = data['documents'][idx]['text']
  file_name = data['documents'][idx]['file_name']
  spans = data['documents'][idx]['spans']
  spanall=[]
  for span in spans:           
      spanval = string[span[0]: span[1]]
      spanall.append(spanval)
  
  ndas = data['documents'][idx]['annotation_sets'][0]['annotations'] 
    
  hypothesis = data['labels'][ke]['hypothesis']  

  choice =ndas[ke] ['choice']
  spansC = ndas[ke] ['spans']    
  for si in range(len(spanall)):
    span_nbr =si
    if si in spansC:
      val=choice
    else:
      val="Neutral"
    premise = spanall[si]
    itm = [docid, file_name , ke, hypothesis, span_nbr, premise,val ]
    dataM.append(itm)
  df = pd.DataFrame(dataM ,columns = ['docid', 'file_name', 'hypotheis_key', 'hypotheis', 'span_nbr', 'premise', 'choice'   ])
  
  
  df['premise'] = df['premise'].apply(preprocess)
  df['hypotheis']= df['hypotheis'].apply(preprocess) 
  df['label'] = df['choice'].map(lambda s: 0 if s=='Entailment' else (1 if s== 'Contradiction' else 2 ))
  df['entl'] =  df['choice'].map(lambda s : 1 if s== 'Entailment' else 0)
  df['cont'] =  df['choice'].map(lambda s : 1 if s== 'Contradiction' else 0)
  df['neut'] =  df['choice'].map(lambda s : 1 if s== 'Neutral' else 0)
   
  
  #df2= df2.sample(n = n)
  if use_faiss:
    df1=df[df.choice !='Neutral']
    spnNotNeut = df1.span_nbr.values.tolist() 
    df2=df[df.choice =='Neutral']
    res = flt[idxke]
    res = [eval(i) for i in res]
    res = list(set(res)-set(spnNotNeut))
    res = res[:10] # Selecting top 10 from 15 items
      #print(res)
    df2 = df2[df2.span_nbr.isin(res)]
    df = pd.concat([df1, df2], sort=False)
    df = df[df.span_nbr !=-1 ][df.premise != ''] [df.hypotheis != '']
    return df
  else:
    df1=df[df.choice !='Neutral']
    df2=df[df.choice =='Neutral']
    n = min(max_neutral, len(df2)-1)
    df2= df2.sample(n = n)
    df = pd.concat([df1, df2], sort=False)
    df = df[df.span_nbr !=-1 ][df.premise != ''] [df.hypotheis != '']    
    return df
  

def preprocess(sentence):
  sentence=str(sentence)
  sentence = sentence.lower().replace(","," ,").replace(";"," ;").replace(":"," :").replace("__"," ").replace("  "," ")
  sentence=sentence.replace('{html}',"") 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  rem_url=re.sub(r'http\S+', '',cleantext)
  rem_num = re.sub('[0-9]+', '', rem_url)
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(rem_num)  

  return " ".join(tokens)

def getPositionEncoding(max_length, d=100, n=10000):
    P = np.zeros((max_length, d))
    #print(max_length,P.shape)
    for k in range(max_length):      
      for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[k, 2*i] = np.sin(k/denominator)
        P[k, 2*i+1] = np.cos(k/denominator)
    return P

def emd(sent, max_length=100):
  tok = get_tokenizer("basic_english")
  tkn = tok(sent)  
  ln = len(tkn)
  x = global_vectors.get_vecs_by_tokens(tkn)
  if ln<max_length:  
    x = torch.nn.ConstantPad2d((0, 0, 0, max_length-ln), 0)(x)
  else:
    x= x[:max_length,:]
  x=x+Pg 
  return x

