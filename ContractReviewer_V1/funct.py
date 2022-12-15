
# This function is used by CreateFAISSsample.ipynbd to create FAISS file
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sent_embs(file, json, ke, premise=True):
    kes=['nda-11', 'nda-16', 'nda-15', 'nda-10', 'nda-2', 'nda-1', 'nda-19', 'nda-12', 'nda-20', 'nda-3', 'nda-18', 'nda-7', 'nda-17', 'nda-8', 'nda-13', 'nda-5', 'nda-4']

    flnmidx={}
    for dcl in range(len(json['documents'])):
        flnmidx[json['documents'][dcl]['file_name']]= dcl
    idx= flnmidx[file]
    ad =get_data(json, idx,ke, max_neutral=5000)  
    if premise:
        sentences = ad.premise.values.tolist()
        span_nbr = ad.span_nbr.values.tolist()
        '''
        spans= pdf_struct.predict(
                            format='paragraphs',
                            in_path=x[0],
                            model='PDFContractEnFeatureExtractor'
                          )
        sentences = pd.DataFrame(spans)[0].apply(preprocess).tolist()
        '''
    else:
        sentences = [ json['labels'][ke]['hypothesis'] for ke in kes]
        #[ad[ad.hypotheis_key==ke].hypotheis.values[0]]
        span_nbr=kes
    
    
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-bert-base',  model_max_length=300)
    model = AutoModel.from_pretrained('sentence-transformers/nli-bert-base')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return span_nbr, sentences, embeddings

# Do Not delete
def get_data(data, idx,ke, max_neutral=5000):
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
  df['bert_sent']= df.apply(lambda s: [s.premise,s.hypotheis], axis=1)
  return df

def create_completedf(data, refdf, max_neutral=50):
  cl = get_data(data, 0,'nda-1', 1)
  cols = list(cl.columns)
  df1= pd.DataFrame(columns=cols)
  for ind, row in refdf.iterrows():    
    df2 = get_data(data, row["idx"],row["nda_key"], max_neutral)
    df1=pd.concat([df1,df2])
  return df1

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
nltk.download('omw-1.4')
import re
nltk.download('stopwords')
nltk.download('wordnet')


mlngt = 100 #max_length,
mn =10 #max_neutral

def preprocess(sentence):
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer() 
  sentence=str(sentence)
  sentence = sentence.lower().replace(","," ,").replace(";"," ;").replace(":"," :").replace("__"," ").replace("  "," ")
  sentence=sentence.replace('{html}',"") 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  rem_url=re.sub(r'http\S+', '',cleantext)
  rem_num = re.sub('[0-9]+', '', rem_url)
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(rem_num)  
  #filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
  #stem_words=[stemmer.stem(w) for w in filtered_words]
  #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
  return " ".join(tokens)

