from yaml.loader import SafeLoader
import yaml

import torch
import torchtext
from torchtext import datasets
from torch import nn
from d2l import torch as d2l

from preprocessing.data_management import DataService
from model.decomposable_attention import DecomposableAttention
from train import Train
from predict import Predict

# Open the file and load the file
with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
    print(config)

args = config['Train']
ds = DataService()
train_set,test_set= ds.load_data('data/train.csv', 'data/test.csv')
device = d2l.try_all_gpus()
num_workers = d2l.get_dataloader_workers()
train_iter,test_iter,vocab= ds.create_snli_dataset(train=train_set, \
                                                   test=test_set, num_workers=num_workers)

#Train
tr = Train()
tr.run_training(train_iter, test_iter, vocab,args["learning_rate"],\
                args["epochs"],args["embed_size"],args["num_hiddens"], device) 


#Predict
pr = Predict(args["embed_size"],args["num_hiddens"])
pr.make_multiple_prediction_with_classification_report(test_set)

#Make Single Prediction
test_set.iloc[4, :]
pr.make_single_prediction(test_set.iloc[4, :]["premise"].split(), \
             test_set.iloc[4, :]["hypotheis"].split())