from yaml.loader import SafeLoader
import yaml
from d2l import torch as d2l

from contract_nli.preprocessing.data_management import DataService
from contract_nli.train import Train
from contract_nli.predict import Predict
from contract_nli.config.core import Core

# Open the file and load the file
with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
    print(config)

c = Core()
vocab_path = f"{c.TRAINED_MODEL_DIR}/{c.VOCAB_PATH}"
model_path = f"{c.TRAINED_MODEL_DIR}/{c.MODEL_PATH}"
train_path = f"{c.DATA_DIR}/{c.TRAIN_PATH}"
test_path = f"{c.DATA_DIR}/{c.TEST_PATH}"

args = config['Train']
ds = DataService()
train_set,test_set= ds.load_data(train_path, test_path)
device = d2l.try_all_gpus()
num_workers = d2l.get_dataloader_workers()
train_iter,test_iter,vocab= ds.create_snli_dataset(train=train_set, \
                                                   test=test_set, num_workers=num_workers)

#Train
tr = Train()
tr.run_training(train_iter, test_iter, vocab,args["learning_rate"],\
                args["epochs"],args["embed_size"],args["num_hiddens"],
                device, model_path, vocab_path) 

#Predict
pr = Predict(args["embed_size"],args["num_hiddens"], model_path, vocab_path)
pr.make_multiple_prediction_with_classification_report(test_set)


#Make Single Prediction
test_set.iloc[4, :]
pr.make_single_prediction(test_set.iloc[4, :]["premise"].split(), \
             test_set.iloc[4, :]["hypotheis"].split())