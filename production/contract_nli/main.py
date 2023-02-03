from yaml.loader import SafeLoader
import yaml
from d2l import torch as d2l

from contract_nli.preprocessing.data_management import DataService
from contract_nli.train import Train
from contract_nli.predict import Predict
from contract_nli.config.core import config, TRAINED_MODEL_DIR, DATA_DIR

# Open the file and load the file
trained_model_dir_path = TRAINED_MODEL_DIR.as_posix()
data_dir_path = DATA_DIR.as_posix()

vocab_path = f"{trained_model_dir_path}/{config.app_config.vocab_path}"
model_path = f"{trained_model_dir_path}/{config.app_config.model_path}"
train_path = f"{data_dir_path}/{config.app_config.train_path}"
test_path = f"{data_dir_path}/{config.app_config.test_path}"

ds = DataService()
train_set,test_set= ds.load_data(train_path, test_path)
device = d2l.try_all_gpus()
num_workers = d2l.get_dataloader_workers()
train_iter,test_iter,vocab= ds.create_snli_dataset(train=train_set, \
                                                   test=test_set, num_workers=num_workers)

#Train
model_config = config.model_config
tr = Train()
tr.run_training(train_iter, test_iter, vocab,model_config.learning_rate,\
                model_config.epochs,model_config.embed_size,model_config.num_hiddens,
                device, model_path, vocab_path) 

#Predict
pr = Predict(model_config.embed_size,model_config.num_hiddens, model_path, vocab_path)
pr.make_multiple_prediction_with_classification_report(test_set)


#Make Single Prediction
test_set.iloc[4, :]
pr.make_single_prediction(test_set.iloc[4, :]["premise"].split(), \
             test_set.iloc[4, :]["hypotheis"].split())