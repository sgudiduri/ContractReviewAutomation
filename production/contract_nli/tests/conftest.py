#!/usr/bin/python

'''
https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
conftest.py is used to define 
fixture used to define static data used by tests
External plugin
Hooks
Test root path
'''

import pytest
from contract_nli.config.core import config, TRAINED_MODEL_DIR
from contract_nli.predict import Predict

trained_model_dir_path = TRAINED_MODEL_DIR.as_posix()
model_config = config.model_config
embed_size=model_config.embed_size
num_hiddens=model_config.num_hiddens

@pytest.fixture()
def raw_app_config():
    #For larger datasets, here we would use a testing sub-sample.
    return config.app_config

@pytest.fixture()
def raw_model_config():
    return model_config

@pytest.fixture()
def load_predict_class():
    model_path =  f"{trained_model_dir_path}/{config.app_config.model_path}"
    vocab_path =  f"{trained_model_dir_path}/{config.app_config.vocab_path}"
    pr = Predict(model_config.embed_size,model_config.num_hiddens, model_path, vocab_path)
    return pr

@pytest.fixture()
def sample_input_data_1():
    row_1 = {
        "hypothesis":"Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement",
        "premise": "I the completion or termination of the dealings between the parties contemplated hereunder or",
        "result": "Entailment"  
        }
    return row_1

@pytest.fixture()
def sample_input_data_2():
    row_2 = {
        "hypothesis":"All Confidential Information shall be expressly identified by the Disclosing Party",
        "premise": "i marked confidential or proprietary or",
        "result": "Contradiction"    
        }
    return row_2

@pytest.fixture()
def sample_input_data_3():
    row_3 = {
        "hypothesis":"Receiving Party shall not reverse engineer any objects which embody Disclosing Party s Confidential Information",
        "premise": "6 Compelled Disclosure of Confidential Information",
        "result": "neutral"   
        }
    return row_3

