import os

class Core():
    def __init__(self):        
        self.PWD = os.path.dirname(os.path.abspath(__file__))
        self.PACKAGE_ROOT = os.path.abspath(os.path.join(self.PWD, '..'))
        self.TRAINED_MODEL_DIR = os.path.join(self.PACKAGE_ROOT, 'trained_model')
        self.DATA_DIR = os.path.join(self.PACKAGE_ROOT, 'data')

        #DATA PATH
        self.TRAIN_PATH = "train.csv"
        self.TEST_PATH = "test.csv"

        # MODEL PERSISTING
        self.VOCAB_PATH = "vocab.pth"
        self.MODEL_PATH = "model.pth"

        self.embed_size= 100
        self.num_hiddens= 200  