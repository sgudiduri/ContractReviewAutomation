import torch
import d2l
from torch import nn
from model.decomposable_attention import DecomposableAttention

class Train():
    def __init__(self):
       pass
      
    def run_training(self, train_iter, test_iter,vocab, learning_rate, 
                            epochs, embed_size, num_hiddens,device,save_result: bool = True):         
        net = DecomposableAttention(vocab, embed_size, num_hiddens)
        glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
        embeds = glove_embedding[vocab.idx_to_token]
        net.embedding.weight.data.copy_(embeds)
        trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss(reduction="none")   
        d2l.train_ch13(net, train_iter, test_iter, loss, trainer, epochs, device)


def main():           

    from preprocessing.data_management import DataService
    from yaml.loader import SafeLoader
    import yaml

    # Open the file and load the file
    with open('contract_nli/config/config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
        print(config)

    args = config['Train']
    ds = DataService()
    if torch.cuda.is_available():
        device = d2l.try_all_gpus()
        num_workers = d2l.get_dataloader_workers()
    else:
       device = 'cpu'
       num_workers = 4

    train_set,test_set= ds.load_data('contract_nli/data/train.csv', 'contract_nli/data/test.csv')
    train_iter,test_iter,vocab= ds.create_snli_dataset(train=train_set, test=test_set, num_workers=num_workers)
    train_model = Train()
    
    train_model.run_training(train_iter, test_iter,vocab,args["learning_rate"], 
                                args["epochs"],args["embed_size"],args["num_hiddens"], device)    

if __name__ == "__main__":
    main()