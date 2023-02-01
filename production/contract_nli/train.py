import torch
from d2l import torch as d2l
from torch import nn
from contract_nli.model.decomposable_attention import DecomposableAttention

class Train():
    def __init__(self):
       pass
      
    def run_training(self, train_iter, test_iter,vocab, learning_rate, 
                            epochs, embed_size, num_hiddens,device,
                            save_path,vocab_path,save_result: bool = True):         
        net = DecomposableAttention(vocab, embed_size, num_hiddens)
        glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
        embeds = glove_embedding[vocab.idx_to_token]
        net.embedding.weight.data.copy_(embeds)
        trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss(reduction="none")   
        d2l.train_ch13(net, train_iter, test_iter, loss, trainer, epochs, device)
        
        if save_result:
          torch.save(net.state_dict(), save_path)
          torch.save(vocab, vocab_path)
