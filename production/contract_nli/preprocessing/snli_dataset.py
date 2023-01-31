import torch
from d2l import torch as d2l

''''
This class is responsible for creating SNLI dataset from pandas dataframe.
'''
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset["hypotheis"])
        all_hypothesis_tokens = d2l.tokenize(dataset["premise"])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset["label"])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        torch.manual_seed(123)
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        torch.manual_seed(123)
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        torch.manual_seed(123)
        return len(self.premises)
