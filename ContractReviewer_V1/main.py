import yaml
import argparse
import time
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utility import load
from modules import DecomposableAttention
from data_.my_dataset import MyDataset, coll
from losses import FocalLoss, reweight
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='CS7643 Contract Reviewer')
parser.add_argument('--config', default='configs/config.yaml') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#This method is used to load configuration
def load_config():
#Load configuration    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    return args

# Run this when setting up configurations via notebook
def load_config_notebook():
  class Args:
    batch_size= 64
    learning_rate= 0.001
    reg= 0.0001
    epochs= 10
    steps= [6, 8]
    warmup= 0
    momentum= 0.9
    gamma= 1
    beta= .9999
    max_netural: 10
    save_best: True

  return Args()

#Method to load train, test and validate data set.
def load_train_test(batch_size):
    data_train, ref_train, data_valid, ref_valid, data_valid, ref_test = load()     

    # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders     
    train_data=MyDataset(data_train, ref_train)
    train_loader=DataLoader(train_data,batch_size=batch_size, collate_fn=coll, shuffle=False)

    valid_data=MyDataset(data_valid, ref_valid, use_faiss=False, max_neutral=5)
    valid_loader=DataLoader(valid_data,batch_size=batch_size, collate_fn=coll, shuffle=False)

    test_data=MyDataset(data_valid, ref_test, use_faiss=False, max_neutral=5)
    test_loader=DataLoader(test_data,batch_size=batch_size, collate_fn=coll, shuffle=False)

    return train_loader, valid_loader, test_loader

"""
This function computes the number of labels for each of
ENTAILMENT, CONTRADICTION and NOTMENTIONED classes in
preparation to use such counts in the focal loss 
implementation at a later stage in the code.
"""
def get_counts_training_data(train_loader):
    entailment = 0
    contradiction = 0
    neutral = 0

    for i, x in enumerate(train_loader):
        a = x['Label'].bincount().cpu().numpy()
        entailment += a[0]
        contradiction += a[1]
        neutral += a[2]

    # We'll feed this list to the focal loss implementation.
    cls_num_list = list([entailment,contradiction,neutral])
    return cls_num_list

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    _, pred = torch.max(output, dim=-1)
    correct = pred.eq(target).sum() * 1.0
    acc = correct / batch_size
    return acc

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
   
    for idx, x in enumerate(data_loader): 
        start = time.time()
        optimizer.zero_grad()
        outputs = model.forward(x)
       
        loss = criterion(outputs, x["Label"])
        loss.backward()
        optimizer.step()        
        batch_acc = accuracy(outputs, x["Label"])
        losses.update(loss.item(), outputs.shape[0])
        acc.update(batch_acc, outputs.shape[0])
        
        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))

def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 3
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, x in enumerate(val_loader):
        start = time.time()   

        torch.no_grad()
        out = model(x)
        loss = criterion(out, x["Label"])     
        batch_acc = accuracy(out, x["Label"])

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(x["Label"], preds.view(-1)):
            cm[t.long(), p.long()] += 1       

        losses.update(loss.item())
        acc.update(batch_acc)

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist() 
    
    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm
    

def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    #Load config for command line
    #args = load_config()
    print("\r Loading config")
    #Load config for jupyter notebook
    args = load_config_notebook()

    # Load Data
    print("\r Loading data")
    train_loader, _ , test_loader = load_train_test(args.batch_size)

    #Reweight training    
    print("\r reweighting data")
    #cls_num_list= list(get_counts_training_data(train_loader))
    cls_num_list= list([0.00020353383324465444, 0.0006849963955165558, 0.00010011808071699749])
    per_cls_weights = reweight(cls_num_list, beta=args.beta)

    print("\r Setting up Module")
    #https://github.gatech.edu/Sgudiduri3/DeepLearning2022/blob/main/dataM_focal_loss_20221121.ipynb
    net = DecomposableAttention(100, 200).to(device)

    #Optimization using focal loss
    criterion = FocalLoss(weight=per_cls_weights, gamma=args.gamma).to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Loop through epoch
    # Loop through dataset
    print("\r Training and Testing")
    best = 0.0
    best_cm = None
    best_model = None
    for epoch in range(args.epochs):  # loop over the dataset multiple times 
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, net, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, test_loader, net, criterion)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(net)

    print('\r Best Prec @1 Acccuracy: {:.4f}'.format(best))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')    
    print('\r Finished Training')

if __name__ == '__main__':
    main()
