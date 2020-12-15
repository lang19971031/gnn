import torch
import torch.nn.functional as F
import numpy as np
from utils import load_data, sparse_mx_to_torch_sparse_tensor, accuracy
import argparse
from gcn import GCN
import torch.optim as optim
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='name of dataset')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--activation', default='ReLU', help='activation function')
parser.add_argument('--dropout', type=float, default=0.5, help='Droput rate')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay(L2 loss)')
args = parser.parse_args()

def train(epoch):
    b = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    loss_val = F.nll_loss(output[val_idx], labels[val_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    print('epoch[{}]: loss_train:{:.4f}, acc_train:{:.2f}, loss_val:{:.4f}, acc_val:{:.2f}, time:{:.4f}'.format(
        epoch, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), time.time()-b 
    ))

def test():
    b = time.time()
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    print('accuracy:{:.2f}, time:{:.4f}'.format(
        acc_test.item(), time.time()-b 
    ))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

adj, features, labels, train_idx, val_idx, test_idx = load_data(args.dataset)


model = GCN(features.shape[1], args.hidden, labels.shape[1], args.activation, args.dropout)
_, labels =torch.max(labels, 1)

model = model.to(device)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.epochs):
    train(epoch)
    
test()