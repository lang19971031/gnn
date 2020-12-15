import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNConv

class GCN(nn.Module):
    def __init__(self, in_ft, h_ft, out_ft, activation, dropout):
        super(GCN, self).__init__()
        self.gcnconv1 = GCNConv(in_ft, h_ft, activation)
        self.gcnconv2 = GCNConv(h_ft, out_ft, activation)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = self.gcnconv1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcnconv2(x, adj)
        return F.log_softmax(x, dim=1)
