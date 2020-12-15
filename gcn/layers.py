import torch
import torch.nn as nn

class GCNConv(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCNConv, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=None)
        if act == 'PReLU':
            self.act = nn.PReLU()
        elif act == "ReLU":
            self.act = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
        for m in self.modules():#遍历整体网络结构以及每一层的结构
            self.weights_init(m)
        # print("*"*50)
        # for m in self.children():只遍历下一层的子节点
        #     print(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, x, adj):
        x = self.fc(x)
        out = torch.spmm(adj, x)
        if self.bias is not None:
            out +=self.bias
        return self.act(out)



