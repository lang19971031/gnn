import torch.nn
import networkx as nx
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import os
import time

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_path = os.path.join('data/', dataset_str)
    for name in names:
        with open(data_path+'/raw/ind.{}.{}'.format(dataset_str, name), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    # print(objects[0])
    # for ob in objects:
    #     print(type(ob))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = np.fromfile(data_path+'/raw/ind.{}.test.index'.format(dataset_str), dtype=int, count=-1, sep='\n')
    # print(test_idx.shape)
    # print(allx.shape)
    # print(tx.shape)
    features = sp.vstack((allx, tx))
    features[test_idx, :] = features[np.sort(test_idx), :]
    features = normalize_features(features)
    # print(features.shape)
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[np.sort(test_idx), :]
    train_idx = np.arange(np.size(labels, 0))
    val_idx = train_idx[:500]
    train_idx = train_idx[500: -1000]
    # test_idx = np.sort(test_idx)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # b = time.time()    
    adj = adj + sp.eye(adj.shape[0])
    normalize_adj(adj)
    # print('time:{}'.format(time.time() - b))
    # print(type(adj))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    return adj, features, labels, train_idx, val_idx, test_idx 

def normalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print(type(d_mat_inv_sqrt))
    # print(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))
    # print("*"*50)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # print(type(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)))

def normalize_features(features):
    rowsum = np.array(features.sum(axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    # print('shape-->', type(sparse_mx.shape))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
