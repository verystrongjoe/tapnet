import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# load the data generated from MUSE
def load_muse_data(path="./data/muse/", dataset="ECG", tensor_format=True):

    def loadsparse(filename):
        df = pd.read_csv(filename, header=None, delimiter=",")
        a = np.array(df.as_matrix())
        a = sp.csr_matrix(a)
        return a

    def loaddata(filename):
        df = pd.read_csv(filename, header=None, delimiter=",")
        a = np.array(df.as_matrix())
        return a

    path = path + dataset + "/"
    file_header = dataset.lower() + "_"

    train_features = loadsparse(path + file_header + "train.csv")[:500]
    test_features = loadsparse(path + file_header + "test.csv")
    features = sp.vstack([train_features, test_features])

    train_labels = loaddata(path + file_header + "train_label.csv")
    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)

    nclass = np.amax(labels) + 1

    features = normalize(features)

    # total data size: 934
    train_size = train_features.shape[0]
    #train_size = 10
    total_size = features.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return features, labels, idx_train, idx_val, idx_test, nclass

def load_ts_data(path="./data/time_series/", tensor_format=True):

    def loadsparse(filename):
        df = pd.read_csv(filename, header=None, delimiter=",")
        a = np.array(df.as_matrix())
        a = sp.csr_matrix(a)
        return a

    def loaddata(filename):
        df = pd.read_csv(filename, header=None, delimiter=",")
        a = np.array(df.as_matrix())
        return a

    adj = loadsparse(path + "graph.csv")
    # features = loadsparse(path + "feature.csv")
    features = loadsparse(path + "all_features.csv")
    labels = loaddata(path + "labels.csv")

    # build symmetric adjacency matrix
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # total data size: 934
    train_size = 300
    idx_train = range(train_size)
    idx_val = range(train_size, 934)
    idx_test = range(train_size, 934)

    if tensor_format:
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    # rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)

    row_sums = mx.sum(axis=1)
    mx = mx / row_sums

    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)