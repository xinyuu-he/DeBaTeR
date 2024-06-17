from torch.utils.data import Dataset
import random
import math
import numpy as np

class BiG(Dataset):
    def __init__(self, uidx, iidx, T, minT, m):
        super(BiG, self).__init__()
        assert len(uidx) == len(iidx)
        self.uidx = uidx
        self.iidx = iidx
        self.m = m
        self.T = T
        self.minT = minT

    def __getitem__(self, item):
        return self.uidx[item], self.iidx[item] \
        , (self.T[(self.uidx[item], self.iidx[item])]-self.minT).astype(np.int64), random.choice(range(self.m))

    def __len__(self):
        return len(self.uidx)

def NDCG_k(y, y_true, ids, k):
    assert y.shape[1] == k
    avg_ndcg = 0.0
    for i in range(y.shape[0]):
        u = int(ids[i])
        idcg = 0.0
        dcg = 0.0
        for t in range(y.shape[1]):
            if y[i,t] in y_true[u]:
                dcg += 1.0/math.log(t+2, 2)
        for t in range(min(k, len(y_true[u]))):
            idcg += 1.0/math.log(t+2, 2)
        avg_ndcg += dcg/idcg
    return avg_ndcg

def precision_recall(y, y_true, ids, k):
    hits = {}
    for i in range(y.shape[0]):
        u = int(ids[i])
        items = y_true[u]
        predicted = y[i,:].squeeze().tolist()
        hits[u] = len(set(items).intersection(set(predicted)))
    prec = sum([hits[u] for u in hits])
    prec = prec / k
    recall_list = [hits[u]/len(y_true[u]) for u in hits]
    recall = sum(recall_list)
    return prec, recall
