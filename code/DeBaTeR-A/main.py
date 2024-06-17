import torch
import numpy as np
import pickle
import scipy.sparse as sp
from model import Model
from torch.utils.data import DataLoader
import utils
import time
import argparse
import os

class config(object):
    d = 64
    nlayer = 2
    nepoch = 100
    batch_size = 2048
    lr = 1e-3
    wd = 1e-4
    cl = 0.2
    au = 1
    au_u = 0.7
    k = [10, 20, 50]
    beta = 0.35
    eps = 0.2
    train=True
    load_time = 'Sun-Feb-18-01-12-16-2024'

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, help='dataset name')
args = parser.parse_args()

dataset = args.d
if dataset == 'ml-100k':
    config.batch_size = 64
    config.lr = 1e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

adj_path = './dataset/{}/adj.pkl'.format(dataset)
time_path = './dataset/{}/train.pkl'.format(dataset)
time_min_path = './dataset/{}/time_min.pkl'.format(dataset)
time_max_path = './dataset/{}/time_max.pkl'.format(dataset)
test_path = './dataset/{}/test.pkl'.format(dataset)
test_time_path = './dataset/{}/test_time.pkl'.format(dataset)

A = pickle.load(open(adj_path, 'rb'))
T = pickle.load(open(time_path, 'rb'))
minT = pickle.load(open(time_min_path, 'rb'))
maxT = pickle.load(open(time_max_path, 'rb'))

n, m = A.shape
tdim = minT.shape[0]
tlen = (maxT - minT + 1).astype(np.int32).tolist()

model = Model(n, m, config.d, config.nlayer, tdim, tlen, device, config.eps).to(device)

test = pickle.load(open(test_path, 'rb'))
test_T = pickle.load(open(test_time_path, 'rb'))
test_T = torch.from_numpy((test_T-minT).astype(np.int64)).to(device)
test_id = torch.LongTensor(list(test.keys())).to(device)
torch.cuda.empty_cache()

A_keys = np.array(list(T.keys())).transpose()
A_u = torch.from_numpy(A_keys[0,:].astype(np.int64)).to(device)
A_i = torch.from_numpy(A_keys[1,:].astype(np.int64)).to(device)
T_tensor = torch.sparse_coo_tensor(A_keys, np.stack(list(T.values())-minT, axis=0)
                                .astype(np.int64)).coalesce()

test_batch_size = 2048
num_test_batch = int((test_id.shape[0] - 1)/test_batch_size + 1)

if config.train:

    train = utils.BiG(A.nonzero()[0].astype(np.int64), A.nonzero()[1].astype(np.int64), T, minT, m)
    dataloader = DataLoader(train, shuffle=True
                                , batch_size=config.batch_size
                                , num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_ndcg = 0.0
    print("Start Training!", flush=True)

    for epoch in range(config.nepoch):
        model.train()
        for i ,(u_id, pos_id, t, neg_id) in enumerate(dataloader):
            optimizer.zero_grad()
            model(A_u, A_i, T_tensor.values().to(device), config.beta)
            #torch.cuda.empty_cache()
            bpr, alignment, uniformity = model.get_loss(u_id, pos_id, neg_id, t.to(device))
            cl = config.cl * model.cl_loss(u_id, pos_id)
            loss =  bpr + cl + config.au * (alignment + config.au_u * uniformity)
            loss.backward()
            optimizer.step()
            if i%50 == 0:
                print('{} Epoch {}/{} Batch {}/{}: loss {} BPR {} CL {}'
                    .format(time.ctime(), epoch + 1, config.nepoch, i + 1, len(dataloader), loss.detach().cpu(), bpr.detach().cpu(), cl.detach().cpu()))
        model.eval()
        with torch.no_grad():
            ndcg, prec, recall = {}, {}, {}
            for k in config.k:
                ndcg[k], prec[k], recall[k] = 0.0, 0.0, 0.0
            for i in range(num_test_batch):
                st = i*test_batch_size
                end = min((i+1)*test_batch_size, test_id.shape[0])
                batch_id = test_id[st:end].cpu()
                p = model.predict(batch_id, test_T[batch_id, :])
                for i in range(batch_id.shape[0]):
                    p[i, A[batch_id[i]].nonzero()[1]] = -np.inf
                for k in config.k:
                    ilist = torch.argsort(p, dim=-1, descending=True)[:, :k].cpu()
                    ndcg[k] += utils.NDCG_k(ilist, test, batch_id, k)
                    prec_, recall_ = utils.precision_recall(ilist, test, batch_id, k)
                    prec[k] += prec_
                    recall[k] += recall_
            for k in config.k:
                ndcg[k] /= test_id.shape[0]
                prec[k] /= test_id.shape[0]
                recall[k] /= test_id.shape[0]
            if ndcg[config.k[0]] > best_ndcg:
                best_ndcg = ndcg[config.k[0]]
                if not os.path.exists('./model/{}'.format(dataset)):
                    os.makedirs('./model/{}'.format(dataset))
                print('Saving model in epoch {}, {}'.format(epoch + 1, time.ctime()))
                torch.save(model.state_dict(), './model/{}/{}.pt'.format(dataset, time.ctime()))
            print('Epoch {}/{} NDCG: {}  Prec: {}  Recall: {}'
                .format(epoch + 1, config.nepoch, ndcg, prec, recall))

    print('Saving model, ', time.ctime())
    config.load_time = time.ctime()
    torch.save(model.state_dict(), './model/{}/{}.pt'.format(dataset, config.load_time))
else:
    model.load_state_dict(torch.load('./model/{}/{}.pt'.format(dataset, config.load_time)))
    model.eval()

    with torch.no_grad():
        model(A_u, A_i, T_tensor.values().to(device), config.beta)
        ndcg, prec, recall = {}, {}, {}
        for k in config.k:
            ndcg[k], prec[k], recall[k] = 0.0, 0.0, 0.0
        for i in range(num_test_batch):
            st = i*test_batch_size
            end = min((i+1)*test_batch_size, test_id.shape[0])
            batch_id = test_id[st:end].cpu()
            p = model.predict(batch_id, test_T[batch_id, :])
            for i in range(batch_id.shape[0]):
                p[i, A[batch_id[i]].nonzero()[1]] = -np.inf
            for k in config.k:
                ilist = torch.argsort(p, dim=-1, descending=True)[:, :k].cpu()
                ndcg[k] += utils.NDCG_k(ilist, test, batch_id, k)
                prec_, recall_ = utils.precision_recall(ilist, test, batch_id, k)
                prec[k] += prec_
                recall[k] += recall_
        for k in config.k:
            ndcg[k] /= test_id.shape[0]
            prec[k] /= test_id.shape[0]
            recall[k] /= test_id.shape[0]
        print('NDCG: {}  Prec: {}  Recall: {}'
            .format(ndcg, prec, recall))

