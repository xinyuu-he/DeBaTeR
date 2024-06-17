import pickle
import random
import scipy.sparse as sp
import numpy as np

dataset = 'ml-1m'

adj = pickle.load(open('./{}/adj.pkl'.format(dataset), 'rb'))
train = pickle.load(open('./{}/train.pkl'.format(dataset), 'rb'))

train_p = train.copy()
(U, I) = adj.nonzero()
U = U.tolist()
I = I.tolist()
i_len = max(I)+1

for (u,i) in train.keys():
    if random.random() < 0.2:
        i_ = random.randint(0, adj.shape[1]-1)
        t = train[(u, i)]
        train_p[(u,i_)] = t
        U.append(u)
        I.append(i_)

adj_p = sp.csr_matrix((np.ones(len(U)), [U,I]))

pickle.dump(adj_p, open('./{}-p/adj.pkl'.format(dataset), 'wb'))
pickle.dump(train_p, open('./{}-p/train.pkl'.format(dataset), 'wb'))
