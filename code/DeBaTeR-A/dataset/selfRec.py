import pickle

dataset='ml-1m-p'

with open('{}/adj.pkl'.format(dataset, dataset), 'rb') as f:
    train = pickle.load(f)

i, j = train.nonzero()

with open('../../SELFRec/dataset/{}/train.txt'.format(dataset), 'w') as f:
    for k in range(len(i)):
        f.write('{} {} 1\n'.format(i[k], j[k]))

with open('{}/test.pkl'.format(dataset, dataset), 'rb') as f:
    test = pickle.load(f)

with open('../../SELFRec/dataset/{}/test.txt'.format(dataset), 'w') as f:
    for u in test:
        for i in test[u]:
            f.write('{} {} 1\n'.format(u, i))
