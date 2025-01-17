import pickle
import time
import numpy as np
import scipy.sparse as sp
import random

with open('./u.data', 'r') as f:
    next(f)
    ratings = f.readlines()

split = 0.3
uid = {}
iid = {}
edges = ([], [])
train = {}
test = {}
ucnt = 0
icnt = 0
max_time = np.zeros((4,))-np.inf
min_time = np.zeros((4,))+np.inf

timestamps = {}
for rate in ratings:
    line = rate[:-1].split('\t')
    u = line[0]
    i = line[1]
    timestamp = line[3][:-1]
    if u not in uid:
        uid[u] = ucnt
        ucnt += 1
    if i not in iid:
        iid[i] = icnt
        icnt += 1
    if u not in timestamps:
        timestamps[u] = {}
    timestamps[u][i] = int(timestamp)

splitTime = {}
testTime = np.zeros((len(uid), 4))
for u in timestamps:
    k = int(len(timestamps[u]) * (1-split))
    stamps = list(timestamps[u].values())
    stamps.sort()
    timeStruct = time.gmtime(int(stamps[k+1]))
    timestamp = np.array(timeStruct[2:6])
    splitTime[u] = stamps[k]
    testTime[uid[u], :] = timestamp

for rate in ratings:
    line = rate[:-1].split('\t')
    u = line[0]
    i = line[1]
    timestamp = line[3][:-1]
    if int(timestamp) < splitTime[u]:
        timeStruct = time.gmtime(int(timestamp))
        timestamp = np.array(timeStruct[2:6])
        for j in range(4):
            if timestamp[j] < min_time[j]: min_time[j] = timestamp[j]
            if timestamp[j] > max_time[j]: max_time[j] = timestamp[j]
        edges[0].append(uid[u])
        edges[1].append(iid[i])
        train[(uid[u], iid[i])] = timestamp
    else:
        if uid[u] not in test:
            test[uid[u]] = []
        test[uid[u]].append(iid[i])

adj = sp.csr_matrix((np.ones(len(edges[0])), edges))

print(ucnt, icnt, len(edges[0]))
pickle.dump(adj, open('./adj.pkl', 'wb'))
pickle.dump(train, open('./train.pkl', 'wb'))
pickle.dump(min_time, open('./time_min.pkl', 'wb'))
pickle.dump(max_time, open('./time_max.pkl', 'wb'))
pickle.dump(test, open('./test.pkl', 'wb'))
pickle.dump(testTime, open('./test_time.pkl', 'wb'))
