import json
import numpy as np
import time
import scipy.sparse as sp
import pickle

path = './Movies_and_TV_5.json'

split = 0.3
U = {}
I = {}
user_cnt = 0
item_cnt = 0
timestamps = {}

for line in open(path, 'r', encoding='utf-8'):
    review = json.loads(line)
    user = review['reviewerID']
    item = review['asin']
    if review['overall'] >= 4.5:
        if user not in U:
            U[user] = user_cnt
            user_cnt += 1
        if item not in I:
            I[item] = item_cnt
            item_cnt += 1
        u = U[user]
        i = I[item]
        timestamp = review['unixReviewTime']
        if u not in timestamps:
            timestamps[u] = {}
        timestamps[u][i] = timestamp

print(len(U), len(I), flush=True)

splitTime = {}
testTime = np.zeros((len(U), 3))
for u in timestamps:
    k = int(len(timestamps[u]) * (1 - split))
    if (k + 1) < len(timestamps[u]):
        stamps = list(timestamps[u].values())
        stamps.sort()
        timeStruct = time.gmtime(stamps[k+1])
        timestamp = np.array(timeStruct[:3])
        splitTime[u] = stamps[k+1]
        testTime[u, :] = timestamp

max_time = np.zeros((3,)) - np.inf
min_time = np.zeros((3,)) + np.inf
edges = ([], [])
train = {}
test = {}

for line in open(path, 'r', encoding='utf-8'):
    review = json.loads(line)
    user = review['reviewerID']
    item = review['asin']
    if review['overall'] >= 4.5:
        u = U[user]
        i = I[item]
        timestamp = review['unixReviewTime']
        if (u not in splitTime) or (timestamp < splitTime[u]) or ((timestamp == splitTime[u]) and (np.random.rand() < (1 - split))):
            timeStruct = time.gmtime(timestamp)
            timestamp = np.array(timeStruct[:3])
            for j in range(3):
                if timestamp[j] < min_time[j]: min_time[j] = timestamp[j]
                if timestamp[j] > max_time[j]: max_time[j] = timestamp[j]
            edges[0].append(u)
            edges[1].append(i)
            train[(u, i)] = timestamp
        else:
            if u not in test:
                test[u] = []
            test[u].append(i)

adj = sp.csr_matrix((np.ones(len(edges[0])), edges))

print(len(edges[0]))
pickle.dump(adj, open('./adj.pkl', 'wb'))
pickle.dump(train, open('./train.pkl', 'wb'))
pickle.dump(min_time, open('./time_min.pkl', 'wb'))
pickle.dump(max_time, open('./time_max.pkl', 'wb'))
pickle.dump(test, open('./test.pkl', 'wb'))
pickle.dump(testTime, open('./test_time.pkl', 'wb'))
test_cnt = 0
for u in test: test_cnt += len(test[u])
print(test_cnt)

# 282524 58998
# 1682426
# 346365
