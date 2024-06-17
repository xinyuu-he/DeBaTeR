import json
import numpy as np
import time
import scipy.sparse as sp
import pickle

path = './yelp/yelp_academic_dataset_review.json'
upath = './yelp/yelp_academic_dataset_user.json'
ipath = './yelp/yelp_academic_dataset_business.json'

split = 0.3
U = {}
I = {}
user_cnt = 0
item_cnt = 0
timestamps = {}

for line in open(upath, 'r', encoding='utf-8'):
    review = json.loads(line)
    user = review['user_id']
    count = review['review_count']
    if count >= 50:
        U[user] = user_cnt
        user_cnt += 1

for line in open(ipath, 'r', encoding='utf-8'):
    review = json.loads(line)
    item = review['business_id']
    count = review['review_count']
    if count >= 50:
        I[item] = item_cnt
        item_cnt += 1

print(len(U), len(I), flush=True)

for line in open(path, 'r', encoding='utf-8'):
    review = json.loads(line)
    user = review['user_id']
    item = review['business_id']
    if user in U and item in I and review['stars'] >= 4:
        u = U[user]
        i = I[item]
        timestamp = review['date']
        if u not in timestamps:
            timestamps[u] = {}
        timestamps[u][i] = timestamp

splitTime = {}
testTime = np.zeros((len(U), 6))
max_time = np.zeros((6,))-np.inf
min_time = np.zeros((6,))+np.inf
for u in timestamps:
    k = int(len(timestamps[u]) * (1-split))
    if (k+1) < len(timestamps[u]):
        stamps = list(timestamps[u].values())
        stamps.sort()
        timeStruct = time.strptime(stamps[k+1], "%Y-%m-%d %H:%M:%S")
        timestamp = np.array(timeStruct[:6])
        splitTime[u] = stamps[k+1]
        testTime[u, :] = timestamp
        for j in range(6):
            if timestamp[j] < min_time[j]: min_time[j] = timestamp[j]
            if timestamp[j] > max_time[j]: max_time[j] = timestamp[j]

print(min_time, max_time)

edges = ([], [])
train = {}
test = {}

for line in open(path, 'r', encoding='utf-8'):
    review = json.loads(line)
    user = review['user_id']
    item = review['business_id']
    if user in U and item in I and review['stars'] >= 4:
        u = U[user]
        i = I[item]
        timestamp = review['date']
        if (u not in splitTime) or (timestamp < splitTime[u]):
            timeStruct = time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = np.array(timeStruct[:6])
            for j in range(6):
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

# 189552 29255
# [2.005e+03 1.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00] [2022.   12.   31.   23.   59.   59.]
# 1055317
# 334385
