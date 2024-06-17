import os.path
from os import remove
from re import split


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, dtype):
        data = []
        if dtype == 'graph':
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])

        if dtype == 'sequential':
            training_data, test_data = [], []
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    user_id = items[0]
                    seq = items[1].strip().split()
                    training_data.append(seq[:-1])
                    test_data.append(seq[-1])
                data = (training_data, test_data)
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data

    def load_temporal_data(file):
        import pickle
        import numpy as np
        time_path = '{}/train.pkl'.format(file)
        time_min_path = '{}/time_min.pkl'.format(file)
        time_max_path = '{}/time_max.pkl'.format(file)
        test_time_path = '{}/test_time.pkl'.format(file)
        time_data = pickle.load(open(time_path, 'rb'))
        minT = pickle.load(open(time_min_path, 'rb'))
        maxT = pickle.load(open(time_max_path, 'rb'))
        test_time = pickle.load(open(test_time_path, 'rb'))
        tdim = minT.shape[0]
        tlen = (maxT - minT + 1).astype(np.int32).tolist()
        return time_data, tdim, tlen, minT, test_time
