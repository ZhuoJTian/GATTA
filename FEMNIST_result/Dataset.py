import numpy as np
from tensorflow.keras.utils import to_categorical
from copy import deepcopy
import json
import os
from collections import defaultdict

def read_dir(data_dir, us):
    users = []
    num_samples = []
    data = defaultdict(lambda : None)

    if len(str(us))==1:
    	user_id = 'f000'+str(us)
    elif len(str(us))==2:
    	user_id = 'f00'+str(us)
    elif len(str(us))==3:
    	user_id = 'f0'+str(us)
    elif len(str(us))==4:
    	user_id = 'f'+str(us)
    files = os.listdir(data_dir)
    files = [f for f in files if f.startswith(user_id)]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        users.append(cdata['user_name'])
        if cdata['num_samples']!=len(cdata['user_data']['x']):
            print(us)
        num_samples.append(cdata['num_samples'])
        data.update(cdata['user_data'])
    if len(num_samples)>1:
        print("error")
        print(users)

    return users, num_samples, data


class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] + \
                    self.random_order[:overflow]
            self.start = overflow
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        # print(len(perm0))
        assert len(perm0) == batch_size

        return self.x[perm0], self.y[perm0]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, Users, UsersToClient, local_dist, local_dist_test, one_hot=True, split=0):
        parent_path1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        parent_path = os.path.join(parent_path1, 'FEMNIST_result')
        data_dir = os.path.join(parent_path, 'femnist', 'data', '3500_client_json')
        path_train = os.path.join(data_dir, 'train')
        path_test = os.path.join(data_dir, 'test')
        # split = number of clients
        self.train = self.splited_batch_train(Users, path_train, split, UsersToClient, local_dist)
        self.test = self.splited_batch_test(Users, path_test, split, UsersToClient, local_dist_test)


    def splited_batch_train(self, Users, path_train, count, UsersToClient, local_dist):
        res = []
        remained = [0 for _ in range(3596)]
        for i in range(count):
            x_data=[]
            y_data=[]
            for cls in range(2):
                us = UsersToClient[i][cls]
                # print(us)
                users, num_train, data_train = read_dir(path_train, us)
                # print(num_train)
                #index = []
                for num in range(int(remained[Users.index(us)]), int(remained[Users.index(us)] + local_dist[i][cls])):
                    #index.append(sample_index[cls][num])
                    if num>=len(data_train['x']):
                        print(num, len(data_train['x']), num_train[0], int(remained[Users.index(us)] + local_dist[i][cls]))
                        print(i, us)
                    x_data.append(data_train['x'][num])
                    y_data.append(to_categorical(data_train['y'][num], 62))
                remained[Users.index(us)] += local_dist[i][cls]
                # print("a")
            x_array=np.array(x_data)
            x_data=np.reshape(x_array, [x_array.shape[0], x_array.shape[1], 1])
            y_data=np.array(y_data)
            res.append(
                BatchGenerator(x_data,
                               y_data))
        return res

    def splited_batch_test(self, Users, path_test, count, UsersToClient, local_dist_test):
        res = []
        for i in range(count):
            x_data = []
            y_data = []
            for cls in range(2):
                us = UsersToClient[i][cls]
                users, num_test, data_test = read_dir(path_test, us)
                for num in range(0, int(local_dist_test[i][cls])):
                    x_data.append(data_test['x'][num])
                    y_data.append(to_categorical(data_test['y'][num], 62))
            x_array=np.array(x_data)
            x_data=np.reshape(x_array, [x_array.shape[0], x_array.shape[1], 1])
            y_data=np.array(y_data)
            res.append(
                BatchGenerator(x_data,
                               y_data))
        return res

