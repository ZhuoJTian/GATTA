#######################################################################################################
# For the whole system, there are 10 clients in total, this file was generated for deciding local data
# distribution for each client (non i.i.d. distribution)
#######################################################################################################

import numpy as np
import random
import sys
import json
import os
from collections import defaultdict

# Define minimum and maximum samples for each client
NUM_OF_CLIENTS = 100
Users_in_client = 2
MIN = 100
MAX = 1000

# 决定每个用户的样本总数
def read_dir(data_dir):
    users = []
    num_samples = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        users.append(cdata['user_name'])
        if cdata['num_samples']==len(cdata['user_data']['x']):
            num_samples.append(cdata['num_samples'])
        else:
            print(cdata['user_name'])
            num_samples.append(len(cdata['user_data']['x']))
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return users, num_samples, clients, data
'''
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_path, 'femnist', 'data', '3500_client_json')
path_train = os.path.join(data_dir, 'train')
path_test = os.path.join(data_dir, 'test')
users_train, num_train, clients_tr, data_train = read_dir(path_train)
users_test, num_test, clients_te, data_test = read_dir(path_test)
assert users_train==users_test

Users=[]
for str in users_train:
    Users.append(int(str[1:5]))
print(len(Users))
np.savetxt("Users.txt", Users, fmt='%d')
'''
Users_array=np.loadtxt("Users.txt")
# print(Users_array.shape[0])
Users = [int(Users_array[i]) for i in range(Users_array.shape[0])]
# 决定每一个agent拥有的user类别数
'''
NumOfUsers= []
CandidateCate = [100]
for _ in range(NUM_OF_CLIENTS):
    NumOfUsers.append(100)
# print(NumOfUsers)
'''
NumOfUsers = [Users_in_client for i in range(NUM_OF_CLIENTS)]
'''
# 决定每个用户拥有的具体用户id
UsersToClients = [[] for _ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    Categories = np.random.choice(Users, NumOfUsers[client], replace=False)
    for item in Categories:
        UsersToClients[client].append(item)

np.savetxt("UsersToClients2.txt", UsersToClients, fmt='%d')
'''
UsersToClients_array=np.loadtxt("UsersToClients2.txt")
UsersToClients =[[int(UsersToClients_array[i, j])
                     for j in range(UsersToClients_array.shape[1])]
                    for i in range(UsersToClients_array.shape[0])]
'''
# 统计每一个user有多少agent有
Categories = [0 for i in range(len(Users))]
for client in range(NUM_OF_CLIENTS):
    for item in UsersToClients[client]:
        Categories[Users.index(item)] += 1
np.savetxt("Categories2.txt", Categories, fmt='%d')
'''
Categories_array = np.loadtxt("Categories2.txt")
# print(Categories_array.shape[0])
Categories = [int(Categories_array[i]) for i in range(Categories_array.shape[0])]
count = Categories.copy()
'''
LocalDist_test = [[0.0 for _ in range(Users_in_client)] for __ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    for cls in UsersToClients[client]:
        LocalDist_test[client][UsersToClients[client].index(cls)] = num_test[Users.index(cls)]

LocalDist_train = [[0.0 for _ in range(Users_in_client)] for __ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    for cls in UsersToClients[client]:
        count[Users.index(cls)] -= 1
        if count[Users.index(cls)] != 0:
            LocalDist_train[client][UsersToClients[client].index(cls)] = int(num_train[Users.index(cls)] / Categories[Users.index(cls)])*1.0
        else:
            LocalDist_train[client][UsersToClients[client].index(cls)] = 1.0 * (num_train[Users.index(cls)] - (Categories[Users.index(cls)] - 1)
                                                               * np.ceil(num_train[Users.index(cls)] / Categories[Users.index(cls)]))

LocalDist_test_array=np.array(LocalDist_test)
LocalDist_train_array=np.array(LocalDist_train)
np.savetxt("LocalDist_niid_test2.txt", LocalDist_test_array, fmt='%d')
np.savetxt("LocalDist_niid2.txt", LocalDist_train_array, fmt='%d')
'''
LocalDist_array=np.loadtxt("LocalDist_niid2.txt")
LocalDist_test_array=np.loadtxt("LocalDist_niid_test2.txt")

SamplesToClients=[0.0 for i in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    SamplesToClients[client] = sum(LocalDist_array[client, :])

SamplesToClients_test=[0.0 for i in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    SamplesToClients_test[client] = sum(LocalDist_test_array[client, :])

LocalDist=LocalDist_array.tolist()
LocalDist_test=LocalDist_test_array.tolist()

MAX_NUM = max(SamplesToClients)

def getAttr():
    """
    :return:
    CategoryToClients: 每个用户本地占有样本类别数
    LocalDist: 每个用户本地样本组成
    SamplesToClients: 每个用户本地样本数
    MAX_NUM: 最大用户本地样本数
    TimeVal：每个用户本地样本比例（以最大用户本地样本数归一化）
    """
    return Users, UsersToClients, LocalDist, LocalDist_test, SamplesToClients, SamplesToClients_test, MAX_NUM
