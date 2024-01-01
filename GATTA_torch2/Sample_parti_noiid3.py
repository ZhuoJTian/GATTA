#######################################################################################################
# For the whole system, there are 10 clients in total, this file was generated for deciding local data
# distribution for each client (non i.i.d. distribution)
#######################################################################################################

import numpy as np
import random
import sys
import json

# Define minimum and maximum samples for each client
NUM_OF_CLIENTS = 50
'''
RemainedSamples_train = [5000 for i in range(10)]
RemainedSamples_test = [1000 for i in range(10)]

# 决定每一个用户拥有的样本类别数
NumOfCategories = [3 for i in range(NUM_OF_CLIENTS)]

# 决定每个用户拥有的具体类别
CategoryToClients = [[] for _ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    Categories = np.random.choice(10, NumOfCategories[client], replace=False)
    for item in Categories:
        CategoryToClients[client].append(item)

# print(CategoryToClients)

np.savetxt("50_CategoryToClients3.txt", CategoryToClients, fmt='%d')
'''
CategoryToClients_array=np.loadtxt("50_CategoryToClients3.txt")
CategoryToClients =[[int(CategoryToClients_array[i, j])
                     for j in range(CategoryToClients_array.shape[1])]
                    for i in range(CategoryToClients_array.shape[0])]

# 统计每一种类别有多少用户有
'''
Categories = [0 for i in range(10)]
for client in range(NUM_OF_CLIENTS):
    for item in CategoryToClients[client]:
        Categories[item] += 1
print(Categories)


count = Categories.copy()
LocalDist_test = [[0.0 for _ in range(10)] for __ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    for cls in CategoryToClients[client]:
        LocalDist_test[client][cls] = RemainedSamples_test[cls]

LocalDist_train = [[0.0 for _ in range(10)] for __ in range(NUM_OF_CLIENTS)]
for client in range(NUM_OF_CLIENTS):
    for cls in CategoryToClients[client]:
        count[cls] -= 1
        if count[cls] != 0:
            LocalDist_train[client][cls] = int(RemainedSamples_train[cls] / Categories[cls])*1.0
        else:
            LocalDist_train[client][cls] = 1.0 * (RemainedSamples_train[cls] - (Categories[cls] - 1) * int(RemainedSamples_train[cls] / Categories[cls]))

LocalDist_test_array=np.array(LocalDist_test)
LocalDist_train_array=np.array(LocalDist_train)
np.savetxt("50_LocalDist_niid_test3.txt", LocalDist_test_array, fmt='%d')
np.savetxt("50_LocalDist_niid3.txt", LocalDist_train_array, fmt='%d')
'''
LocalDist_array=np.loadtxt("50_LocalDist_niid3.txt")
LocalDist_test_array=np.loadtxt("50_LocalDist_niid_test3.txt")

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
    return CategoryToClients, LocalDist, LocalDist_test, SamplesToClients, SamplesToClients_test, MAX_NUM
