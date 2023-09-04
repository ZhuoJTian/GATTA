import numpy as np
from tensorflow.keras.utils import to_categorical
from copy import deepcopy

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
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            self.start = overflow
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        #print(len(perm0))
        assert len(perm0) == batch_size

        return self.x[perm0], self.y[perm0]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, load_data_func, local_dist, local_dist_test, one_hot=True, split=0):
        (x_train, y_train), (x_test, y_test) = load_data_func()
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        x_train = [item for item in x_train]
        x_test = [item for item in x_test]
        y_train = [[item] for item in y_train]
        y_test = [[item] for item in y_test]
        
        sample_index_train = [[] for _ in range(10)]
        for category in range(10):
            for index in range(len(y_train)):
                if y_train[index][0] == category:
                    sample_index_train[category].append(index)

        sample_index_test = [[] for _ in range(10)]
        for category in range(10):
            for index in range(len(y_test)):
                if y_test[index][0] == category:
                    sample_index_test[category].append(index)

        if one_hot:
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        x_train = np.array(x_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        x_train /= 255.0  # minmax_normalization
        x_test /= 255.0
        y_train = np.reshape(y_train, [-1, 10])
        y_test = np.reshape(y_test, [-1, 10])

        x_train_eval = x_train.copy()
        y_train_eval = y_train.copy()
        self.train_eval = BatchGenerator(x_train_eval, y_train_eval)

        # split = number of clients
        self.train = self.splited_batch_train(x_train, y_train, split, sample_index_train, local_dist)
        self.test = self.splited_batch_test(x_test, y_test, split, sample_index_test, local_dist_test)
        self.test_eval = BatchGenerator(x_test, y_test)


    def splited_batch_train(self, x_data, y_data, count, sample_index, local_dist):
        res = []
        remained = [0 for _ in range(10)]
        for i in range(count):
            index = []
            for cls in range(10):
                if local_dist[i][cls] > 0:
                    for num in range(int(remained[cls]), int(remained[cls]+local_dist[i][cls])):
                        index.append(sample_index[cls][num])
                    remained[cls] += local_dist[i][cls]

            res.append(
                BatchGenerator(x_data[index],
                               y_data[index]))
        return res


    def splited_batch_test(self, x_data, y_data, count, sample_index_test, local_dist_test):
        res = []
        for i in range(count):
            index = []
            for cls in range(10):
                if local_dist_test[i][cls] > 0:
                    for num in range(0, int(local_dist_test[i][cls])):
                        index.append(sample_index_test[cls][num])

            res.append(
                BatchGenerator(x_data[index],
                               y_data[index]))
        return res
