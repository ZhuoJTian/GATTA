from torch.utils.data import Dataset
import numpy as np
import torchvision
import torch


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def create_datasets(data_path, dataset_name, num_clients, local_dist, local_dist_test):
    """Split the whole dataset in non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()

        # prepare raw training & test datasets
        
        training_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=False,
            transform=transform
        ) #__dict__[dataset_name]
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=False,
            transform=transform
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]

    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()

    x_train = [item for item in training_dataset.data]
    x_test = [item for item in test_dataset.data]
    y_train = [[item] for item in training_dataset.targets]
    y_test = [[item] for item in test_dataset.targets]

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

    # split training dataset according to non-i.i.d. label distribution

    # get the indices of training dataset for each client
    idx_clients_train = [[] for i in range(num_clients)]
    remained = [0 for _ in range(10)]
    for i in range(num_clients):
        index = []
        for cls in range(10):
            if local_dist[i][cls] > 0:
                for num in range(int(remained[cls]), int(remained[cls] + local_dist[i][cls])):
                    index.append(int(sample_index_train[cls][num]))
                remained[cls] += local_dist[i][cls]
        idx_clients_train[i] = index

    # get the indices of testing dataset for each client
    idx_clients_test = [[] for i in range(num_clients)]
    remained = [0 for _ in range(10)]
    for i in range(nu