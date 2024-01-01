import gc
import pickle
import logging
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, _lambda, local_data_train, local_data_test, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self._lambda = _lambda
        self.data_train = local_data_train
        self.data_test = local_data_test
        self.device = device
        self.__model = None
        self.ns_para_local = 0
        self.ns_para_neig = 0
        self.i_out = {}
        self.j_out = {}

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader_train = DataLoader(self.data_train, batch_size=client_config["batch_size"], shuffle=True)
        self.dataloader_test = DataLoader(self.data_test, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        key_feature = []
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                key_feature.append(n)
        self.key_feature_weight = [k+'.weight' for k in key_feature]

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.dataloader_train:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data, self.ns_para_local.to(self.device), self.ns_para_neig.to(self.device))
                loss = eval(self.criterion)(reduction='mean')(outputs, labels)
                loss.backward()
                optimizer.step() 

                if self.device == "cuda": torch.cuda.empty_cache()               
        self.model.to("cpu")

    def client_evaluate_vali(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader_test:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data, self.ns_para_local.to(self.device), self.ns_para_neig.to(self.device))
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader_test)
        test_accuracy = correct / len(self.data_test)
        return test_loss, test_accuracy

    def client_evaluate_train(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader_train:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data, self.ns_para_local.to(self.device), self.ns_para_neig.to(self.device))
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader_train)
        test_accuracy = correct / len(self.data_train)
        return test_loss, test_accuracy
    
    def client_savemodel(self, path):
        torch.save(self.model.state_dict(), path+str(self.id)+"_alexnet.pt")

    def def_ns(self, ns_para_local, ns_para_neig):
        self.ns_para_local = ns_para_local
        self.ns_para_neig = ns_para_neig

    def get_ns_res(self):
        return self.model.fc_weig, self.model.fc_bias