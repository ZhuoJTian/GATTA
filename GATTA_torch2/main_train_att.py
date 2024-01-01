import argparse
import os
import copy
import torch
import numpy as np
from datasets import cifar10
from client import Client
from network_model.resnet18_att import RestNet18_att
from Sample_parti_noiid3 import getAttr
from tqdm.auto import tqdm
import torch.nn as nn

cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)

torch.cuda.set_device(2)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(cpu_num)

dataset_path = '/mnt/HD2/yyz/Datasets/CIFAR10'
sub_folder = "personal_model"

def create_clients(local_datasets_train, local_datasets_test, _lambda):
    clients = []
    for k in range(len(local_datasets_train)):
        client = Client(client_id=k, _lambda=_lambda, local_data_train=local_datasets_train[k],
                        local_data_test=local_datasets_test[k], device="cuda")
        clients.append(client)
    return clients


def set_up_clients(args):
    CCategoryToClients, LocalDist, LocaDis_test, SamplesToClients, SamplesToClients_test, MAX_NUM = getAttr()
    local_datasets_train, local_datasets_test= \
        cifar10.create_datasets(args.data_path, args.dataset,
                                args.num_clients, LocalDist, LocaDis_test)
    print("load the datasets")
    torch.manual_seed(1)
    Ini_model = RestNet18_att()
    optim_config = {'lr': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': args.decay,
                    'eps': 1e-8}
    clients = create_clients(local_datasets_train, local_datasets_test, args._lambda)
    for k, client in tqdm(enumerate(clients), leave=False):
        client.model = copy.deepcopy(Ini_model).cuda()
        # aa = torch.load("./verify/global_model/resnet_global.pt")
        # client.model.load_state_dict(aa, strict=True)
        client.setup(batch_size=args.batch_size,
                     criterion='nn.CrossEntropyLoss',
                     num_local_epochs=1,
                     optimizer='optim.RMSprop',
                     optim_config=optim_config)
    print("load the clients")
    return clients


def run_test(clients, num_clients, ep):
    train_acc = np.zeros(num_clients)
    train_loss = np.zeros(num_clients)
    val_acc = np.zeros(num_clients)
    val_loss = np.zeros(num_clients)
    for client_id in range(num_clients):
        t_loss, t_acc = clients[client_id].client_evaluate_train()
        train_acc[client_id] = t_acc
        train_loss[client_id] = t_loss
        v_loss, v_acc = clients[client_id].client_evaluate_vali()
        val_acc[client_id] = v_acc
        val_loss[client_id] = v_loss

    print("Localtest: [epoch {}, {} train_inst, {} test_inst] Training ACC: {:.4f}, Training_Loss: {:.4f}, "
          "Testing ACC: {:.4f}, Testing_Loss: {:.4f}"
          .format(ep + 1, 60000, 10000, np.average(train_acc), np.average(train_loss),
                  np.average(val_acc), np.average(val_loss)))
    return np.average(train_acc)


def Agg_model(clients, adj_matrix, Models):
    glob_list = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8', 'a_weig_out', 'a_bn1']
    running = ['running_mean', 'running_var', 'num_batches_tracked']
    weight_keys = list(Models[0].keys())
    list_new = [k for k in weight_keys \
                if any(glob in k for glob in glob_list) and \
                    all(run not in k for run in running)]

    for client_id in range(len(clients)):
        client = clients[client_id]
        model_dict = Models[client_id]
        client_vars_sum = copy.deepcopy(model_dict)
        neig_location = list(np.nonzero(adj_matrix[:, client_id])[0])
        num_neig = len(neig_location)
        # select output channel
        for k in list_new:
            temp = client_vars_sum[k]
            for i in neig_location:
                temp += Models[i][k]
            client_vars_sum[k] = torch.true_divide(temp, num_neig+1)
        client.model.load_state_dict(client_vars_sum)
    return clients


def ns_para_def(clients, para_ns, adj_matrix, dim_ns):
    for client_id in range(len(clients)):
        client = clients[client_id]
        neig_location = list(np.nonzero(adj_matrix[:, client_id])[0])
        num_neig = len(neig_location)
        ns_para_local = torch.zeros([1, 1, dim_ns])
        ns_para_neig = torch.zeros([1, num_neig, dim_ns])
        ns_para_local[0, 0, :] = copy.deepcopy(para_ns[client_id])
        for j in range(num_neig):
            ns_para_neig[0, j, :] = copy.deepcopy(para_ns[neig_location[j]])
        client.def_ns(ns_para_local, ns_para_neig)
    return clients


def collab_train(args, clients, adj_matrix):
    ep = -1
    Client_num=args.num_clients
    # initialize -- 
    dim_ns = 64*10+10
    shape_ns = [1, 1, dim_ns]
    vars_ns = torch.randn(shape_ns)
    for client_id in range(Client_num):
        client = clients[client_id]
        neig_location = list(np.nonzero(adj_matrix[:, client_id])[0])
        num_neig = len(neig_location)
        ns_para_local = torch.zeros([1, 1, dim_ns])
        ns_para_neig = torch.zeros([1, num_neig, dim_ns])
        ns_para_local = vars_ns
        ns_para_neig[0, :, :] = vars_ns.repeat(1, num_neig, 1)
        client.def_ns(ns_para_local, ns_para_neig)

    acc = run_test(clients, Client_num, ep)
    for ep in range(args.epochs):
        Models = [0]*Client_num
        para_ns = []
        for client_id in tqdm(range(Client_num), ascii=True):
            client = clients[client_id]
            client.client_update()
            f_weig, f_bias = client.get_ns_res()
            para_ns.append(torch.cat((f_weig, f_bias), 2).cpu())
            Models[client_id] = copy.deepcopy(client.model.state_dict())
        clients = Agg_model(clients, adj_matrix, Models)
        clients = ns_para_def(clients, para_ns, adj_matrix, dim_ns)
        acc = run_test(clients, Client_num, ep)
    print("aa")
    return clients


def main(args, adj_matrix):
    clients = set_up_clients(args)
    model_dict = clients[0].model.state_dict()
    # acc = run_test(clients, args.num_clients, ep=-2)
    clients_trained = collab_train(args, clients, adj_matrix)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=dataset_path,
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'femnist'], default='cifar10',
                        help='choose dataset from cifar10 or femnist')
    parser.add_argument('--model_path', type=str, default="./verify/"+sub_folder+"/save_model50/",
                        help='path to trained model')
    parser.add_argument('--fig_path', type=str, default="./verify/"+sub_folder+"/save_fig50/",
                        help='path to plotted figure')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num_clients', type=int, default=50,
                        help='number of workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='momentum (default: 0.0)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (L2) (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Learning rate step gamma (default: 0.2)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='saving model (default: True)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--zero_threshold', type=float, default=0.001,
                        help='threshold to define zero weight (default: 0.001)')
    parser.add_argument('--_lambda', type=float, default=0.08,
                        help='hyperparameter for regularization term (default: 0.001)')

    args = parser.parse_args()
    adj_matrix = np.loadtxt("adj_matrix.txt")
    main(args, adj_matrix)
