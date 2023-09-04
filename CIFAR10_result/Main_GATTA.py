import numpy as np
from tqdm import tqdm
from copy import deepcopy
from Client_GATTA import Clients_GATTA
from Sample_parti_noiid3 import getAttr
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

#####################################################################################
# simulate the GATNN1 in non-i.i.d distribution
#####################################################################################

def buildClients(num, adj_matrix, local_dist, local_dist_test, Me):
    # flag=0, D-SGD;  flag=1, ADMM
    learning_rate = 0.001
    num_input = 32  # image shape: 28*28
    num_input_channel = 3  # image channel: 1
    num_classes = 10  # MNIST total classes (0-9 digits)

    #create Client and model
    return Clients_GATTA(input_shape=[None, num_input, num_input, num_input_channel],
                   num_classes=num_classes,
                   learning_rate=learning_rate,
                   clients_num=num, local_dist=local_dist, local_dist_test=local_dist_test, me=Me)

##########################GATNN#######################################
def GATTA_run_local_test(clients, adjm, global_vars, para_ns, ep):
    train_acc = np.zeros(CLIENT_NUMBER)
    train_loss = np.zeros(CLIENT_NUMBER)
    val_acc = np.zeros(CLIENT_NUMBER)
    val_loss = np.zeros(CLIENT_NUMBER)

    for client_id in range(CLIENT_NUMBER):
        clients.set_global_vars(global_vars[client_id])
        dataset_train = clients.dataset.train[client_id]
        local_para = para_ns[client_id]
        neig_location = list(np.nonzero(adjm[:, client_id])[0])
        neig_para = np.concatenate([para_ns[i] for i in neig_location], axis=1)
        with clients.graph.as_default():
            batch_x, batch_y = dataset_train.next_batch(int(SamplesToClients[client_id]))
            feed_dict = {
                clients.model.X: batch_x,
                clients.model.Y: batch_y,
                clients.model.lp: local_para,
                clients.model.np: neig_para,
            }
            t_loss, t_acc = clients.sess.run([clients.model.loss_op, clients.model.accuracy], feed_dict=feed_dict)
        train_acc[client_id] = t_acc
        train_loss[client_id] = t_loss
        dataset_test = clients.dataset.test[client_id]
        with clients.graph.as_default():
            batch_x, batch_y = dataset_test.next_batch(int(SamplesToClients_test[client_id]))
            feed_dict = {
                clients.model.X: batch_x,
                clients.model.Y: batch_y,
                clients.model.lp: local_para,
                clients.model.np: neig_para,
            }
            v_loss, v_acc= clients.sess.run([clients.model.loss_op, clients.model.accuracy], feed_dict=feed_dict)
        val_acc[client_id] = v_acc
        val_loss[client_id] = v_loss

    print("Localtest: [epoch {}, {} train_inst, {} test_inst] Training ACC: {:.4f}, Training_Loss: {:.4f}, "
          "Testing ACC: {:.4f}, Testing_Loss: {:.4f}"
          .format(ep + 1, 60000, 10000, np.average(train_acc), np.average(train_loss),
                  np.average(val_acc), np.average(val_loss)))

    with open("./record/local_GATNN.txt", "a+") as f:
        f.write("[epoch {}, {} train_inst, {} test_inst] Training ACC: {:.4f}, Training_Loss: {:.4f}, "
                "Testing ACC: {:.4f}, Testing_Loss: {:.4f}\n"
                .format(ep + 1, 60000, 10000, np.average(train_acc), np.average(train_loss),
                        np.average(val_acc), np.average(val_loss)))
    return np.average(train_acc)


def GATTA_calculte_average(adj_matrix, client_num, Models, index_ag):
    """calculate the neighbor_average of the model agnostic layers"""
    new_glovars = [0] * client_num
    for client_id in range(client_num):
        client_vars_sum = deepcopy(Models[client_id])
        neig_location = np.nonzero(adj_matrix[:, client_id])[0]
        num_neig = len(neig_location)
        for neig in neig_location:
            for cv, cvv in zip([client_vars_sum[i] for i in index_ag], [Models[neig][j] for j in index_ag]):
                cv += cvv
        # print(cv.shape, num_neig)
        vars_new = []
        for i in range(len(client_vars_sum)):
            var = client_vars_sum[i]
            if i in index_ag:
                vars_new.append(1.0* var / (num_neig+1))
            else:
                vars_new.append(var)
        new_glovars[client_id] = vars_new
    return new_glovars


def GATTA(client):
    #### BEGIN TRAINING ####
    global_vars = [0] * CLIENT_NUMBER  #store all parameters in all clients
    para_agnos = [0] * CLIENT_NUMBER  #model agonostic paramters
    para_ns = [0] * CLIENT_NUMBER   #node-specific parameters
    vars = client.get_client_vars()
    vars_ag, index_ag = client.get_agnostic_vars()
    shape_ns = [1, 1, 192*CATEGORIES+CATEGORIES]
    vars_ns = client.initial_ns(shape_ns)

    ep=-1
    for client_id in range(CLIENT_NUMBER):
        global_vars[client_id] = vars.copy()
        para_agnos[client_id] = vars_ag.copy()
        para_ns[client_id] = vars_ns.copy()

    acc = GATTA_run_local_test(client, adj_matrix, global_vars, para_ns, ep)

    print(acc)
    for ep in range(round):
        Models = [0] * CLIENT_NUMBER
        para1_new = [0] * CLIENT_NUMBER
        para2_new = [0] * CLIENT_NUMBER
        for client_id in tqdm(range(CLIENT_NUMBER), ascii=True):
            neig_location = np.nonzero(adj_matrix[:, client_id])[0]
            local_old = para_ns[client_id]
            neig_old = np.concatenate([para_ns[i] for i in neig_location], axis=1)
            # Restore global vars to client's model
            client.set_global_vars(global_vars[client_id])

            # Train for th epochsb  本地训练几个epoch之后进行节点之间的参数传递
            for cnt in range(epoch_local):
                client.train_epoch(client_id, local_old, neig_old)
                if cnt != epoch_local - 1:
                    client.set_global_vars(client.get_client_vars())

            # Obtain current client's vars
            current_client_vars = client.get_client_vars()
            Models[client_id] = current_client_vars.copy()
            # 得到每个节点最新的node specific parameters
            para1_new[client_id],  para2_new[client_id],= client.get_ns_paras(local_old, neig_old)
            
            
        para_ns1 = deepcopy(para1_new)
        para_ns2 = deepcopy(para2_new)
        global_vars = GATTA_calculte_average(adj_matrix, CLIENT_NUMBER, Models, index_ag)
        acc = GATTA_run_local_test(client, adj_matrix, global_vars, para_ns, ep)
        ep += 1


###########################################################

CATEGORIES = 10
CLIENT_NUMBER = 100
epoch_total = 400
epoch_local = 1
round=int(epoch_total/epoch_local)
adj_matrix=np.loadtxt('adj_matrix.txt')

#### CREATE CLIENT AND LOAD DATASET ####
CCategoryToClients, LocalDist, LocaDis_test, SamplesToClients, SamplesToClients_test,MAX_NUM = getAttr()
client = buildClients(CLIENT_NUMBER, adj_matrix, local_dist=LocalDist, local_dist_test=LocaDis_test, Me=2)
GATTA(client)
