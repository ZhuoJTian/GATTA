import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def gener_net_erdos(N, p):
    rg=nx.erdos_renyi_graph(N, p)
    ps=nx.shell_layout(rg)
    nx.draw(rg, ps, with_labels=True, node_size=200)
    # plt.savefig('./test1.png')
    adj_matrix=nx.adjacency_matrix(rg)
    return adj_matrix.todense()


def gener_net_regular(d, N):
    rg=nx.random_regular_graph(d, N)
    ps=nx.shell_layout(rg)
    nx.draw(rg, ps, with_labels=True, node_size=200)
    # plt.savefig('./test1.png')
    adj_matrix=nx.adjacency_matrix(rg)
    return adj_matrix.todense()
'''
N=50
p=0.2
adj_matrix=gener_net_erdos(N, p)
# print(adj_matrix)
adj=np.array(adj_matrix)
for i in range(N):
    if np.any(np.sum(adj[i, :])==0):
    	print("False")

np.savetxt('adj_matrixN50P2.txt', adj, fmt="%d")

graph = nx.from_numpy_matrix(adj_matrix)
nx.draw(graph,pos = nx.shell_layout(graph), node_color = 'y',edge_color = 'k',with_labels = True, width=0.5, font_size=15, node_size =350)
plt.show()
'''
