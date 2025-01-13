import networkx as nx
import numpy as np
import scipy

def ex1():
    G = ex11()
    ex12(G)
    for node in list(G.nodes())[:5]:
        print(f"Node {node}:")
        print(f"  Ni        = {G.nodes[node]['Ni']}")
        print(f"  Ei        = {G.nodes[node]['Ei']}")
        print(f"  Wi        = {G.nodes[node]['Wi']}")
        print(f"  lambda_wi = {G.nodes[node]['lambda_wi']}")
        print("")


    # # 3. Print
    # for node in list(G.nodes())[:10]:
    #     print(f"Node {node} -> Features {node_features[node]}")

# Create OddBall algorithm
def ex11():
    G = nx.Graph()
    with open('ca-AstroPh.txt', 'r') as f:
        for line in f:
            nodes = line.strip().split()
            node1, node2 = nodes[0], nodes[1]

            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += 1
            else:
                G.add_edge(node1, node2, weight=1)
    return G


def ex12(G):
    """
    For each node in G, compute and store the 4 requested features:
      1) Ni: number of neighbors
      2) Ei: number of edges in egonet i
      3) Wi: total weight of egonet i
      4) lambda_w,i: principal eigenvalue of the weighted adjacency matrix of egonet i
    """

    # Dictionaries for each feature
    Ni_dict = {}
    Ei_dict = {}
    Wi_dict = {}
    lambda_wi_dict = {}

    for node in G.nodes():
        # 1) Ni
        neighbors = list(G.neighbors(node))
        Ni = len(neighbors)

        ego_nodes = [node] + neighbors
        H = G.subgraph(ego_nodes)

        # 2) Ei
        Ei = H.number_of_edges()

        # 3) Wi
        Wi = sum(data['weight'] for _, _, data in H.edges(data=True))

        # 4) lambda_w,i
        if len(ego_nodes) == 1:
            lambda_wi = 0.0
        else:
            A = nx.to_numpy_array(H, nodelist=ego_nodes, weight='weight')
            w, _ = np.linalg.eig(A)
            lambda_wi = max(w.real)

        # Populate dictionaries
        Ni_dict[node] = Ni
        Ei_dict[node] = Ei
        Wi_dict[node] = Wi
        lambda_wi_dict[node] = lambda_wi

    # Now store these in the graph as node attributes
    nx.set_node_attributes(G, Ni_dict, name='Ni')
    nx.set_node_attributes(G, Ei_dict, name='Ei')
    nx.set_node_attributes(G, Wi_dict, name='Wi')
    nx.set_node_attributes(G, lambda_wi_dict, name='lambda_wi')

def ex13():
    return

def ex14():
    return

def ex15():
    return

def ex2():
    # ex21()
    ex22()

def ex21():
    G1 = nx.random_regular_graph(3, 100)
    print(G1)
    G2 = nx.connected_caveman_graph(10, 20)
    print(G2)

    G3 = nx.union(G1, G2, rename=("G1", "G2"))
    print(G3)

def ex22():
    G1 = nx.random_regular_graph(3, 100)
    G2 = nx.random_regular_graph(5, 100)
    G3 = nx.union(G1, G2, rename=("G1", "G2"))
    print(G3)


# Graph Autoencoder (GAE)
def ex3():
    ex32()
    return

def ex32():
    # Read data from file
    data = scipy.io.loadmat('ACM.mat')
    print(data)

def ex33():
    # Design graph
    return

def ex34():
    return

def ex35():
    return


if __name__ == '__main__':
    # ex1()
    # ex2()
    ex3()