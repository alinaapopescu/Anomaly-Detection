import random
import scipy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score


def ex1():
    """
    Main function to run the entire OddBall-like pipeline.
    1) Build the graph (ex11)
    2) Compute node features (ex12)
    3) Compute anomaly scores (ex13)
    4) Visualize and highlight top 10 anomalies (ex14)
    5) Combine the old score with LOF to get a final score and visualize (ex15)
    """
    # 1. Build the graph from the first 1500 lines
    G = ex11()

    # 2. Compute N_i, E_i, W_i, lambda_w,i
    ex12(G)

    # 3. Fit power-law model on (E_i, N_i) and compute anomaly score
    ex13(G)

    # 4. Sort scores in descending order, draw the graph,
    #    highlight the top 10 anomalies
    ex14(G)

    # 5. Compute LOF, combine with old score, draw again
    ex15(G)


def ex11():
    """
    Create the graph by reading only the first 1500 lines
    of 'ca-AstroPh.txt'. Each occurrence of an edge
    increments its 'weight' by 1.
    """
    G = nx.Graph()
    max_lines = 1500  # as per requirement: use only the first 1500 lines

    with open('ca-AstroPh.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            nodes = line.strip().split()
            if len(nodes) < 2:
                continue

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
    Store these in the Graph as node attributes.
    """
    Ni_dict = {}
    Ei_dict = {}
    Wi_dict = {}
    lambda_wi_dict = {}

    for node in G.nodes():
        # 1) Ni
        neighbors = list(G.neighbors(node))
        Ni = len(neighbors)

        # Build the subgraph for the "egonet": node + its neighbors
        ego_nodes = [node] + neighbors
        H = G.subgraph(ego_nodes)

        # 2) Ei (number of edges in egonet)
        Ei = H.number_of_edges()

        # 3) Wi (sum of edge weights in egonet)
        Wi = sum(data['weight'] for _, _, data in H.edges(data=True))

        # 4) lambda_w,i (largest eigenvalue of weighted adjacency)
        if len(ego_nodes) == 1:
            # If the node has no neighbors, eigenvalue = 0
            lambda_wi = 0.0
        else:
            A = nx.to_numpy_array(H, nodelist=ego_nodes, weight='weight')
            w, _ = np.linalg.eig(A)  # w are eigenvalues
            lambda_wi = max(w.real)  # largest real part

        Ni_dict[node] = Ni
        Ei_dict[node] = Ei
        Wi_dict[node] = Wi
        lambda_wi_dict[node] = lambda_wi

    # Store them as node attributes
    nx.set_node_attributes(G, Ni_dict, name='Ni')
    nx.set_node_attributes(G, Ei_dict, name='Ei')
    nx.set_node_attributes(G, Wi_dict, name='Wi')
    nx.set_node_attributes(G, lambda_wi_dict, name='lambda_wi')


def ex13(G):
    """
       y = C * x^theta.
       The anomaly score will be:
         score_i = [ max(Ei, Ei_pred) / min(Ei, Ei_pred ) ] * log( |Ei - Ei_pred| + 1 )
         where Ei_pred = C * Ni^theta
    """
    # Prepare data for regression in log-scale:
    # We skip nodes where Ni = 0 or Ei = 0 to avoid log(0).
    valid_nodes = []
    logN = []
    logE = []

    for node in G.nodes():
        Ni = G.nodes[node]['Ni']
        Ei = G.nodes[node]['Ei']
        if Ni > 0 and Ei > 0:
            valid_nodes.append(node)
            logN.append(np.log(Ni))
            logE.append(np.log(Ei))

    # Fit a linear regression model: log(E) = a + b * log(N)
    logN = np.array(logN).reshape(-1, 1)
    logE = np.array(logE).reshape(-1, 1)

    if len(logN) == 0:
        # Edge case: no valid nodes to fit => all scores = 0
        for node in G.nodes():
            G.nodes[node]['anomaly_score'] = 0.0
        return

    model = LinearRegression()
    model.fit(logN, logE)

    a = model.intercept_[0]  # log(C) = a
    b = model.coef_[0][0]  # theta = b
    C = np.exp(a)
    theta = b

    # Now compute anomaly scores for ALL nodes (even those with Ni=0 or Ei=0)
    # If Ni=0 => predicted Ei_pred= C*(0^theta)=0 => handle carefully
    for node in G.nodes():
        Ei = G.nodes[node]['Ei']
        Ni = G.nodes[node]['Ni']

        Ei_pred = C * (Ni ** theta) if Ni > 0 else 0.0

        # The anomaly score:
        # score_i = (max(Ei, Ei_pred)/min(Ei, Ei_pred)) * log(|Ei - Ei_pred| + 1)
        # Handle min(Ei, Ei_pred)=0 => to avoid division by zero, interpret score as Ei + Ei_pred
        # or define a small epsilon. We will do a safe check:
        if Ei == 0 or Ei_pred == 0:
            # Score if either is zero
            ratio = Ei + Ei_pred  # or max(Ei, Ei_pred) if you want simpler approach
        else:
            ratio = max(Ei, Ei_pred) / min(Ei, Ei_pred)

        diff_part = np.log(abs(Ei - Ei_pred) + 1)

        anomaly_score = ratio * diff_part
        G.nodes[node]['anomaly_score'] = anomaly_score


def ex14(G):
    # Sort by anomaly_score (descending)
    # If 'anomaly_score' does not exist for some node, use 0
    node_scores = []
    for node in G.nodes():
        score = G.nodes[node].get('anomaly_score', 0.0)
        node_scores.append((node, score))

    node_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top 10
    top10 = set([n for n, _ in node_scores[:10]])

    # Prepare a color map: red for top 10, blue otherwise
    colors = []
    for node in G.nodes():
        if node in top10:
            colors.append('red')
        else:
            colors.append('blue')

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # deterministic layout
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, width=0.5)
    plt.title("Top 10 Anomalies (Score #1) in Red")
    plt.axis('off')
    plt.show()


def ex15(G):
    """
    5. Modify the anomaly score by adding the Local Outlier Factor (LOF)
       (for the pair {E_i, N_i}), normalized, to the old anomaly score. Then
       redraw the graph highlighting the new top 10 anomalies.
    """
    # Build array X = [[Ei, Ni], ...] for all nodes
    nodes_list = list(G.nodes())
    X = []
    for node in nodes_list:
        Ei = G.nodes[node]['Ei']
        Ni = G.nodes[node]['Ni']
        X.append([Ei, Ni])

    X = np.array(X)

    # Fit LOF (LocalOutlierFactor) - we want the "outlier score", so we use negative_outlier_factor_
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof.fit(X)  # The fit_predict approach can also be used, but we want the raw factor.

    # negative_outlier_factor_ is more negative for outliers.
    # Typically, anomaly score = -negative_outlier_factor_.
    # We'll invert sign so that bigger => more anomalous
    lof_scores = -lof.negative_outlier_factor_

    # Normalize both old_score and lof_score to [0,1], then sum
    old_scores = []
    for node in nodes_list:
        old_scores.append(G.nodes[node].get('anomaly_score', 0.0))

    old_scores = np.array(old_scores)

    # Avoid constant array problems
    def minmax_norm(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_old = minmax_norm(old_scores)
    norm_lof = minmax_norm(lof_scores)

    final_scores = norm_old + norm_lof

    # Store the final scores
    for i, node in enumerate(nodes_list):
        G.nodes[node]['final_score'] = final_scores[i]

    # Sort by final_score (descending) and highlight top 10
    scored_nodes = [(node, G.nodes[node]['final_score']) for node in nodes_list]
    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    top10_final = set([n for n, _ in scored_nodes[:10]])

    # Prepare color map: green for top 10, gray otherwise
    colors = []
    for node in G.nodes():
        if node in top10_final:
            colors.append('green')
        else:
            colors.append('gray')

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, width=0.5)
    plt.title("Top 10 Anomalies (Score #2 = Score #1 + LOF) in Green")
    plt.axis('off')
    plt.show()


def ex2():
    print("Running Exercise 2.1 ...")
    ex21()
    print("Running Exercise 2.2 ...")
    ex22()

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def make_connected(G, max_tries=2000):
    """
    Add random edges to ensure G is connected (useful after nx.union).
    Tries up to max_tries random edges between components.
    """
    tries = 0
    while not nx.is_connected(G) and tries < max_tries:
        comps = list(nx.connected_components(G))
        if len(comps) == 1:
            break
        # Take the first two separate components
        c1 = comps[0]
        c2 = comps[1]
        node1 = random.choice(list(c1))
        node2 = random.choice(list(c2))
        G.add_edge(node1, node2)
        tries += 1

def compute_egonet_features(G):
    """
    For each node, compute:
      - Ni = number of neighbors
      - Ei = edges in the egonet
      - Wi = sum of weights in the egonet (default=1 if no weight)
    Store them in G.nodes[node]['Ni'], ['Ei'], ['Wi'].
    """
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        Ni = len(neighbors)

        # Egonet subgraph
        ego_nodes = neighbors + [node]
        H = G.subgraph(ego_nodes)

        Ei = H.number_of_edges()
        # If no 'weight' attribute, assume weight=1
        Wi = 0
        for u, v, d in H.edges(data=True):
            Wi += d.get("weight", 1)

        G.nodes[node]['Ni'] = Ni
        G.nodes[node]['Ei'] = Ei
        G.nodes[node]['Wi'] = Wi

def powerlaw_anomaly_score(G, x_attr, y_attr, score_attr):
    """
    Fits log(y) = a + b*log(x) for nodes with x>0,y>0.
    Then anomaly score for each node:
        score = (max(y, y_pred)/min(y, y_pred)) * log(|y - y_pred| + 1)
    Stores in G.nodes[node][score_attr].
    """
    valid_nodes = []
    logX = []
    logY = []

    for n in G.nodes():
        x_val = G.nodes[n][x_attr]
        y_val = G.nodes[n][y_attr]
        if x_val > 0 and y_val > 0:
            valid_nodes.append(n)
            logX.append(np.log(x_val))
            logY.append(np.log(y_val))

    # If no valid points, set score=0
    if len(valid_nodes) == 0:
        for n in G.nodes():
            G.nodes[n][score_attr] = 0.0
        return

    # Fit linear regression in log scale
    X_np = np.array(logX).reshape(-1, 1)
    Y_np = np.array(logY).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(X_np, Y_np)

    a = reg.intercept_[0]  # log(C)
    b = reg.coef_[0][0]    # theta
    C = np.exp(a)

    # Compute anomaly scores
    for n in G.nodes():
        x_val = G.nodes[n][x_attr]
        y_val = G.nodes[n][y_attr]
        if x_val > 0:
            y_pred = C * (x_val ** b)
        else:
            y_pred = 0.0

        # ratio part
        if y_val == 0 or y_pred == 0:
            ratio = y_val + y_pred
        else:
            ratio = max(y_val, y_pred) / min(y_val, y_pred)

        diff_part = np.log(abs(y_val - y_pred) + 1)
        score = ratio * diff_part
        G.nodes[n][score_attr] = score

def highlight_top_k(G, score_attr, k, title, seed=42):
    """
    Sort by G.nodes[node][score_attr] descending, highlight top k in red, others in blue.
    Draw the graph with a spring layout.
    """
    node_score_pairs = [(n, G.nodes[n].get(score_attr, 0)) for n in G.nodes()]
    node_score_pairs.sort(key=lambda x: x[1], reverse=True)
    topk = {n for n, _ in node_score_pairs[:k]}

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=seed)
    colors = []
    for n in G.nodes():
        if n in topk:
            colors.append('red')
        else:
            colors.append('blue')
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors)
    nx.draw_networkx_edges(G, pos, width=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()


###############################################################################
# EX2.1
###############################################################################
def ex21():
    # 1) 3-regular graph, 100 nodes
    G1 = nx.random_regular_graph(d=3, n=100)
    # 2) connected caveman graph with 10 cliques of 20 nodes each
    G2 = nx.connected_caveman_graph(10, 20)

    # Merge
    M = nx.union(G1, G2, rename=('G1-', 'G2-'))

    # Add edges to make connected
    make_connected(M)

    # Compute Ni, Ei, Wi
    compute_egonet_features(M)

    # Fit power-law (N_i -> E_i)
    powerlaw_anomaly_score(M, x_attr='Ni', y_attr='Ei', score_attr='clique_score')

    # Highlight top 10
    highlight_top_k(M, 'clique_score', k=10,
                    title="Ex2.1 - Potential Clique Nodes (Top 10 in Red)")

###############################################################################
# EX2.2
###############################################################################
def ex22():
    # 1) Regular graph with degree=3, 100 nodes
    G1 = nx.random_regular_graph(d=3, n=100)
    # 2) Regular graph with degree=5, 100 nodes
    G2 = nx.random_regular_graph(d=5, n=100)

    # Merge
    M = nx.union(G1, G2, rename=('G1-', 'G2-'))

    # We want weighted edges = 1 for all
    for u,v in M.edges():
        M[u][v]['weight'] = 1

    # Pick 2 random nodes and add +10 in their egonets
    all_nodes = list(M.nodes())
    if len(all_nodes) >= 2:
        chosen = random.sample(all_nodes, 2)
        for node in chosen:
            neighbors = list(M.neighbors(node)) + [node]
            for u in neighbors:
                for w in neighbors:
                    if M.has_edge(u,w):
                        M[u][w]['weight'] += 10

    # Compute features
    compute_egonet_features(M)  # (Ei, Wi, Ni)

    # Power-law on (E_i -> W_i) for heavy vicinity detection
    powerlaw_anomaly_score(M, x_attr='Ei', y_attr='Wi', score_attr='heavy_score')

    # Highlight top 4
    highlight_top_k(M, 'heavy_score', k=4,
                    title="Ex2.2 - HeavyVicinity Nodes (Top 4 in Red)")


def ex3():
    # 1. Load data
    X, A, labels = load_acm_data('ACM.mat')

    # 2. Convert adjacency to edge_index (PyTorch Geometric)
    edge_index = from_scipy_sparse_matrix(A)[0]

    # 3. Instantiate GAE model
    in_dim = X.shape[1]
    model = GraphAutoencoder(in_dim=in_dim)

    # 4. Train
    train_gae(model, X, A, labels, edge_index, alpha=0.8, lr=0.004, epochs=50)


def load_acm_data(mat_file):
    data = scipy.io.loadmat(mat_file)
    # Keys might differ, adapt if needed
    # 'Attributes', 'Network', 'Label'
    X = data["Attributes"]     # possibly a sparse matrix
    A = data["Network"]        # adjacency, sparse
    labels = data["Label"]     # shape [N, 1] or [1, N], adapt as needed

    # Convert to torch
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X)
    labels = torch.LongTensor(labels).squeeze()  # e.g. shape [N]

    return X, A, labels


###############################################################################
# GAE Model Definitions
##############################################################################


class Encoder(nn.Module):
    """
    2-layer GCN encoder:
      input -> GCNConv -> 128 -> ReLU -> GCNConv -> 64 -> ReLU
    """
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return x  # shape [num_nodes, 64]


class AttributeDecoder(nn.Module):
    """
    Reverse the process:
      64 -> GCNConv -> 128 -> ReLU -> GCNConv -> in_dim
    """
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, out_dim)

    def forward(self, z, edge_index):
        x_hat = self.conv1(z, edge_index)
        x_hat = torch.relu(x_hat)
        x_hat = self.conv2(x_hat, edge_index)
        # no activation on last
        return x_hat


class StructureDecoder(nn.Module):
    """
    1 GCNConv(64->64), ReLU, then A_hat = Z Z^T
    """
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(64, 64)

    def forward(self, z, edge_index):
        z = self.conv(z, edge_index)
        z = torch.relu(z)
        A_hat = torch.matmul(z, z.t())
        return A_hat


class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = Encoder(in_dim)
        self.attr_decoder = AttributeDecoder(out_dim=in_dim)
        self.struct_decoder = StructureDecoder()

    def forward(self, x, edge_index):
        # encode
        z = self.encoder(x, edge_index)
        # decode attributes
        x_hat = self.attr_decoder(z, edge_index)
        # decode structure
        A_hat = self.struct_decoder(z, edge_index)
        return x_hat, A_hat, z


###############################################################################
# Training
###############################################################################
def train_gae(model, X, A, labels, edge_index, alpha=0.8, lr=0.004, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # If A is large, watch memory usage. Convert to dense or keep it sparse.
    A_dense = torch.FloatTensor(A.toarray())

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        x_hat, A_hat, z = model(X, edge_index)

        # L = alpha * ||X - X_hat||^2_F + (1-alpha)*||A - A_hat||^2_F
        loss_attr = (X - x_hat).pow(2).sum()       # Frobenius norm squared
        loss_struct = (A_dense - A_hat).pow(2).sum()
        loss = alpha*loss_attr + (1-alpha)*loss_struct

        loss.backward()
        optimizer.step()

        # Evaluate reconstruction-based anomaly or classification if labels are known
        # For demonstration, let's do a simple reconstruction error-based AUC if labels are {0,1}.
        if epoch % 5 == 0:
            # Example "node anomaly score" = attribute reconstruction error
            node_error = (X - x_hat).pow(2).sum(dim=1).detach().cpu().numpy()
            # Suppose label=0 => anomaly, label=1 => normal (just as an example)
            y_true = (labels == 0).int().numpy()  # or adapt for your data
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, node_error)
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")
            else:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | (No binary labels, skipping AUC)")
        else:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")



if __name__ == '__main__':
    # ex1()
    # ex2()
    ex3()