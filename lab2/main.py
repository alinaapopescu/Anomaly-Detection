import random

from pyod.models.combination import average, maximization
from pyod.utils import standardizer
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF
from pyod.utils.data import generate_data_clusters
import pyod.models.knn
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score


def normal_distribution(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def function(x, mean, sd):
    a = random.randrange(0, 100)
    b = random.randrange(0, 100)
    return a * x + b + normal_distribution(x, mean, sd)

def ex1():
    mean = np.linspace(-10, 10, 20)
    sd = np.linspace(-10, 10, 20)

    x_vals = np.linspace(-10, 10, 20)

    y_vals = [function(x_val, mean_val, sd_val) for x_val in x_vals for mean_val in mean for sd_val in sd]

    X = []

    for idx, y in enumerate(y_vals):
        X.append([1, y])

    X = np.array(X)

    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    H = U @ U.T

    for idx, H_idx in enumerate(H):
        print(f'H_{idx} : {H_idx[idx]}')

def ex2():

    X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_clusters=2, contamination=0.1)

    ngh = [5, 10, 15, 20]
    for n in ngh:
        clf = KNN(n_neighbors=n)
        clf.fit(X_train)

        y_train_pred = clf.labels_
        y_train_scores = clf.decision_scores_
        y_test_pred = clf.predict(X_test)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        fig.suptitle(f'KNN (n_neighbors={n})')

        axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o')
        axs[0, 0].set_title('Ground Truth Labels (Train) ')

        axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm', marker='o')
        axs[0, 1].set_title('Predicted Labels (Train)')

        axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x')
        axs[1, 0].set_title('Ground Truth Labels (Test)')

        axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm', marker='x')
        axs[1, 1].set_title('Predicted Labels (Test)')


        plt.tight_layout()
        plt.show()

        train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        print(f'Balanced Accuracy (Train): {train_bal_acc}')
        print(f'Balanced Accuracy (Test): {test_bal_acc}')

def ex3():
    # Generate 2 clusters
    centers = [[-10, 10], [10, 10]]
    cluster_std = [2, 6]
    samples = [200, 100]

    X, y = make_blobs(n_samples=samples,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=42)

    n_neighbors = 5
    contamination_rate = 0.07
    # LOF
    clf = LOF(contamination=contamination_rate)
    clf.fit(X, y)
    lof_scores = clf.decision_scores_

    # KNN
    clf2 = pyod.models.knn.KNN(n_neighbors=n_neighbors, contamination=contamination_rate)
    clf2.fit(X, y)
    knn_scores = clf2.decision_scores_

    knn_threshold = np.percentile(knn_scores, 100 * (1 - contamination_rate))
    lof_threshold = np.percentile(lof_scores, 100 * (1 - contamination_rate))

    knn_outliers = (knn_scores > knn_threshold).astype(int)
    lof_outliers = (lof_scores > lof_threshold).astype(int)

    colors = np.array(['#377eb8', '#ff7f00'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X[:, 0], X[:, 1], c=colors[knn_outliers], edgecolor='k', s=50)
    ax[0].set_title(f'KNN (n_neighbors={n_neighbors})')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Plot for LOF
    ax[1].scatter(X[:, 0], X[:, 1], c=colors[lof_outliers], edgecolor='k', s=50)
    ax[1].set_title(f'LOF')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()


def ex4():

    # Load data
    data = scipy.io.loadmat('cardio.mat')
    X, y = data['X'], data['y'].flatten()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=42)

    # KNN and LOF models
    neighbors = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    train_scores, test_scores = [], []

    for n in neighbors:
        # KNN
        knn = KNN(n_neighbors=n)
        knn.fit(X_train)
        train_scores.append(knn.decision_scores_)  # Use decision scores for normalization
        test_scores.append(knn.decision_function(X_test))
        print(f'KNN Balanced Accuracy Test with n_neighbors= {n}: ', knn.decision_scores_)
        print(f'KNN Balanced Accuracy Train with n_neighbors= {n}: ', knn.decision_function(X_train))

        # LOF
        lof = LOF(n_neighbors=n)
        lof.fit(X_train)
        train_scores.append(lof.decision_scores_)  # Use decision scores for normalization
        test_scores.append(lof.decision_function(X_test))
        print(f'LOF Balanced Accuracy Test with n_neighbors= {n}: ', lof.decision_scores_)
        print(f'LOF Balanced Accuracy Train with n_neighbors= {n}: ',lof.decision_function(X))

        print('--------------------------------------------------------------------')

    train_scores_normalized = [standardizer(scores.reshape(-1, 1)) for scores in train_scores]
    test_scores_normalized = [standardizer(scores.reshape(-1, 1)) for scores in test_scores]

    train_avg_score = average(np.hstack(train_scores_normalized))
    test_avg_score = average(np.hstack(test_scores_normalized))
    train_max_score = maximization(np.hstack(train_scores_normalized))
    test_max_score = maximization(np.hstack(test_scores_normalized))

    contamination_rate = np.mean(y)
    avg_threshold = np.quantile(test_avg_score, 1 - contamination_rate)
    max_threshold = np.quantile(test_max_score, 1 - contamination_rate)

    y_test_avg_pred = (test_avg_score > avg_threshold).astype(int)
    y_test_max_pred = (test_max_score > max_threshold).astype(int)

    avg_ba = balanced_accuracy_score(y_test, y_test_avg_pred)
    max_ba = balanced_accuracy_score(y_test, y_test_max_pred)

    print(f'Balanced Accuracy (Average Strategy): {avg_ba}')
    print(f'Balanced Accuracy (Maximization Strategy): {max_ba}')


if __name__ == '__main__':
    # ex1()
    ex2()
    # ex3()
    # ex4()
