import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.pca import PCA


def ex1():
    samples = 500
    features = 2

    X, _ = make_blobs(n_samples=samples, n_features=features, random_state=42)

    # Generate 5 vectors
    mean = [0, 0]
    vectors = np.random.multivariate_normal(mean, np.eye(2), 5)

    # Normalize the vectors
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    # Generate projection
    projections = np.dot(X, unit_vectors.T)

    # Histograms and bin edge
    histograms = []
    bin_edges = []
    for i in range(5):
        min_proj, max_proj = projections[:, i].min(), projections[:, i].max()
        hist, bins = np.histogram(projections[:, i], bins=10, range=(min_proj - 1, max_proj + 1), density=True)
        histograms.append(hist)
        bin_edges.append(bins)

    # Compute probabilities for each histogram
    probabilities = [hist / hist.sum() for hist in histograms]

    # Scores
    anomaly_scores = []
    for i in range(samples):
        sample_scores = []
        for j in range(5):
            idx = np.clip(np.digitize(projections[i, j], bin_edges[j]) - 1, 0, len(bin_edges[j]) - 2)
            sample_scores.append(probabilities[j][idx])
        anomaly_scores.append(np.mean(sample_scores))

    # Generate a test dataset
    X_test = np.random.uniform(low=-3, high=3, size=(samples, features))

    # Project test data and compute its anomaly scores
    test_projections = np.dot(X_test, unit_vectors.T)
    test_anomaly_scores = []
    for i in range(samples):
        test_sample_scores = []
        for j in range(5):
            idx = np.clip(np.digitize(test_projections[i, j], bin_edges[j]) - 1, 0, len(bin_edges[j]) - 2)
            test_sample_scores.append(probabilities[j][idx])
        test_anomaly_scores.append(np.mean(test_sample_scores))

    # Plotting
    plt.scatter(X_test[:, 0], X_test[:, 1], c=test_anomaly_scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.title('Test Dataset Anomaly Scores')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def ex2():
    #1
    centers = [[10, 0], [0, 10]]
    cluster_std = 1
    samples = 1000
    contamination = 0.02
    X, _ = make_blobs(n_samples=samples,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=42)

    #2
    iforest = IForest(contamination=contamination, random_state=42)
    iforest.fit(X)

    X_test = np.random.uniform(-10, 20, (1000, 2))

    #3
    scores_iforest = iforest.decision_function(X_test)

    #4
    loda = LODA(contamination=contamination, n_bins=10)
    loda.fit(X)
    scores_loda = loda.decision_function(X_test)

    #5
    pca = PCA(contamination=contamination, random_state=42)
    pca.fit(X)
    scores_pca = pca.decision_function(X_test)

    #6
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_iforest, cmap='viridis')
    plt.colorbar()
    plt.title('Isolation Forest Anomaly Scores')

    plt.subplot(132)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_loda, cmap='viridis')
    plt.colorbar()
    plt.title('LODA Anomaly Scores')

    plt.subplot(133)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_pca, cmap='viridis')
    plt.colorbar()
    plt.title('PCA Anomaly Scores')

    plt.show()



if __name__ == '__main__':
    # ex1()
    ex2()