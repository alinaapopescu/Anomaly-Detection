import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from pyod.models.pca import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models




def ex1():
    mean = [5, 10, 2]
    cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
    size = 500

    data = np.random.multivariate_normal(mean, cov, size)
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])

    # 3D scatter of generated data
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], c='blue', marker='o', s=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Scatter Plot of Generated Data')
    plt.show()

    # Center the data
    centered_data = df - df.mean()

    # Covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("Covariance Matrix:\n", cov_matrix)
    print("\nEigenvalues:\n", eigenvalues)
    print("\nEigenvectors:\n", eigenvectors)

    # b) Plot explained variance and cumulative explained variance
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # reorder eigenvectors correspondingly

    explained_variances = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variances)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(eigenvalues) + 1), explained_variances, label='Individual Variance')
    plt.step(range(1, len(eigenvalues) + 1), cumulative_explained_variance, where='mid', color='red',
             label='Cumulative Explained Variance')

    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance and Cumulative Explained Variance')
    plt.xticks(ticks=range(1, len(eigenvalues) + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Project data onto the principal components
    proj_data = np.dot(centered_data, eigenvectors)  # shape: (500,3)

    # c) Identify outliers based on the 3rd principal component
    # Use contamination rate = 0.1 and np.quantile
    contamination_rate = 0.1

    # For the 3rd principal component (index 2)
    pc3 = proj_data[:, 2]
    pc3_mean = pc3.mean()
    deviations_pc3 = np.abs(pc3 - pc3_mean)

    # Threshold at the 90th percentile (since contamination_rate=0.1)
    threshold_pc3 = np.quantile(deviations_pc3, 1 - contamination_rate)
    outliers_pc3 = (deviations_pc3 > threshold_pc3).astype(int)  # 1 for outlier, 0 for inlier

    # Plot with a different color for anomalies
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors_pc3 = np.where(outliers_pc3 == 1, 'red', 'blue')
    ax.scatter(df['X'], df['Y'], df['Z'], c=colors_pc3, marker='o', s=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Outliers based on 3rd Principal Component')
    plt.show()

    # Repeat the same steps for the 2nd principal component (index 1)
    pc2 = proj_data[:, 1]
    pc2_mean = pc2.mean()
    deviations_pc2 = np.abs(pc2 - pc2_mean)
    threshold_pc2 = np.quantile(deviations_pc2, 1 - contamination_rate)
    outliers_pc2 = (deviations_pc2 > threshold_pc2).astype(int)

    # Plot with a different color for anomalies (based on 2nd PC)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors_pc2 = np.where(outliers_pc2 == 1, 'green', 'blue')
    ax.scatter(df['X'], df['Y'], df['Z'], c=colors_pc2, marker='o', s=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Outliers based on 2nd Principal Component')
    plt.show()

    mean_pcs = proj_data.mean(axis=0)
    std_pcs = proj_data.std(axis=0)

    # 2. Compute the normalized distance: sum of squares of ((proj_data - mean)/std)
    normalized_data = (proj_data - mean_pcs) / std_pcs
    dist = np.sqrt(np.sum(normalized_data**2, axis=1))

    # 3. Use contamination rate and quantile to find threshold
    threshold_dist = np.quantile(dist, 1 - contamination_rate)
    outliers_dist = (dist > threshold_dist).astype(int)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors_dist = np.where(outliers_dist == 1, 'magenta', 'blue')
    ax.scatter(df['X'], df['Y'], df['Z'], c=colors_dist, marker='o', s=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Outliers based on Normalized Distance in PCA space')
    plt.show()


def ex2():
    # a
    n_test = 0.4
    contamination = 0.02
    data = scipy.io.loadmat('shuttle.mat')
    X, y = data['X'], data['y'].flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42)

    pca = PCA(contamination=contamination, random_state=42)
    pca.fit(X_train)

    scores_pca = pca.decision_function(X_test)

    plt.figure(figsize=(12, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_pca)
    plt.colorbar(label='Anomaly Score')
    plt.title('PCA Anomaly Scores - Exercise 2.1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.show()

    # b
    y_train_pred_pca = pca.predict(X_train)
    y_test_pred_pca = pca.predict(X_test)

    balanced_accuracy_train_pca = balanced_accuracy_score(y_train, y_train_pred_pca)
    balanced_accuracy_test_pca = balanced_accuracy_score(y_test, y_test_pred_pca)

    print("Exercise 2.2:")
    print(f"Balanced Accuracy (Train): {balanced_accuracy_train_pca:.4f}")
    print(f"Balanced Accuracy (Test): {balanced_accuracy_test_pca:.4f}\n")



def ex3():
    # a
    n_test = 0.5
    data = scipy.io.loadmat('shuttle.mat')
    X, y = data['X'], data['y'].flatten()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=n_test, shuffle=True, random_state=42)
    print("Data shapes:", X.shape, y.shape)
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # b - Define layer structure
    layer_1 = [8, 5, 3]   # Encoder layers
    layer_2 = [5, 8, 9]   # Decoder layers (last layer has sigmoid activation)

    class Autoencoder(tf.keras.Model):
        def __init__(self, encoder_layers, decoder_layers, input_dim):
            super(Autoencoder, self).__init__()
            # Build encoder
            enc = []
            enc.append(layers.InputLayer(input_shape=(input_dim,)))
            for units in encoder_layers:
                enc.append(layers.Dense(units, activation='relu'))
            self.encoder = tf.keras.Sequential(enc)

            # Build decoder
            dec = []
            for i, units in enumerate(decoder_layers):
                if i == len(decoder_layers)-1:
                    # Last layer with sigmoid activation
                    dec.append(layers.Dense(units, activation='sigmoid'))
                else:
                    dec.append(layers.Dense(units, activation='relu'))
            self.decoder = tf.keras.Sequential(dec)

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(layer_1, layer_2, input_dim)

    # 3. Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(X_train, X_train,
                              epochs=100,
                              batch_size=1024,
                              validation_data=(X_test, X_test),
                              verbose=1)

    # Plot training and validation loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 4. Compute reconstruction errors and threshold
    # Reconstruction on train data
    X_train_pred = autoencoder.predict(X_train)
    train_errors = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

    # Compute contamination rate
    contamination_rate = np.mean(y)  # fraction of anomalies in the dataset

    # threshold based on quantile
    threshold = np.quantile(train_errors, 1 - contamination_rate)
    print("Contamination rate:", contamination_rate)
    print("Threshold:", threshold)

    # Classify train data
    y_train_pred = (train_errors > threshold).astype(int)

    # Balanced accuracy for training set
    bal_acc_train = balanced_accuracy_score(y_train, y_train_pred)

    # Classify test data
    X_test_pred = autoencoder.predict(X_test)
    test_errors = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
    y_test_pred = (test_errors > threshold).astype(int)

    # Balanced accuracy for test set
    bal_acc_test = balanced_accuracy_score(y_test, y_test_pred)

    print("Balanced Accuracy on Training set:", bal_acc_train)
    print("Balanced Accuracy on Test set:", bal_acc_test)




def ex4():
    # Load and preprocess dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add a channel dimension since we need (H,W,1) for Conv2D
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # add noise using tf.random.normal with std factor=0.35,
    # values remain in [0, 1].
    noise_factor = 0.35
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
    x_test_noisy = tf.clip_by_value(x_test_noisy, 0.0, 1.0)

    # Convert x_test_noisy to numpy if needed
    x_test_noisy = x_test_noisy.numpy()

    class ConvAutoencoder(tf.keras.Model):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()

            # Encoder:
            #   - 1st Conv2D: 8 filters, kernel_size=3x3, relu, strides=2, padding='same'
            #   - 2nd Conv2D: 4 filters, kernel_size=3x3, relu, strides=2, padding='same'

            self.encoder = tf.keras.Sequential([
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
                layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
            ])

            # Decoder:
            #   - 1 Conv2DTranspose with same params as last layer of encoder (4 filters, 3x3, relu, strides=2, padding='same')
            #   - 1 Conv2DTranspose with same params as the first layer of encoder (8 filters, 3x3, relu, strides=2, padding='same')
            #   - 1 Conv2D layer with 1 filter and sigmoid activation to reconstruct image

            self.decoder = tf.keras.Sequential([
                layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
                layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
                layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = ConvAutoencoder()


    # Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(x_train, x_train,
                              epochs=10,
                              batch_size=64,
                              validation_data=(x_test, x_test))


    x_train_pred = autoencoder.predict(x_train)
    train_mse = np.mean(np.power(x_train - x_train_pred, 2), axis=(1, 2, 3))

    mean_mse = np.mean(train_mse)
    std_mse = np.std(train_mse)
    threshold = mean_mse + std_mse

    print("Threshold:", threshold)

    # For original test images
    x_test_pred = autoencoder.predict(x_test)
    test_mse = np.mean(np.power(x_test - x_test_pred, 2), axis=(1, 2, 3))
    test_pred_labels = (test_mse > threshold).astype(int)  # If error > threshold => anomaly (1), else normal (0)
    test_true_labels = np.zeros_like(test_pred_labels)  # Original test images are considered normal (0)

    test_accuracy_original = np.mean(test_pred_labels == test_true_labels)
    print("Accuracy on original test images:", test_accuracy_original)

    # For noisy test images
    x_test_noisy_pred = autoencoder.predict(x_test_noisy)
    test_noisy_mse = np.mean(np.power(x_test_noisy - x_test_noisy_pred, 2), axis=(1, 2, 3))
    test_noisy_pred_labels = (test_noisy_mse > threshold).astype(int)
    test_noisy_true_labels = np.ones_like(test_noisy_pred_labels)  # Noisy images considered anomalies (1)

    test_accuracy_noisy = np.mean(test_noisy_pred_labels == test_noisy_true_labels)
    print("Accuracy on noisy test images:", test_accuracy_noisy)

    # Plot 5 images from the test set
    n = 5
    plt.figure(figsize=(10, 8))

    # Original images
    for i in range(n):
        plt.subplot(4, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Original", fontsize=14)

    # Noisy images
    for i in range(n):
        plt.subplot(4, n, n + i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Noisy", fontsize=14)

    # Reconstructed from original
    for i in range(n):
        plt.subplot(4, n, 2 * n + i + 1)
        plt.imshow(x_test_pred[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Reconstructed from Original", fontsize=14)

    # Reconstructed from noisy
    for i in range(n):
        plt.subplot(4, n, 3 * n + i + 1)
        plt.imshow(x_test_noisy_pred[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Reconstructed from Noisy", fontsize=14)

    plt.tight_layout()
    plt.show()

    # Modify to become a Denoising Autoencoder (train noisy image)

    # Create a new model instance
    denoising_autoencoder = ConvAutoencoder()
    denoising_autoencoder.compile(optimizer='adam', loss='mse')

    # Train with noisy inputs and original (clean) as targets
    denoising_history = denoising_autoencoder.fit(x_train + noise_factor * tf.random.normal(shape=x_train.shape),
                                                  x_train,
                                                  epochs=10,
                                                  batch_size=64,
                                                  validation_data=(x_test_noisy, x_test))

    # Predict again using the denoising autoencoder
    x_test_denoised_pred = denoising_autoencoder.predict(x_test)
    x_test_noisy_denoised_pred = denoising_autoencoder.predict(x_test_noisy)

    # Plot the same figure again with denoised results
    plt.figure(figsize=(10, 8))

    for i in range(n):
        plt.subplot(4, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Original", fontsize=14)

    for i in range(n):
        plt.subplot(4, n, n + i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Noisy", fontsize=14)

    for i in range(n):
        plt.subplot(4, n, 2 * n + i + 1)
        plt.imshow(x_test_denoised_pred[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Denoised from Original (DAE)", fontsize=14)

    for i in range(n):
        plt.subplot(4, n, 3 * n + i + 1)
        plt.imshow(x_test_noisy_denoised_pred[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Denoised from Noisy (DAE)", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ex1()
    # ex2()
    # ex3()
    # ex4()