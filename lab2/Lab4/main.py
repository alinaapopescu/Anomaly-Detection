from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, make_scorer
from matplotlib import pyplot as plt
from pyod.models.deep_svdd import DeepSVDD
import scipy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM


def plot_data(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred):
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(
        x_train[y_train == 0, 0], x_train[y_train == 0, 1], x_train[y_train == 0, 2],
        c='blue', label='Inliers (GT)', alpha=0.7
    )
    ax1.scatter(
        x_train[y_train == 1, 0], x_train[y_train == 1, 1], x_train[y_train == 1, 2],
        c='red', label='Outliers (GT)', alpha=0.7
    )
    ax1.set_title("Ground Truth: Training Data")
    ax1.legend()


    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(
        x_test[y_test == 0, 0], x_test[y_test == 0, 1], x_test[y_test == 0, 2],
        c='blue', label='Inliers (GT)', alpha=0.7
    )
    ax2.scatter(
        x_test[y_test == 1, 0], x_test[y_test == 1, 1], x_test[y_test == 1, 2],
        c='red', label='Outliers (GT)', alpha=0.7
    )
    ax2.set_title("Ground Truth: Test Data")
    ax2.legend()

    # Predicted labels: training data
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(
        x_train[y_train_pred == 0, 0], x_train[y_train_pred == 0, 1], x_train[y_train_pred == 0, 2],
        c='green', label='Inliers (Predicted)', alpha=0.7
    )
    ax3.scatter(
        x_train[y_train_pred == 1, 0], x_train[y_train_pred == 1, 1], x_train[y_train_pred == 1, 2],
        c='orange', label='Outliers (Predicted)', alpha=0.7
    )
    ax3.set_title("Predicted Labels: Training Data")
    ax3.legend()

    # Predicted labels: test data
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(
        x_test[y_test_pred == 0, 0], x_test[y_test_pred == 0, 1], x_test[y_test_pred == 0, 2],
        c='green', label='Inliers (Predicted)', alpha=0.7
    )
    ax4.scatter(
        x_test[y_test_pred == 1, 0], x_test[y_test_pred == 1, 1], x_test[y_test_pred == 1, 2],
        c='orange', label='Outliers (Predicted)', alpha=0.7
    )
    ax4.set_title("Predicted Labels: Test Data")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def deep_svdd_impl(x_train, x_test,y_train, y_test, features):

    clf_name = 'deep_svdd'
    clf = DeepSVDD(n_features=features)
    clf.fit(x_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    y_test_scores = clf.decision_function(x_test)

    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_scores)

    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"ROC AUC: {roc_auc}")

    # c
    plot_data(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred)



def osvm_impl(x_train, x_test,y_train, y_test, contamination):

    clf_name = 'OSVM'
    clf = OCSVM(kernel='linear', contamination=contamination)
    clf.fit(x_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    y_test_scores = clf.decision_function(x_test)

    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_scores)

    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"ROC AUC: {roc_auc}")

    # c
    plot_data(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred)

# Ex 1
def ex1():
    # a
    n_train = 300
    n_test = 200
    contamination = 0.15
    features = 3
    x_train, x_test,y_train, y_test= generate_data(n_train=n_train,
                                                    n_test=n_test,
                                                    contamination=contamination,
                                                    n_features= features)

    # b
    osvm_impl(x_train, x_test,y_train, y_test, contamination)

    # d
    deep_svdd_impl(x_train, x_test,y_train, y_test, features)


def gridSearchImpl(estimator, X_train, X_test, y_train, y_test, scoring, param_grid):

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    # Convert One-Class SVM output to binary classification
    y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    if scoring == 'accuracy':
        print("Accuracy Score:", accuracy_score(y_test, y_pred_binary))
    elif scoring == 'balanced_accuracy':
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)
        print("Balanced Accuracy on Test Set:", balanced_accuracy)

def ex2():
    n_test = 0.6
    data = scipy.io.loadmat('cardio.mat')
    X, y = data['X'], data['y'].flatten()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= n_test, shuffle=True, random_state=42)
    print(X.shape, y.shape)

    estimator = OCSVM()
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'nu': [0.1, 0.2, 0.5, 0.7, 0.9]
    }

    # b
    gridSearchImpl(estimator, X_train, X_test, y_train, y_test, 'accuracy', param_grid)

    # c
    gridSearchImpl(estimator, X_train, X_test, y_train, y_test, 'balanced_accuracy', param_grid)

    # d
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', OneClassSVM())
    ])

    param_grid2 = {
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Corrected
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Corrected
        'svm__nu': [0.1, 0.2, 0.5, 0.7, 0.9]  # Corrected
    }

    gridSearchImpl(pipeline, X_train, X_test, y_train, y_test, 'balanced_accuracy', param_grid2)

    # e f

    y_sklearn = 2 * y - 1  # Convert pyod to sklearn format

    # Split into training and testing sets again
    X_train, X_test, y_train, y_test = train_test_split(X, y_sklearn, test_size=n_test, shuffle=True, random_state=42)
    print("Data shapes:", X.shape, y.shape)

    scorer = make_scorer(balanced_accuracy_score)


    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid2, cv=5, scoring=scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    y_pred = best_model.predict(X_test)

    # Sklearn: 1 for inliers, -1 for outliers
    # PyOD: 0 for inliers, 1 for outliers
    y_pred_pyod = (-1 * y_pred + 1) // 2
    y_test_pyod = (-1 * y_test + 1) // 2

    balanced_accuracy = balanced_accuracy_score(y_test_pyod, y_pred_pyod)
    print("Balanced Accuracy on Test Set:", balanced_accuracy)

    print("Classification Report:")
    print(classification_report(y_test_pyod, y_pred_pyod))


def ocsvm_ex3(x_train, x_test, y_train, y_test):

    clf_name = 'OSVM'
    clf = OCSVM(kernel='linear')
    clf.fit(x_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    y_test_scores = clf.decision_function(x_test)

    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_scores)

    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"ROC AUC: {roc_auc}")

def deep_svv_impl_params(x_train, x_test, y_train, y_test, features,params):
    ba = []
    roc= []
    clf_name = 'deep_svdd'
    for param in params:
        clf = DeepSVDD(n_features=features, hidden_neurons=param)
        clf.fit(x_train)

        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        y_test_scores = clf.decision_function(x_test)

        balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_scores)

        ba.append(balanced_acc)
        roc.append(roc_auc)

    for i in range(len(params)):
        print(f"Hidden neurons {params[i]}")
        print(f"Balanced Accuracy: {ba[i]}")
        print(f"ROC AUC: {roc[i]}")
        print("--------------------------------------")


def ex3():
    # a
    n_test = 0.5
    data = scipy.io.loadmat('shuttle.mat')
    X, y = data['X'], data['y'].flatten()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(X_scaled.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=n_test, shuffle=True, random_state=42)

    # b
    # OCSVM
    ocsvm_ex3(X_train, X_test, y_train, y_test)

    # deepSVDD
    deep_svdd_impl(X_train, X_test, y_train, y_test, X_scaled.shape[1])

    # c
    params =[[9,7], [63, 64], [9,5]]
    deep_svv_impl_params(X_train, X_test, y_train, y_test, X_scaled.shape[1], params)



if __name__ == '__main__':
    ex1()
    ex2()
    ex3()