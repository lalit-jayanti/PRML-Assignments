import numpy as np
import matplotlib.pyplot as plt
import time


def ridge_regression(X_train, y_train,
                     lam=0, iterations=1000,
                     learning_rate=0.01):
    """Ridge Regression

    (X - data as column vectors)
    Working:
        - Obtain solution to the ridge regression
        problem by using Gradient Descent
        - Returns solution for a particular
        lambda
    """

    wr = np.ones(X_train.shape[0])*1e-3
    itr = iterations
    eta = learning_rate

    xxt = (X_train@(X_train.T))
    xy = X_train@y_train

    for i in range(itr):
        gradf = 2*((xxt@wr) - (xy)) + 2*lam*wr
        wr = wr - eta*gradf/np.linalg.norm(gradf)

    return wr


def cross_validate_ridge_regression(X_train, y_train, split=0.2,
                                    k_fold=False,
                                    lams=np.logspace(0, 5, 10),
                                    iterations=1000,
                                    learning_rate=0.01,
                                    batches=1,
                                    seed=0):
    """Cross Validation

    (X - data as column vectors)
    Working:
        - Find validation errors for given set of lambda
        - Apply ridge regression
    
    Methods:
        - K-Fold: False
            - split data randomly and run ridge regression
            for every given lambda for given batches
            - Find Validation error on remaining data
            for each batch

        - K-Fold: True
            - Use K-Fold validation
            - Split data into k parts and run
            ridge regression k-1 parts and use 
            remaining part as validation set
            - Do this k times
        
    """

    np.random.seed(seed=seed)
    rng = np.random.default_rng()

    dims, num_points = X_train.shape
    val_size = int(num_points*split)
    if k_fold:
        batches = int(1/split)

    val_errors = []

    for batch in range(batches):

        # print(f"BATCH {batch}")

        val_idx = np.zeros(X_train.shape[1], dtype=bool)
        train_idx = np.zeros(X_train.shape[1], dtype=bool)
        choices = None

        if k_fold:
            choices = range(batch*val_size,
                            (batch+1)*val_size)
        else:
            choices = rng.choice(X_train.shape[1],
                                 size=val_size,
                                 replace=False)

        val_idx[choices] = True
        train_idx = ~val_idx

        X_t = X_train[:, train_idx]
        y_t = y_train[train_idx]
        X_val = X_train[:, val_idx]
        y_val = y_train[val_idx]

        errors = []
        for i, lam in enumerate(lams):
            # print(f"LAM {i}")

            wr = ridge_regression(X_train=X_t,
                                  y_train=y_t,
                                  lam=lam,
                                  iterations=iterations,
                                  learning_rate=learning_rate)

            errors.append([lam,
                           np.linalg.norm(((X_val.T)@wr) - y_val)**2])

        val_errors.append(errors)

    return np.array(val_errors)


if __name__ == '__main__':

    data_file = "../data/A2Q2Data_train.csv"
    data = np.loadtxt(data_file, delimiter=',')

    X = data[:, 0:100].T
    y = data[:, 100]

    # direct
    w = np.matmul(np.linalg.inv(X@(X.T)), X@y)

    test_file = "../data/A2Q2Data_test.csv"
    test = np.loadtxt(test_file, delimiter=',')

    Xt = test[:, 0:100].T
    yt = test[:, 100]

    wr = ridge_regression(X_train=X,
                          y_train=y,
                          lam=8.69749 * 1e3,
                          iterations=1000,
                          learning_rate=0.01)

    val_errors = cross_validate_ridge_regression(X_train=X,
                                                 y_train=y,
                                                 split=0.2,
                                                 k_fold=False,
                                                 lams=np.logspace(-5, 5, 10),
                                                 iterations=1000,
                                                 learning_rate=0.01,
                                                 batches=1,
                                                 seed=0)

    val_errors = np.array(val_errors)

    best_lam_idx = [np.argmin(val_errors[i, :, 1])
                    for i in range(val_errors.shape[0])]
    lams = [val_errors[i, idx, 0] for i, idx in enumerate(best_lam_idx)]

    legend = []

    for i in range(val_errors.shape[0]):
        plt.semilogx(val_errors[i, :, 0], val_errors[i, :, 1])
        legend.append(f"Batch {i+1}")

    plt.grid(True, which="both")
    plt.legend(legend)
    plt.show()

    wr = ridge_regression(X_train=X,
                          y_train=y,
                          lam=lams[0],
                          iterations=1000,
                          learning_rate=0.01)

    terror = np.linalg.norm(((Xt.T)@w) - yt)**2
    terror_r = np.linalg.norm(((Xt.T)@wr) - yt)**2

    error = np.linalg.norm(((X.T)@w) - y)**2
    error_r = np.linalg.norm(((X.T)@wr) - y)**2

    print(f"Dire: {error}   {terror}")
    print(f"Ridg: {error_r} {terror_r}")
