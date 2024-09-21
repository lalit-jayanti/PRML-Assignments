import numpy as np
import matplotlib.pyplot as plt
import time


def linear_regression(X_train, y_train):
    """Linear Regression

    (X - data as column vectors)
    Working:
        - Returns analytical to the least squares problem

    """
    w = np.matmul(np.linalg.inv(X_train@(X_train.T)), X_train@y_train)
    return w


def gradient_descent_linear_regression(X_train, y_train,
                                       iterations=1000,
                                       learning_rate=0.01):
    """Gradient Descent

    (X - data as column vectors)
    Working:
        - Uses gradient descent to provide the 
        solution to the least squares problem
    """

    w = linear_regression(X_train=X_train,
                          y_train=y_train)

    wg = np.ones(X_train.shape[0])*1e-3
    itr = iterations
    eta = learning_rate

    xxt = X_train@(X_train.T)
    xy = X_train@y_train

    arr_g = []

    for i in range(itr):
        gradf = 2*((xxt@wg) - (xy))
        wg = wg - eta*gradf/np.linalg.norm(gradf)
        arr_g.append(np.linalg.norm(wg-w))

    return wg, arr_g


def stochastic_descent_linear_regression(X_train, y_train,
                                         batch_size,
                                         iterations=1000,
                                         learning_rate=0.01,
                                         seed=0):
    """Stochastic Gradient Descent

    (X - data as column vectors)
    Working:
        - Uses stochastic gradient descent to provide the 
        solution to the least squares problem
        - Batch size can be varied based on computational
        resources available
    """
    np.random.seed(seed=seed)
    rng = np.random.default_rng()

    w = linear_regression(X_train=X_train,
                          y_train=y_train)

    ws = np.ones(X_train.shape[0])*1e-3
    itr = iterations
    eta = learning_rate

    arr_s = []

    for i in range(itr):

        idx = rng.choice(X_train.shape[1], size=batch_size, replace=False)
        xi = X_train[:, idx]
        yi = y_train[idx]

        xxt = xi@(xi.T)
        xy = xi@yi

        grads = 2*((xxt@ws) - (xy))
        ws = ws - eta*grads/np.linalg.norm(grads)
        arr_s.append(np.linalg.norm(ws-w))

    return ws, arr_s


if __name__ == '__main__':
    pass