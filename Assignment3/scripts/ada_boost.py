import numpy as np
import matplotlib.pyplot as plt

from decision_tree import decisionTree


class adaBoost:

    # data points are along rows
    # y assumed to be {+1, -1}

    def __init__(self, data_file, max_itr=10, max_tree_depth=3, data=None, seed=0):

        self.data_file = data_file

        # if data_file is None
        # set data to given data
        # else override given data
        self.data = data

        self.max_itr = max_itr
        self.trees = np.empty(max_itr, dtype=object)
        self.alpha = np.empty(max_itr, dtype=float)

        self.max_tree_depth = max_tree_depth

        np.random.seed(seed=seed)

    def load_data(self, data_file):
        if data_file is None:
            data = self.data
        else:
            data = np.loadtxt(data_file, delimiter=',')

        self.num_points = data.shape[0]
        self.dim = data.shape[1]-1

        # data points are in rows
        # features along columns
        self.X = data[:, 0:self.dim]
        self.y = data[:, self.dim].astype(int)
        # convert all non-one entries to -1
        self.y[np.nonzero(self.y != 1)] = -1


        # initialize selection probability to be uniform
        self.D = np.ones_like(self.y).astype(float)/self.num_points

    def step(self, itr):
        self.D_prev = self.D.copy()

        # sample with replacement with specified probability
        idx = np.random.choice(
            self.num_points, size=self.num_points, p=self.D, replace=True)

        # generate sampled data
        self.Xi = self.X[idx]
        self.yi = self.y[idx]

        # train weak learner
        ht = decisionTree(data_file=None,
                          max_depth=self.max_tree_depth,
                          X_data=self.Xi,
                          y_data=self.yi)
        ht.run()

        # run prediction
        y_pred, accuracy = ht.predict(self.X, self.y)
        err = 1-accuracy

        # handle err = 0
        if err == 0:
            alf = 0.0
            print(f"Overfit Tree; itr{itr}")
        else:
            alf = np.log(np.sqrt((1-err)/(err)))

        self.trees[itr] = ht
        self.alpha[itr] = alf

        # compute next self.D
        zt = 0

        for i in range(self.num_points):

            self.D[i] = self.D_prev[i]*np.exp(-alf*self.y[i]*y_pred[i])
            zt += self.D_prev[i]*np.exp(-alf*self.y[i]*y_pred[i])

        self.D = self.D/zt

    def predict(self, X_test, y_test=None):
        # data points are along rows
        y_pred = np.zeros(X_test.shape[0]).astype(int)

        for i, xi in enumerate(X_test):
            hx = 0
            for t in range(self.max_itr):
                hti, _ = self.trees[t].predict(np.array([xi]))
                hti = hti[0]
                hx += self.alpha[t]*hti

            if hx >= 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1

        # convert all non-one values to -1 in y_test
        y_t = None
        if y_test is not None:
            y_t = y_test.copy()
            y_t[np.nonzero(y_t != 1)] = -1

        accuracy = None
        if y_test is not None:
            error = 0
            for i, yi in enumerate(y_pred):
                if (yi != y_t[i]):
                    error += 1

            accuracy = 1.0 - (error/y_t.shape[0])

        return y_pred, accuracy

    def run(self):

        self.load_data(self.data_file)
        for itr in range(self.max_itr):
            #print(itr)
            self.step(itr)

        print(self.alpha)


if __name__ == "__main__":
    pass
