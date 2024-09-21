import numpy as np


class naiveBayes():

    # data points are along rows
    # y assumed to be {+1, 0}

    def __init__(self, data_file, laplace_smoothing=False):
        self.data_file = data_file
        self.smoothing = laplace_smoothing

        self.p_cap = None
        self.p_class = None

    def load_data(self):
        data = np.loadtxt(self.data_file, delimiter=',')

        self.num_points = data.shape[0]
        self.dim = data.shape[1]-1

        # convert all non-zero entries to 1
        self.X = data[:, 0:self.dim].astype(int)
        self.X[np.nonzero(self.X)] = 1

        # convert all non-one entries to 0
        self.y = data[:, self.dim].astype(int)
        self.y[np.nonzero(self.y != 1)] = 0

    def compute_mle(self):
        self.p_cap = self.y.sum()/self.num_points
        self.p_class = np.zeros((2, self.dim))

        self.cnt = np.zeros(2)

        for i in range(self.num_points):
            self.p_class[self.y[i]] += self.X[i]
            self.cnt[self.y[i]] += 1

        if self.smoothing == True:
            for i in range(2):
                self.p_class[i] += np.ones_like(self.p_class[i])
                self.cnt[i] += 1

        for i in range(2):
            self.p_class[i] /= self.cnt[i]

        # compute decision boundary based on bayes thm

        # TODO calculations iterartively
        self.w = (np.log(self.p_class[1])
                  - np.log(self.p_class[0])
                  + np.log(1 - self.p_class[0])
                  - np.log(1 - self.p_class[1]))

        self.b = (np.log(1 - self.p_class[1])
                  - np.log(1 - self.p_class[0]))

        self.b = self.b.sum()
        self.b += np.log(self.p_cap) - np.log(1 - self.p_cap)

    def predict(self, X_test, y_test=None):
        # convert all non-zero values to 1 in X_test
        X_t = X_test.copy().astype(int)
        X_t[np.nonzero(X_t)] = 1

        # convert all non-one values to 0 in y_test
        y_t = None
        if y_test is not None:
            y_t = y_test.copy()
            y_t[np.nonzero(y_t != 1)] = 0

        y_pred = self.w@(X_t.T) + self.b

        for i in range(y_pred.shape[0]):
            if y_pred[i] < 0:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        accuracy = None
        if y_test is not None:
            error = 0
            for i, yi in enumerate(y_pred):
                if (yi != y_t[i]):
                    error += 1

            accuracy = 1.0 - (error/y_t.shape[0])

        return y_pred, accuracy

    def run(self):
        self.load_data()
        self.compute_mle()


if __name__ == "__main__":
    pass
