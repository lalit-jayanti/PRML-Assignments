import numpy as np
import matplotlib.pyplot as plt
import pdb


class gaussianEM():

    def __init__(self, data_file, num_mixtures, seed=0, max_iterations=100):
        """Gaussian Mixture Model

        (data as row vectors)
        Working:
            - Use the Gaussian Distribution and
            perform E step and M step (repeatedly 
            until max iteration or convergence)

            - E step
                - Compute lambdas (self.lam)
            - M step
                - Compute:
                    Means (self.means)
                    Covariance (self.covar)
                    Mixture Weights (self.pi)
        """

        self.data_file = data_file
        self.k = num_mixtures
        self.seed = seed
        self.max_iterations = max_iterations
        np.random.seed(seed=seed)

    def load_data(self):
        self.data = np.loadtxt(self.data_file, delimiter=',')
        self.data.astype(float)

        if len(self.data.shape) == 1:
            self.data = self.data.reshape(self.data.shape[0], 1)

        self.X = self.data.copy()

        self.dim = self.data.shape[1]
        self.num_points = self.data.shape[0]

        self.means = np.random.rand(self.k, self.dim)*1

        self.covar = np.broadcast_to((1e0)*np.identity(
            self.dim, dtype=float), (self.k, self.dim, self.dim)).copy()

        self.covar_inv = np.empty_like(self.covar, dtype=float)
        self.covar_det = np.empty((self.k), dtype=float)

        for j in range(self.k):
            self.covar_inv[j] = np.linalg.inv(self.covar[j])
            self.covar_det[j] = np.linalg.det(self.covar[j])

        self.pi = np.ones((self.k), dtype=float)/self.k

        self.lam = np.zeros((self.k, self.num_points), dtype=float)

    def gauss_pdf(self, i, j):

        x = self.X[i]-self.means[j]
        ep = -0.5*(x@(self.covar_inv[j])@(x.T))

        res = (-self.dim/2)*np.log(2*np.pi) + \
            (-0.5)*np.log(self.covar_det[j]) + ep

        return np.exp(res)

    def e_step(self):

        for i in range(self.num_points):

            sm = 0
            for j in range(self.k):
                sm += self.pi[j]*self.gauss_pdf(i, j)

            for j in range(self.k):
                self.lam[j, i] = (self.pi[j]*self.gauss_pdf(i, j))/sm

    def m_step(self):

        self.prev_means = self.means.copy()
        self.prev_covar = self.covar.copy()
        self.prev_pi = self.pi.copy()

        for j in range(self.k):

            mean_sm = 0
            covar_sm = np.zeros((self.dim, self.dim))
            lam_sm = 0

            for i in range(self.num_points):
                mean_sm += self.lam[j, i]*self.X[i]
                lam_sm += self.lam[j, i]

            self.means[j] = mean_sm/lam_sm
            self.pi[j] = lam_sm/self.num_points

            for i in range(self.num_points):
                x = self.X[i]-self.means[j]
                covar_sm += self.lam[j, i]*np.outer(x, x)

            self.covar[j] = covar_sm/lam_sm

            reg = 1e-6
            if np.linalg.matrix_rank(self.covar[j]) < self.dim:

                self.covar[j] = self.covar[j] + reg*np.eye(self.dim)
                self.covar[j] = (self.covar[j] + self.covar[j].T)/2

            self.covar_inv[j] = np.linalg.inv(self.covar[j])
            self.covar_det[j] = np.linalg.det(self.covar[j])

    def log_likelihood(self):

        logl = 0

        for i in range(self.num_points):
            mix_sm = 0
            for j in range(self.k):
                mix_sm += self.pi[j]*self.gauss_pdf(i, j)
            logl += np.log(mix_sm)

        if (logl > 0):

            pass

        return logl

    def check_convergence(self):
        return False

    def run(self):
        self.load_data()
        self.logl = []

        for itr in range(self.max_iterations):

            self.e_step()
            self.m_step()
            self.logl.append(self.log_likelihood())


if __name__ == '__main__':

    gem = gaussianEM(data_file="../data/A2Q1.csv",
                     num_mixtures=4,
                     seed=1,
                     max_iterations=20)

    gem.run()

    print(gem.logl)

    plt.plot(gem.logl)
    plt.show()
