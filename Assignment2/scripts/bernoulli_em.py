import numpy as np
import matplotlib.pyplot as plt
import pdb


class bernoulliEM():

    def __init__(self, data_file, num_mixtures, seed=0, max_iterations=100):
        """Bernoulli Mixture Model

        (data as row vectors)
        Working:
            - Use the Bernoulli Distribution and
            perform E step and M step (repeatedly 
            until max iteration or convergence)

            - E step
                - Compute lambdas (self.lam)
            - M step
                - Compute:
                    Probabilities (self.prob)
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

        self.pi = np.ones((self.k), dtype=float)/self.k
        self.lam = np.zeros((self.k, self.num_points), dtype=float)

        self.prob = np.random.rand(self.k, self.dim)
        self.st = self.prob.copy()

    def pdf(self, i, j):

        pd = 0
        for d in range(self.dim):
            pd += (self.X[i, d]*np.log(self.prob[j, d])
                   + (1-self.X[i, d])*np.log((1-self.prob[j, d])))

        return np.exp(pd)

    def e_step(self):

        for i in range(self.num_points):

            sm = 0
            for j in range(self.k):
                sm += self.pi[j]*self.pdf(i, j)

            for j in range(self.k):
                self.lam[j, i] = (self.pi[j]*self.pdf(i, j))/sm

    def m_step(self):
        self.prev_prob = self.prob.copy()
        self.prev_pi = self.pi.copy()

        for k in range(self.k):
            for d in range(self.dim):
                sm_lm_x = 0
                sm_lm = 0
                for i in range(self.num_points):
                    sm_lm_x += self.lam[k, i]*self.X[i, d]
                    sm_lm += self.lam[k, i]

                self.prob[k, d] = sm_lm_x/sm_lm
                reg = 1e-15
                # to prevent pdf from encountering 0 divide in log
                self.prob[k, d] = max(reg, self.prob[k, d])
                self.prob[k, d] = min(1-reg, self.prob[k, d])

        for k in range(self.k):
            sm_lm = 0
            for i in range(self.num_points):
                sm_lm += self.lam[k, i]

            self.pi[k] = sm_lm/self.num_points

    def log_likelihood(self):
        logl = 0
        for i in range(self.num_points):
            mix_sm = 0
            for j in range(self.k):
                mix_sm += self.pi[j]*self.pdf(i, j)
            logl += np.log(mix_sm)

        if (logl > 0):
            pdb.set_trace()

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

    bem = bernoulliEM(data_file="../data/A2Q1.csv",
                      num_mixtures=4,
                      seed=3,
                      max_iterations=10)

    bem.run()

    plt.figure(figsize=(10, 5))

    for i, bi in enumerate(bem.prob):
        plt.subplot(2, 2, i+1)
        plt.axis("off")
        plt.title(f"Cluster {i+1}")
        plt.imshow(bi.reshape(5, 10))

    plt.show()
