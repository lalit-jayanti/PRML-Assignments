import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Kmeans():
    """K-Means algorithm

    Working:
        - Data is loaded from a .csv file or directly (data as row vectors)
        - Clusters are initialized
        - K Means are computed (Step 1)
        - Labels are reassigned (Step 2)
        - Step 1 and Step 2 are repeated until convergence
        - Final clusters are stored in self.z_cur

    Initialization:
        - "random"
        - "Kmeans++"
    """

    def __init__(self, data_file, num_clusters, seed, initialize_method="random", ext_data=None) -> None:
        self.data_file = data_file
        self.k = num_clusters
        # TODO handle more colors (>10)
        self.colors = cm.tab20(range(2*self.k))

        self.initialize = {"random": self.random_init,
                           "Kmeans++": self.kmeans_plus_init}
        self.initialize_method = initialize_method

        self.z_pre = None
        self.z_cur = None

        self.m_cur = None

        self.errors = np.empty((0))

        self.seed = seed
        np.random.seed(seed=self.seed)

        self.data = None

        if ext_data is not None:
            self.data = ext_data.copy()

    def load_data(self, data_file):
        if self.data is None:
            self.data = np.loadtxt(data_file, delimiter=',')
        self.arr = self.data.astype(float)

        self.num_points = self.arr.shape[0]
        self.dims = self.arr.shape[1]

        self.z_cur = np.zeros((self.num_points), dtype=int)
        self.z_pre = np.zeros((self.num_points), dtype=int)

        self.m_cur = np.zeros((self.k, self.dims), dtype=float)
        self.m_pre = np.zeros((self.k, self.dims), dtype=float)

    def random_init(self):
        self.z_cur = np.random.randint(
            low=0, high=self.k, size=(self.num_points))

    def kmeans_plus_init(self):
        # initialization
        self.prob = np.ones(self.num_points)/self.num_points

        # assign k means
        for i in range(self.k):
            # assignment
            idx = np.random.choice(a=self.num_points,
                                   size=None,
                                   p=self.prob)
            self.m_cur[i] = self.arr[idx]

            # probability recalculation
            self.prob = np.ones(self.num_points, dtype=float)*-1
            for n in range(self.num_points):
                for j in range(i):
                    diff = np.inner(self.arr[n]-self.m_cur[j],
                                    self.arr[n]-self.m_cur[j])
                    if (self.prob[n] == -1.0):
                        self.prob[n] = diff
                    else:
                        self.prob[n] = np.minimum(self.prob[n], diff)

            self.prob = self.prob/(np.sum(self.prob))

        # assign points to clusters
        for i in range(self.num_points):
            pre_diff = np.inf
            for ki in range(self.k):
                diff = np.inner(self.arr[i]-self.m_cur[ki],
                                self.arr[i]-self.m_cur[ki])

                if (pre_diff > diff):
                    pre_diff = diff
                    self.z_cur[i] = ki

        self.compute_means()

    def compute_means(self):
        cnt = np.zeros((self.k), dtype=float)
        self.m_cur = np.zeros((self.k, self.dims), dtype=float)

        for i in range(self.num_points):
            ki = self.z_cur[i]
            cnt[ki] += 1
            self.m_cur[ki] += self.arr[i]

        for ki in range(self.k):

            if (cnt[ki] == 0):
                self.m_cur[ki] = np.inf
            else:
                self.m_cur[ki] = self.m_cur[ki]/cnt[ki]

    def compute_error(self):
        err = 0
        for i in range(self.num_points):
            dist = np.linalg.norm(self.arr[i]-self.m_cur[self.z_cur[i]])**2
            err = err + dist

        self.errors = np.append(self.errors, err)

    def reassign(self):
        self.z_pre = self.z_cur.copy()

        for i in range(self.num_points):
            pre_diff = np.linalg.norm(self.arr[i]-self.m_cur[self.z_cur[i]])

            for ki in range(self.k):
                diff = np.linalg.norm(self.arr[i]-self.m_cur[ki])

                if (pre_diff > diff):
                    pre_diff = diff
                    self.z_cur[i] = ki

    def check_convergence(self):
        if (np.array_equal(a1=self.z_cur, a2=self.z_pre)):
            return True
        else:
            return False

    def run(self):
        self.load_data(self.data_file)
        self.initialize[self.initialize_method]()

        while True:

            self.compute_means()
            self.compute_error()
            self.reassign()

            converged = self.check_convergence()

            if converged:
                break

    def plot_data(self):

        plt.subplot(1, 2, 1)
        plt.grid('on')
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        for i in range(self.num_points):
            plt.plot(self.arr[i, 0], self.arr[i, 1], '.',
                     color=self.colors[2*self.z_cur[i]])

        for ki in range(self.k):
            plt.plot(self.m_cur[ki, 0], self.m_cur[ki, 1], 'o',
                     color=self.colors[2*ki])

        plt.subplot(1, 2, 2)
        plt.xlabel("Iteration", fontsize=15)
        plt.ylabel("Error", fontsize=15)
        plt.grid('on')
        plt.plot(self.errors)

    def generate_voronoi(self):
        # works only in 2d case

        # determine bounds
        xmin = np.min(self.arr[:, 0])*1.1
        xmax = np.max(self.arr[:, 0])*1.1

        ymin = np.min(self.arr[:, 1])*1.1
        ymax = np.max(self.arr[:, 1])*1.1

        self.x_res = 1000
        self.y_res = int(self.x_res*((ymax-ymin)/(xmax-xmin)))
        self.voronoi = np.zeros((self.y_res, self.x_res, 4), dtype=float)

        for i in range(self.x_res):
            for j in range(self.y_res):

                xi = i*((xmax-xmin)/self.x_res) + xmin
                yj = j*((ymax-ymin)/self.y_res) + ymin

                p = np.array([xi, yj])

                pre_diff = np.inner(p-self.m_cur[0],
                                    p-self.m_cur[0])
                mi = 0
                for ki in range(self.k):
                    diff = np.inner(p-self.m_cur[ki],
                                    p-self.m_cur[ki])

                    if (pre_diff > diff):
                        pre_diff = diff
                        mi = ki

                self.voronoi[j, i] = self.colors[2*mi + 1]

        plt.imshow(self.voronoi, origin='lower',
                   extent=[xmin, xmax, ymin, ymax])
        plt.grid('on')
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        for i in range(self.num_points):
            plt.plot(self.arr[i, 0], self.arr[i, 1], '.',
                     color=self.colors[2*self.z_cur[i]])


if __name__ == '__main__':

    data_file = "../data/cm_dataset_2 - cm_dataset.csv"
    km = Kmeans(data_file=data_file,
                num_clusters=3,
                seed=10,
                initialize_method="random")
    km.run()
    km.plot_data()
    plt.grid('on')
    plt.show()
