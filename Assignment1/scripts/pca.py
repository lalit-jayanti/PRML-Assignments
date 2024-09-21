import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class PCA():
    """Principal Component Analysis algorithm

    Working:
        - Data is loaded from a .npy file (data as row vectors)
        - Dataset is centred
        - Covariance matrix is computed
        - Principal components are evaluated (stored in self.w)
    """

    def __init__(self, data_file, label_file=None) -> None:

        self.data_file = data_file

        self.data = None
        self.arr = None
        self.mean = None

        # if labels are known before hand (for testing)
        # currently supported for 10 labels
        self.label_file = label_file

        self.num_points = 0
        self.dims = 0
        self.num_cols = 0
        self.num_rows = 0

        self.lam = None
        self.w = None

    def load_data(self, data_file):
        self.data = np.load(data_file)
        self.arr = self.data.astype(float)

        self.num_points = self.arr.shape[0]
        self.dims = self.arr.shape[1]

        self.num_rows = np.sqrt(self.arr.shape[1]).astype(int)
        self.num_cols = self.num_rows

        self.colors = cm.tab10(range(10))

        if self.label_file is not None:
            self.labels = np.load(self.label_file).astype(int)
        else:
            self.labels = np.zeros((self.num_points), dtype=int)

    def center_dataset(self):
        self.mean = np.sum(a=self.arr, axis=0)/self.num_points
        self.arr = self.arr - self.mean

    def compute_covariance_matrix(self):
        self.c = np.zeros((self.dims, self.dims), dtype=float)
        for i in range(self.num_points):
            d = np.outer(a=self.arr[i], b=self.arr[i])
            self.c = self.c + d

        self.c = self.c/self.num_points
        self.c = np.matrix(self.c)

        self.lam, self.w = np.linalg.eigh(self.c)
        self.w = self.w.T

    def reconstruct(self, index, order):
        if (index >= self.num_points or index < 0):
            print(
                f"Invalid index, index must be between 0-{self.num_points-1}")
            return

        if (order > self.dims or order < 0):
            print(f"Invalid order, order must be between 0-{self.dims}")
            return

        image = self.arr[index]
        reconstructed_image = np.zeros_like(image)

        for i in range(order):
            wi = self.w[self.dims-i-1]
            coefficient = np.inner(image, wi)
            component = coefficient*wi
            reconstructed_image = reconstructed_image + component

        return reconstructed_image+self.mean

    def project_plt(self):
        w1 = self.w[self.dims-1]
        w2 = self.w[self.dims-2]
        for i in range(self.num_points):
            xi = np.inner(self.arr[i], w1)
            yi = np.inner(self.arr[i], w2)

            plt.plot(xi, yi, '.', color=self.colors[self.labels[i]])

        plt.grid("on")
        plt.show()

    def run(self):
        self.load_data(data_file=self.data_file)
        self.center_dataset()
        self.compute_covariance_matrix()


if __name__ == '__main__':

    pc = PCA(data_file="../data/mnist/random_mnist_1000.npy",
             label_file="../data/mnist/random_mnist_1000_labels.npy")
    pc.run()
    pc.project_plt()
