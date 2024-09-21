import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class kernelPCA():
    """Kernelized Principal Component Analysis algorithm

    Working:
        - Data is loaded from a .npy file or directly (data as row vectors)
        - Centred Kernel matrix is evaluated
        - Principal components are evaluated (stored in self.alpha)
    
    Kernels:
        - "radial basis"
        - "polynomial"
        - "custom"
        - "none" (standard inner product)
    """

    def __init__(self, data_file, kernel, params, label_file=None, ext_data=None) -> None:

        self.data_file = data_file
        # if labels are known before hand (for testing)
        # currently supported for 10 labels
        self.label_file = label_file

        self.kernel_dict = {"radial basis": self.radial_basis_kernel,
                            "polynomial": self.polynomial_kernel,
                            "custom": self.custom_kernel,
                            "none": self.standard_inner_product}
        self.kernel_type = kernel
        self.params = params

        self.data = None
        self.arr = None
        self.labels = None

        self.num_points = 0
        self.dims = 0
        self.num_cols = 0
        self.num_rows = 0

        self.nlam = None
        self.beta = None
        self.alpha = None

        self.ker_mat = None
        self.ker_mat_sum1 = None
        self.ker_mat_sum2 = None

        if ext_data is not None:
            self.data = ext_data.copy()

    def load_data(self, data_file):
        if self.data is None:
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

    def standard_inner_product(self, i, j):
        return np.inner(self.arr[i], self.arr[j])

    def polynomial_kernel(self, i, j):
        res = np.inner(self.arr[i], self.arr[j])
        return (res+1)**self.params["d"]

    def radial_basis_kernel(self, i, j):
        x = self.arr[i] - self.arr[j]
        res = -np.linalg.norm(x)**2/(2*(self.params["sigma"]**2))
        return np.exp(res)

    def custom_kernel(self, i, j):
        # custom kernel works only on dim 2
        x = np.array([self.arr[i, 0], self.arr[i, 1], (10 **
                     self.params["z"])*np.sqrt(self.arr[i, 0]**2 + self.arr[i, 1]**2)])
        y = np.array([self.arr[j, 0], self.arr[j, 1], (10 **
                     self.params["z"])*np.sqrt(self.arr[j, 0]**2 + self.arr[j, 1]**2)])
        res = np.inner(x, y)
        return res

    def kernel(self, i, j):
        return self.kernel_dict[self.kernel_type](i, j)

    def build_kernel_matrix(self):
        self.ker_mat = np.zeros(
            (self.num_points, self.num_points), dtype=float)
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.ker_mat[i, j] = self.kernel(i, j)

        self.ker_mat_sum1 = np.sum(self.ker_mat, axis=0)
        self.ker_mat_sum2 = np.sum(self.ker_mat)

    def compute_k_matrix(self):
        self.k = np.zeros((self.num_points, self.num_points), dtype=float)
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.k[i, j] = (self.ker_mat[i, j]
                                - (self.ker_mat_sum1[i]/self.num_points)
                                - (self.ker_mat_sum1[j]/self.num_points)
                                + (self.ker_mat_sum2/(self.num_points**2)))

        self.nlam, self.beta = np.linalg.eigh(self.k)
        # TODO handle division and sqrt problems
        self.beta = self.beta.T
        self.alpha = np.zeros_like(self.beta)
        for i in range(len(self.nlam)):
            if (self.nlam[i] > 0.0):
                self.alpha[i] = self.beta[i]/np.sqrt(self.nlam[i])

    def project_plt(self):
        a1 = self.alpha[-1]
        a2 = self.alpha[-2]
        for i in range(self.num_points):
            xi = np.inner(self.k[i], a1)
            yi = np.inner(self.k[i], a2)

            plt.plot(xi, yi, '.', color=self.colors[self.labels[i]])

        plt.grid("on")
        plt.show()

    def run(self):

        self.load_data(self.data_file)
        self.build_kernel_matrix()
        self.compute_k_matrix()


if __name__ == '__main__':

    kpc = kernelPCA(data_file="../data/mnist/random_mnist_1000.npy",
                    kernel="none",
                    params={"d": 2,
                            "z": 8,
                            "sigma": 1800},
                    label_file="../data/mnist/random_mnist_1000_labels.npy")

    kpc.run()

    plt.plot(kpc.nlam)
    plt.show()
    kpc.project_plt()
