import numpy as np
import matplotlib.pyplot as plt


class datasetGenerator():
    """Generates randomized datasets [Meant for MNIST dataset]

    Randomly split the dataset ensuring that
    equal number of data-points are chosen for each class
    """

    def __init__(self) -> None:

        self.data = {0: [], 1: [], 2: [], 3: [], 4: [],
                     5: [], 6: [], 7: [], 8: [], 9: []}
        self.split_dataset = None
        self.split_labels = None

    def convert(self, image_file, label_file, num=None):

        f = open(image_file, "rb")
        l = open(label_file, "rb")

        f_magic_num = int.from_bytes(f.read(4), byteorder="big", signed=True)
        l_magic_num = int.from_bytes(l.read(4), byteorder="big", signed=True)

        num_images = int.from_bytes(f.read(4), byteorder="big", signed=True)
        _ = l.read(4)

        if num is not None:
            num_images = min(num, num_images)

        num_rows = int.from_bytes(f.read(4), byteorder="big", signed=True)
        num_cols = int.from_bytes(f.read(4), byteorder="big", signed=True)

        for i in range(num_images):
            label = ord(l.read(1))
            image = np.zeros(num_cols*num_rows, dtype=int)
            for j in range(num_rows*num_cols):
                image[j] = ord(f.read(1))

            self.data[label].append(image)

        for i in range(10):
            print(f"{i}: {len(self.data[i])}")

        f.close()
        l.close()

    def random_split(self, num, save_location, seed):
        np.random.seed(seed=seed)
        size = len(self.data.keys())
        self.split_dataset = np.empty((size*num, 28*28), dtype=int)
        self.split_labels = np.empty((size*num), dtype=int)

        count = 0
        for label in self.data.keys():
            rand_index = np.random.randint(len(self.data[label]), size=(num))
            for i in rand_index:
                self.split_dataset[count] = self.data[label][i]
                self.split_labels[count] = label
                count = count + 1

        np.save(file=f"{save_location}/random_mnist_{size*num}",
                arr=self.split_dataset)
        np.save(
            file=f"{save_location}/random_mnist_{size*num}_labels", arr=self.split_labels)
        print(
            f"Saved {size*num} data points to '{save_location}/random_mnist_{size*num}.npy'")
        print(
            f"Saved {size*num} data labels to '{save_location}/random_mnist_{size*num}_labels.npy'")


if __name__ == '__main__':
    imgf = "../data/mnist/train-images-idx3-ubyte"
    labf = "../data/mnist/train-labels-idx1-ubyte"
    save_location = "../data/mnist"

    dg = datasetGenerator()
    dg.convert(image_file=imgf, label_file=labf, num=None)
    dg.random_split(num=100, save_location=save_location, seed=0)
