import numpy as np
import matplotlib.pyplot as plt
import sys  # to query system recusrion limit
from treelib import Node, Tree  # to visualize decison tree


def entropy(p):

    thresh = 1e-15

    if p <= thresh:
        return 0
    if 1-p <= thresh:
        return 0

    return -(p*np.log(p) + (1-p)*np.log(1-p))


def compute_decision(X, y, idx):
    ft = X[:, idx]

    sort_idx = np.argsort(ft)
    ft = ft[sort_idx]
    yi = y[sort_idx]

    num = yi.shape[0]
    ones = np.count_nonzero(yi == 1)

    init_entropy = entropy(ones/num)

    cur_ones = 0
    cur_val = ft[0]
    max_infogain = -1

    for i in range(num-1):

        if yi[i] == 1:
            cur_ones += yi[i]

        nex_ones = ones-cur_ones

        if (ft[i] == ft[i+1]):
            continue

        gam = (i+1)/num
        cur_entropy = (gam*entropy(cur_ones/(i+1))
                       + (1-gam)*entropy(nex_ones/(num-i-1)))
        infogain = init_entropy - cur_entropy

        if (max_infogain < infogain):
            cur_val = ft[i]
            max_infogain = infogain

    return cur_val, max_infogain


class decision:

    def __init__(self, value=None, idx=-1, leaf=False):
        self.value = value
        self.idx = int(idx)
        self.leaf = leaf

        self.id = np.random.random(1)

    def __repr__(self):
        if self.leaf:
            return f"(val: {self.value} rnd:{self.id})"
        else:
            return f"(val: {self.value}, idx: {self.idx}, rnd:{self.id})"


class decisionTree:

    # data points are along rows
    # y assumed to be {+1, -1}

    def __init__(self, data_file, max_depth=None, X_data=None, y_data=None):

        self.data_file = data_file
        self.tree = {}

        if max_depth is None:
            self.max_depth = sys.getrecursionlimit()
        else:
            self.max_depth = min(max_depth, sys.getrecursionlimit())

        # if data_file is None
        # set data to given data
        # else override given data

        self.X_data = X_data
        self.y_data = y_data

    def load_data(self, data_file):
        self.X = None
        self.y = None
        self.num_points = 0
        self.dim = 0

        if data_file is None:
            self.X = self.X_data
            self.y = self.y_data.astype(int)
            self.num_points = self.y.shape[0]
            self.dim = self.X.shape[1]

        else:
            data = np.loadtxt(data_file, delimiter=',')
            # data points are in rows
            # features along columns

            self.dim = data.shape[1]-1
            self.num_points = data.shape[0]

            self.X = data[:, 0:self.dim]
            self.y = data[:, self.dim].astype(int)
            # convert all non-one entries to -1
            self.y[np.nonzero(self.y != 1)] = -1

    def build_tree_util(self, X, y, depth):

        num_features = X.shape[1]
        num_points = X.shape[0]

        # BASE CASE
        if (num_points == 1):
            return decision(value=y[0], leaf=True)

        ones = np.count_nonzero(y == 1)
        # if all values are either -1 or 1
        # no furthur computation needed
        if (ones == num_points):
            return decision(value=1, leaf=True)
        elif (ones == 0):
            return decision(value=-1, leaf=True)

        if (depth == self.max_depth):
            # if max depth assign based on
            # count of ones
            if (ones >= num_points/2):
                return decision(value=1, leaf=True)
            else:
                return decision(value=-1, leaf=True)

        # RECURSIVE CASE
        # greedy compute best decision

        best_idx = 0
        best_val = 0
        max_infogain = -1

        for idx in range(num_features):

            val, infogain = compute_decision(X, y, idx)
            if (infogain > max_infogain):
                best_idx = idx
                best_val = val
                max_infogain = infogain

        # split dataset using best decision
        split_idx = np.zeros_like(y).astype(bool)
        for i in range(num_points):
            if (X[i, best_idx] <= best_val):
                split_idx[i] = True
            else:
                split_idx[i] = False

        X_yes = X[split_idx, :]
        y_yes = y[split_idx]

        X_no = X[~split_idx, :]
        y_no = y[~split_idx]

        # recursively call to build tree
        cur_decision = decision(best_val, best_idx)
        yes_decision = self.build_tree_util(X_yes, y_yes, depth+1)
        no_decision = self.build_tree_util(X_no, y_no, depth+1)

        # index 0 corresponds to "YES", index 1 corresponds to "NO"
        self.tree[cur_decision] = [yes_decision, no_decision]

        return cur_decision

    def build_tree(self):
        self.root_decision = self.build_tree_util(self.X, self.y, depth=0)

    def predict_util(self, xi, node):
        # BASE CASE
        if node.leaf == True:
            return node.value

        val, idx = node.value, node.idx
        # RECURSIVE CASE
        if (xi[idx] <= val):
            # index 0 corresponds to "YES"
            res = self.predict_util(xi, self.tree[node][0])
            return res
        else:
            # index 1 corresponds to "NO"
            res = self.predict_util(xi, self.tree[node][1])
            return res

    def predict(self, X_test, y_test=None):
        # data points are along rows
        y_pred = np.zeros(X_test.shape[0]).astype(int)
        for i, xi in enumerate(X_test):
            y_pred[i] = self.predict_util(xi, self.root_decision)

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

    def to_dot_util(self, root, labels=None):
        n = self.tree[root]

        for ni in n:
            if (ni.leaf == True):
                self.vis_tree.create_node(f"{ni.value}", ni, parent=root)
            else:
                if labels is None:
                    self.vis_tree.create_node(
                        f"x[{ni.idx}]<={ni.value}", ni, parent=root)
                else:
                    self.vis_tree.create_node(
                        f"{labels[ni.idx]}<={ni.value}", ni, parent=root)                   

            if (ni.leaf == False):
                self.to_dot_util(ni, labels)

    def to_dot(self, save_file="../outputs/decision_tree.dot", labels=None):
        self.vis_tree = Tree()
        if self.root_decision.leaf == True:
            self.vis_tree.create_node(
                f"{self.root_decision.value}", self.root_decision)
        else:
            if labels is None:
                self.vis_tree.create_node(
                    f"x[{self.root_decision.idx}]<={self.root_decision.value}", self.root_decision)
            else:
                self.vis_tree.create_node(
                    f"{labels[self.root_decision.idx]}<={self.root_decision.value}", self.root_decision)
                               
        self.to_dot_util(self.root_decision, labels)

        self.vis_tree.to_graphviz(save_file)

    def run(self):
        self.load_data(self.data_file)
        self.build_tree()


if __name__ == "__main__":

    pass
