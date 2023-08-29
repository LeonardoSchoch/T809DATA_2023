# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    p = np.array([])
    for c in classes:
        p = np.append(p, np.count_nonzero(np.array(targets) == c) / len(targets))
    return p


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[np.where(features[:, split_feature_index] < theta)[0]]
    targets_1 = targets[np.where(features[:, split_feature_index] < theta)[0]]

    features_2 = features[np.where(features[:, split_feature_index] >= theta)[0]]
    targets_2 = targets[np.where(features[:, split_feature_index] >= theta)[0]]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 0.5*(1-np.sum(np.power(prior(targets, classes), 2)))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return (t1.shape[0] * g1 / n) + (t2.shape[0] * g2 / n)


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(min(features[:, i]), max(features[:, i]), num_tries + 2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            crt_gini = total_gini_impurity(features, targets, classes, i, theta)
            if crt_gini < best_gini:
                best_gini, best_dim, best_theta = crt_gini, i, theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        n = len(self.classes)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = np.sum((self.test_targets == self.classes[i]) & (self.guess() == self.classes[j]))
        return matrix
