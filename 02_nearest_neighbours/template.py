# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.sqrt(np.sum(np.power(x - y, 2)))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i, :])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    return np.argsort(euclidian_distances(x, points))[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    t_cnt = dict.fromkeys(classes, 0)
    for t in targets:
        if t in t_cnt:
            t_cnt[t] += 1
    return max(t_cnt, key=t_cnt.get)
        

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    return vote(point_targets[k_nearest(x, points, k)], classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    pred = np.array([], dtype=int)
    for i in range(points.shape[0]):
        pred = np.append(pred, knn(points[i], remove_one(points, i), remove_one(point_targets, i), classes, k))
    return pred


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    point_predictions = knn_predict(points, point_targets, classes, k)
    n = point_targets.shape[0]
    correct_cnt = 0
    for true, pred in zip(point_targets, point_predictions):
        if true == pred:
            correct_cnt += 1
    return correct_cnt / n


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    point_predictions = knn_predict(points, point_targets, classes, k)
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.sum((point_predictions == classes[i]) & (point_targets == classes[j]))
    return matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    k = np.arange(1, points.shape[0] - 1)
    accuracies = np.zeros(k.shape)
    for i in range(k.shape[0]):
        accuracies[i] = knn_accuracy(points, point_targets, classes, k[i])
    return k[np.argmax(accuracies)]


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    point_predictions = knn_predict(points, point_targets, classes, k)
    colors = ['yellow', 'purple', 'blue']
    edge_colors = ['green' if pred == targ else 'red' for pred, targ in zip(point_predictions, point_targets)]
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edge_colors[i],
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...



def remove_one(points: np.ndarray, i: int):
    '''
    Removes the i-th from points and returns
    the new array
    '''
    return np.concatenate((points[0:i], points[i+1:]))