# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    n, k = X.shape[0], Mu.shape[0]
    out = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            out[i, j] = np.linalg.norm(X[i, :] - Mu[j, :])
    return out


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    n, k = dist.shape
    out = np.zeros((n, k), dtype=int)
    for i in range(n):
        out[i, np.argmin(dist, axis=1)[i]] = 1
    return out


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    return np.sum(R * dist) / R.shape[0]


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    k, f = Mu.shape
    out = np.zeros((k, f))
    for i in range(k):
        out[i, :] = np.sum(R[:, i][:, np.newaxis] * X, axis=0) / np.sum(R[:, i])
    return out


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    # !!! Your code here !!!
    Js = []
    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Js.append(determine_j(R, dist))
        Mu = update_Mu(Mu, X_standard, R)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    # !!! Your code here !!!
    return Mu, R, Js


def plot_j():
    X, y, c = load_iris()
    Mu, R, Js = k_means(X, 4, 10)
    plt.plot(Js)


def plot_multi_j():
    ks = [2, 3, 5, 10]
    X, y, c = load_iris()
    plt.subplots(2, 2)
    for i, k in enumerate(ks):
        Mu, R, Js = k_means(X, k, 10)
        plt.subplot(2, 2, i + 1)
        plt.plot(Js)
        plt.title(f'k = {k}')
    plt.tight_layout()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, R, Js = k_means(X, len(classes), num_its)
    clusters = np.argmax(R, axis=1)
    cluster_to_class = {}
    for cla in range(len(classes)):
        cluster_to_class[cla] = np.argmax(np.bincount(t[clusters == cla]))
    return np.array([cluster_to_class[clu] for clu in clusters])


def _iris_kmeans_accuracy():
    pass


def _my_kmeans_on_image():
    pixels, shape = image_to_numpy()
    return k_means(pixels, 7, 5)


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image)
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
