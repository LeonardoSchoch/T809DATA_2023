# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_hat = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_hat[:, i] = (X[:, i] - mean[i]) / std[i]
    return X_hat


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    plt.scatter(standardize(X[:, i]), standardize(X[:, j]))


def scatter_cancer():
    X, y = load_cancer()
    plt.subplots(5, 6, figsize=(18, 15))
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        scatter_standardized_dims(X, 0, i)
    plt.tight_layout()


def plot_pca_components():
    X, y = load_cancer()
    pca = PCA(n_components=30)
    pca.fit_transform(standardize(X))
    plt.subplots(1, 3)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(pca.components_[i, :])
        plt.title(f"PCA {i + 1}")
    plt.tight_layout()


def plot_eigen_values():
    X, y = load_cancer()
    pca = PCA(n_components=30)
    pca.fit_transform(standardize(X))
    plt.plot(pca.explained_variance_)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()


def plot_log_eigen_values():
    X, y = load_cancer()
    pca = PCA(n_components=30)
    pca.fit_transform(standardize(X))
    plt.plot(np.log10(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()


def plot_cum_variance():
    X, y = load_cancer()
    pca = PCA(n_components=30)
    pca.fit_transform(standardize(X))
    per_var = []
    for i in range(30):
        per_var.append(np.sum(pca.explained_variance_[:i]) / np.sum(pca.explained_variance_))
    plt.plot(per_var)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
