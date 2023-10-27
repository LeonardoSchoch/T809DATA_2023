from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    # if x < -100:
    #     return 0.0
    # return 1 / (1 + np.exp(-x))
    return np.where(x < -100, 0.0, 1 / (1 + np.exp(-x)))
    

def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    wei = np.dot(x, w)
    return wei, sigmoid(wei)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.append(1.0, x)
    a1, y1 = perceptron(z0, W1)

    z1 = np.append(1.0, y1)
    a2, y2 = perceptron(z1, W2)

    y = y2

    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    # 1.
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    delta_k = y - target_y
    delta_j = d_sigmoid(a1) * np.dot(W2[1:], delta_k)
    
    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)
    
    for i in range(dE1.shape[0]):
        dE1[i, :] = delta_j * z0[i]

    for k in range(dE2.shape[0]):
        dE2[k, :] = delta_k * z1[k]
    
    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    N = X_train.shape[0]
    E_total = np.array([])
    misclassification_rate = np.array([])
    
    for _ in range(iterations):

        guesses = np.zeros((t_train.shape[0], K))
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)

        E = 0
        for i in range(N):
            x = X_train[i, :]
            t = np.zeros(K)
            t[t_train[i]] = 1

            y, dE1, dE2 = backprop(x, t, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            guesses[i] = y

            E += -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / N
            
        W1 = W1 - eta * dE1_total / N
        W2 = W2 - eta * dE2_total / N

        E_total = np.append(E_total, E)

        misclassification_rate = np.append(misclassification_rate, \
                                           np.sum(np.argmax(guesses, axis=1) != t_train) / N)

        last_guesses = [np.argmax(g) for g in guesses]

    return W1, W2, E_total, misclassification_rate, last_guesses



def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = X.shape[0]
    guesses = np.array([])
    for i in range(N):
        y, z0, z1, a1, a2 = ffnn(X[i, :], M, K, W1, W2)
        guesses = np.append(guesses, np.argmax(y))
    return guesses


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    pass