import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    return np.exp(-np.power(x-mu, 2)/(2*np.power(sigma, 2)))/ \
        np.sqrt(2*np.pi*np.power(sigma, 2))

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    xs = np.linspace(x_start, x_end, num=500)
    plt.plot(xs, normal(xs, sigma, mu))

def plot_three_normals():
    # Part 1.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    y = np.array([])
    for x_i in x:
        y = np.append(y, np.sum(np.array(weights)/np.sqrt(2*np.pi*np.power(np.array(sigmas), 2))* \
            np.exp(-np.power(x_i-np.array(mus), 2)/(2*np.power(np.array(sigmas), 2)))))
    return y

def plot_normal_mixture(sigmas: list, mus: list, weights: list, x_start: float, x_end: float):
    xs = np.linspace(x_start, x_end, num=500)
    plt.plot(xs, normal_mixture(xs, sigmas, mus, weights))


def compare_components_and_mixture():
    # Part 2.2
    mus = [0, -0.5, 1.5]
    sigmas = [0.5, 1.5, 0.25]
    weights = [1/3, 1/3, 1/3]
    x_start = -5
    x_end = 5

    plot_normal(sigmas[0], mus[0], x_start, x_end)
    plot_normal(sigmas[1], mus[1], x_start, x_end)
    plot_normal(sigmas[2], mus[2], x_start, x_end)
    plot_normal_mixture(sigmas, mus, weights, x_start, x_end)

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    mixture_samples = np.random.multinomial(n_samples, weights)
    samples = np.array([])
    for sigma, mu, num_samples in zip(sigmas, mus, mixture_samples):
        samples = np.append(samples, np.random.normal(mu, sigma, num_samples))
    return samples

def plot_mixture_and_samples():
    # Part 3.2
    sigmas = [0.3, 0.5, 1]
    mus = [0, -1, 1.5]
    weights = [0.2, 0.3, 0.5]
    n_samples = [10, 100, 500, 1000]
    x_start = -10
    x_end = 10

    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples[0])
    plt.subplot(221)
    plot_normal_mixture(sigmas, mus, weights, x_start, x_end)
    plt.hist(samples, 100, density=True)
    plt.ylim(0, 2.5)
    
    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples[1])
    plt.subplot(222)
    plot_normal_mixture(sigmas, mus, weights, x_start, x_end)
    plt.hist(samples, 100, density=True)
    plt.ylim(0, 2.5)
    
    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples[2])
    plt.subplot(223)
    plot_normal_mixture(sigmas, mus, weights, x_start, x_end)
    plt.hist(samples, 100, density=True)
    plt.ylim(0, 2.5)
    
    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples[3])
    plt.subplot(224)
    plot_normal_mixture(sigmas, mus, weights, x_start, x_end)
    plt.hist(samples, 100, density=True)
    plt.ylim(0, 2.5)

    plt.tight_layout()

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    pass