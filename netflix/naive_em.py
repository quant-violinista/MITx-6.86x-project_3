"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    dim = X.shape[1]
    exponents = np.sum((X[:, np.newaxis, :] - mixture.mu) ** 2, axis=2) / (2 * mixture.var[np.newaxis, :])
    weighted_likelihoods = np.transpose(1 / ((2 * np.pi * mixture.var[:, np.newaxis]) ** (dim / 2))) * np.exp(
        -exponents) * mixture.p
    post = np.transpose(weighted_likelihoods.T / np.sum(weighted_likelihoods, axis=1))
    log_likelihood = np.sum(np.log(np.sum(weighted_likelihoods, axis=1))).astype(float)
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    weight = np.sum(self.posteriors, axis=1) / len(self.x)
    mean = np.sum(self.posteriors * self.x, axis=1) / np.sum(self.posteriors, axis=1)
    var = np.sum((np.transpose(np.tile(self.x, (mean.shape[0], 1)).T - mean) ** 2) * self.posteriors,
                 axis=1) / np.sum(self.posteriors, axis=1)

    self.theta = pd.DataFrame({'weight': weight, 'mean': mean, 'var': var})
    return


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
