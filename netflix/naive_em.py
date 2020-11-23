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
    dim = X.shape[1]
    weight = np.sum(post, axis=0) / X.shape[0]
    mean = np.transpose(np.transpose(np.dot(np.transpose(post), X)) / np.sum(post, axis=0))
    var = np.sum(np.sum((X[:, np.newaxis, :] - mean) ** 2, axis=2) * post, axis=0) / (dim * np.sum(post, axis=0))
    return GaussianMixture(mu=mean, var=var, p=weight)


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
    precision = 1e-6
    error = 2 * precision
    posterior, prev_log_likelihood = estep(X, mixture)
    mixture = mstep(X, posterior)
    while error > precision:
        posterior, log_likelihood = estep(X, mixture)
        mixture = mstep(X, posterior)
        error = (log_likelihood - prev_log_likelihood) / abs(log_likelihood)
        prev_log_likelihood = log_likelihood

    return mixture, posterior, prev_log_likelihood
