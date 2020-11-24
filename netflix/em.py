"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    non_null_index = X > 0
    dim = np.sum(non_null_index, axis=1)
    means = non_null_index[:, np.newaxis, :] * mixture.mu
    quadratic = np.sum((X[:, np.newaxis, :] - means) ** 2, axis=2) / (2 * mixture.var[np.newaxis, :])
    normalization = np.transpose(dim / 2 * np.log(2 * np.pi * mixture.var[:, np.newaxis]))
    f = np.log(mixture.p + 1e-16) - normalization - quadratic
    f_max = np.max(f, axis=1)
    log_likelihood = f_max + logsumexp(f.T - f_max, axis=0)
    post = np.exp(np.transpose(f.T - log_likelihood))
    return post, np.sum(log_likelihood).astype(float)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    non_null_index = X.astype(bool).astype(int)
    dim = np.sum(non_null_index, axis=1)
    reduced_post = np.transpose(non_null_index.T[:, np.newaxis, :] * post.T)
    no_update = np.sum(reduced_post, axis=0) < 1
    mu = np.sum(reduced_post * X[:, np.newaxis, :], axis=0) / (np.sum(reduced_post, axis=0) + 1e-16)
    mu[no_update] = mixture.mu[no_update]

    pi = np.sum(post, axis=0) / X.shape[0]

    var = np.sum(np.sum((X[:, np.newaxis, :] - mu) ** 2 * reduced_post, axis=2), axis=0) / np.transpose(
        np.sum(dim * post.T, axis=1))
    var[var < min_variance] = min_variance

    return GaussianMixture(mu=mu, var=var, p=pi)


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


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
