"""
This file contains the GMM from which synthetic images are sampled.
"""

from AxonDeepSeg import ads_utils
from pathlib import Path
import numpy as np


def sample_distribution(distribution, hyperparameters):
    """
    This function samples values from a uniform or normal distribution.
    :param distribution: either 'normal' or 'uniform'
    :param hyperparameters: given by [a,b] where:
    a : mean for normal distr or lower bound for uniform distr
    b : std for normal distr or upper bound for uniform distr
    :return: float sampled from the given distribution
    """
    rng = np.random.default_rng()
    a = hyperparameters[0]
    b = hyperparameters[1]
    if distribution is 'uniform':
        return rng.uniform(a, b)
    elif distribution is 'normal':
        return rng.normal(a, b)
    else:
        print(f'WARNING: No distribution found for {distribution}.')