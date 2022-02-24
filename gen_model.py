"""
This file contains the GMM from which synthetic images are sampled.
"""

from AxonDeepSeg import ads_utils
from pathlib import Path
import numpy as np
import pandas as pd


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
    if distribution == 'uniform':
        return rng.uniform(a, b)
    elif distribution == 'normal':
        return rng.normal(a, b)
    else:
        print(f'WARNING: No distribution found for "{distribution}".')


class ImageGenerator:

    def __init__(self):
        """
        This constructor attemps to read a 'priors.csv' file containing the
        priors computed on the dataset for every class.
        """
        measures = pd.read_csv('priors.csv', index_col=0)
        mean_measures = measures.mean(axis=0)
        std_measures = measures.std(axis=0)
        # find upper/lower bound for uniform distribution of priors
        self.priors_bounds = []
        for label in ['bg', 'my', 'ax']:
            mean_str = 'mean_' + label
            std_str = 'std_' + label
            a_mean = mean_measures[mean_str] - std_measures[mean_str]
            b_mean = mean_measures[mean_str] + std_measures[mean_str]
            a_std = mean_measures[std_str] - std_measures[std_str]
            b_std = mean_measures[std_str] + std_measures[std_str]
            class_priors = [[a_mean, b_mean], [a_std, b_std]]
            self.priors_bounds.append(class_priors)
        self.print_priors()

        # generator for the gaussian parameters
        self.model_params_generator = self._build_model_params()

    def generate_image(self, label_map):
        """
        This method will generate an image conditioned on the given label map
        :param label_map: matrix of same dimensions as image
        uses 0,1 and 2 for background, myelin and axon, respectively
        :return: synthetic image
        """
        gmm_parameters = next(self.model_params_generator)
        out_shape = label_map.shape
        img = np.zeros(out_shape)
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                label = label_map[i,j]
                img[i][j] = sample_distribution('normal', gmm_parameters[label])
        return img


    def _build_model_params(self):
        """This method returns a generator for the gmm parameters."""
        while True:
            model_params = []
            for l, label in enumerate(['bg','my','ax']):
                class_params = []
                for p, param in enumerate(['mean','std']):
                    sample = sample_distribution('uniform', self.priors_bounds[l][p])
                    class_params.append(sample)
                model_params.append(class_params)
            yield model_params

    def print_priors(self):
        if self.priors_bounds is None:
            print('Empty priors')
        else:
            pb = self.priors_bounds
            print(f'\tbackground mean ~ U({pb[0][0][0]:.2f}, {pb[0][0][1]:.2f})')
            print(f'\tbackground std  ~ U({pb[0][1][0]:.2f}, {pb[0][1][1]:.2f})')
            print(f'\tmyelin mean ~ U({pb[1][0][0]:.2f}, {pb[1][0][1]:.2f})')
            print(f'\tmyelin std  ~ U({pb[1][1][0]:.2f}, {pb[1][1][1]:.2f})')
            print(f'\taxon mean ~ U({pb[2][0][0]:.2f}, {pb[2][0][1]:.2f})')
            print(f'\taxon std  ~ U({pb[2][1][0]:.2f}, {pb[2][1][1]:.2f})')