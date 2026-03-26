"""
Utility functions for the State-Space Analysis of Spike Correlations.

Model selection (AIC) and credible interval computation on fitted EMData.

---

This code implements approximate inference methods for State-Space Analysis of
Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012). It is an extension of
the existing code from repository <https://github.com/tomxsharp/ssll> (For
Matlab Code refer to <http://github.com/shimazaki/dynamic_corr>). We
acknowledge Thomas Sharp for providing the code for exact inference.

In this library are additional methods provided to perform the State-Space
Analysis approximately. This includes pseudolikelihood, TAP, and Bethe
approximations. For details see: <http://arxiv.org/abs/1607.08840>

Copyright (C) 2016

Authors of the extensions: Christian Donner (christian.donner@bccn-berlin.de)
                           Hideaki Shimazaki (shimazaki@brain.riken.jp)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import itertools
import numpy
from scipy.special import comb
from scipy.stats import norm


def compute_AIC(emd, alternate_llk=None):
    """
    Computes the Akaike Information Criterion based on the optimization of the hyperparameters.

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param alternate_llk: float or None
    If None, uses the current log likelihood obtained from the model. If not None uses alternate_llk instead.
    :return: float, float
    The Akaike Information Criterion and the number of free hyperparameters in the model used.
    """
    llk = emd.marg_llk(emd)

    if alternate_llk is not None:
        llk = alternate_llk

    d = emd.D
    if type(emd.state_cov_0) == float or type(emd.state_cov_0) == int:
        k_q = 1
    elif emd.state_cov_0 is None:
        k_q = 0
    elif emd.state_cov_0.shape == (d,):
        k_q = d
    else:
        k_q = d * (d - 1) / 2

    if emd.state_ar_0 is None:
        k_f = 0
    else:
        k_f = d * d

    k = k_f + k_q

    aic = - 2 * llk + 2 * k
    return aic, k


def compute_bounds(emd, threshold):
    """
    Computes the credible interval based on the threshold provided.
    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param threshold: int
    How strict the credible interval is
    :return: numpy.ndarray
    Array of shape (T, D, 2) with lower and upper bounds.
    """
    T = emd.spikes.shape[0]
    D = emd.D
    theta = emd.theta_s
    quantile_p = 0.5 + threshold/2 * 0.01
    quantile_m = 0.5 - threshold/2 * 0.01
    if emd.sigma_s.shape == (T, D):
        sigma = emd.sigma_s
    else:
        sigma = numpy.zeros((T, D))
        for t in range(T):
            sigma[t] = numpy.diag(emd.sigma_s[t])

    bounds = numpy.empty([theta.shape[0], theta.shape[1], 2])
    bounds[:, :, 0] = norm(loc=theta[:, :], scale=numpy.sqrt(numpy.abs(sigma[:, :]))).ppf(quantile_m)
    bounds[:, :, 1] = norm(loc=theta[:, :], scale=numpy.sqrt(numpy.abs(sigma[:, :]))).ppf(quantile_p)
    return bounds


def get_neuron_pairs(spikes_timing, neurons, N=None):
    """
    Computes the all neuron pairs and how many there are

    :param spikes_timing: numpy.ndarray
    An array with lists of spike times at indexes of format [neurons][stimulus][trials]
    :param neurons: numpy.ndarray
    Indexes of neurons of interest or 'all' to consider all neurons.
    :param N: int
    the number of neurons to consider. (This is not used as of now, neurons determines which neruons to use.
    :return: list, int
    returns all the pairs of neurons and how many combinations there are
    """
    if type(neurons) == str:
        n_indexes_all = numpy.arange(spikes_timing.shape[0])
        n_indexes = numpy.random.choice(n_indexes_all, N, replace=False)
        n_pairs = itertools.combinations(n_indexes,2)
        combs = comb(N,2)
    else:
        n_pairs = itertools.combinations(neurons,2)
        combs = comb(len(neurons),2)


    return list(n_pairs),combs


def get_spikes_subset(spikes, neurons='rand', N=8):
    """
    Generates a subset with given constraints from the spikes provided

    :param spikes: numpy.ndarray
    A binary spikes array of format [time][trials][neurons]
    :param neurons: String or list
    The neurons to include in the spike subset. This is either 'top' for the N top firing neurons,
    'rand' for N random neurons or a list of neurons index wich represent the ones of interest.
    :param N: int
    The number of neurons to take into account if neurons is 'rand' or 'top'.
    :return: numpy.ndarray
    A spike subset of of spikes.
    """
    if neurons == 'rand':
        neuron_indexes = numpy.arange(spikes.shape[2])
        selected_neurons = numpy.random.choice(neuron_indexes, N, replace=False)

    elif neurons == 'top':
        rate = numpy.mean(numpy.mean(spikes[:, :, :], axis=0), axis=0)
        neurons_sorted = numpy.argsort(rate)
        selected_neurons = neurons_sorted[-N:]
    else:
        selected_neurons = neurons
    spikes_subset = spikes[:,:,selected_neurons]
    return spikes_subset


def shuffle_spikes(spikes):
    """
    Assigns neurons from different trials together

    :param spikes: numpy.ndarray
    A binary spikes array of format [time][trials][neurons]
    :return: numpy.ndarray
    A shuffled binary spikes array
    """
    T, R, N = spikes.shape
    shuffled_spikes = numpy.zeros(spikes.shape)
    numpy.random.seed(1)
    for n in range(N):
        r_idx = numpy.random.permutation(numpy.arange(R))
        shuffled_spikes[:, :, n] = spikes[:, r_idx, n]
    return shuffled_spikes
