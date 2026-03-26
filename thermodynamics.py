"""
Thermodynamic properties with credible intervals for the State-Space Ising Model.

Extends energies.py with Monte Carlo uncertainty quantification: entropy,
heat capacity, and probability of silence, each with credible-interval bounds
computed by sampling from the posterior theta distribution.

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
import numpy
import itertools
import energies
import transforms


def compute_entropy_b(emd, samples, threshold):
    """
    Computes the entropy of the model, the bounds compted based on the threshold,
    the pairwise contribution and its bounds.

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param samples: int
    number of sampled thetas to use when computing the bounds.
    :param threshold: int
    Decides how strictly the credible interval is
    :return: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    The entropy of the model, the bounds compted based on the threshold, the pairwise contribution and its bounds.
    """
    S0 = emd.N * numpy.log(2)

    N = emd.N
    O = emd.order
    thetas = get_theta_samples(emd, samples)

    S_pair_all = numpy.zeros((emd.T, samples))
    S_ratio_all = numpy.zeros((emd.T, samples))
    for n in range(samples):

        theta = thetas[:, :, n]
        eta, emd.eta_sampled = energies.compute_eta(theta, N, O)
        psi = energies.compute_psi(theta, N, O)
        eta1 = eta[:, :N]
        theta1 = energies.compute_ind_theta(eta1)
        psi1 = energies.compute_ind_psi(theta1)
        S1 = energies.compute_entropy(theta1, eta1, psi1, 1)
        eta = eta
        S_pair_all[:, n] = energies.compute_entropy(theta, eta, psi, 2)
        S_ratio_all[:, n] = (S1 - S_pair_all[:, n]) / (S0 - S_pair_all[:, n]) * 100

    S_pair = S_pair_all[:, 0]
    S_ratio = S_ratio_all[:, 0]
    disregard = int((samples - threshold / 100.0 * samples) / 2)
    S_pair_all = numpy.sort(S_pair_all, axis=1)
    S_ratio_all = numpy.sort(S_ratio_all, axis=1)
    return S_pair, S_pair_all[:, [disregard, -disregard - 1]], S_ratio, S_ratio_all[:, [disregard, -disregard - 1]]


def compute_c_b(emd, samples, threshold, beta=1):
    """
    Computes he heat capacity and the bounding heat capacities based on the threshold.
    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param samples: int
    number of sampled thetas to use when computing the bounds.
    :param threshold:  int
    Decides how strictly the credible interval is
    :param beta: float
    the value of beta used to slightly vary the theta parameters.
    :return: numpy.ndarray, numpy.ndarray
    The heat capacity and the bounds based on the threshold
    """
    T, D = emd.theta_s.shape

    thetas = beta * get_theta_samples(emd, samples)

    psi = numpy.zeros(T)
    C = numpy.zeros((T, samples))
    epsilon = 1e-3
    for n in range(samples):
        for t in range(T):
            psi[t] = transforms.compute_psi(thetas[t, :, n])
            tmp1 = transforms.compute_psi(thetas[t, :, n] * (1 + epsilon))
            tmp2 = transforms.compute_psi(thetas[t, :, n] * (1 - epsilon))
            c = tmp1 - 2 * psi[t] + tmp2
            d = epsilon ** 2
            C[t, n] = c / d
    C_map = C[:, 0]
    disregard = int((samples - threshold / 100.0 * samples) / 2)
    C = numpy.sort(C, axis=1)
    return C_map, C[:, [disregard, -disregard - 1]]

def compute_p_silence_b(emd, samples, threshold):
    """
    Computes the probability that all neurons are silent(p_silence) and the bounding p_silence
    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param samples: int
    number of sampled thetas to use when computing the bounds.
    :param threshold: int
    Decides how strictly the credible interval is
    :return: numpy.ndarray, numpy.ndarray
    The probability that all neurons are silent(p_silence) and the bounding p_silence

    """
    thetas = get_theta_samples(emd, samples)
    p_silence_all = numpy.zeros((emd.T, samples))
    for i in range(samples):
        psi = energies.compute_psi(thetas[:, :, i], emd.N, emd.order)
        p_silence_all[:, i] = numpy.exp(-psi)
    p_silence = p_silence_all[:, 0]
    p_silence_all = numpy.sort(p_silence_all, axis=1)
    disregard = int((samples - threshold / 100.0 * samples) / 2)
    p_silence_bounds = p_silence_all[:, [disregard, -disregard - 1]]

    return p_silence, p_silence_bounds


def compute_c(emd, beta=1):
    """
    Computes the heat capacity

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param beta: float
    the value of beta used to slightly vary the theta parameters.
    :return: numpy.ndarray, numpy.ndarray
    The heat capacity (if you wants bounding heat capacity, use compute_c_b)
    """
    psi = numpy.zeros(emd.T)
    C = numpy.zeros(emd.T)
    epsilon = 1e-3
    for t in range(emd.T):
        theta = beta * emd.theta_s[t, :]
        psi[t] = transforms.compute_psi(theta)
        tmp1 = transforms.compute_psi(theta * (1 + epsilon))
        tmp2 = transforms.compute_psi(theta * (1 - epsilon))
        c = tmp1 - 2 * psi[t] + tmp2
        d = epsilon ** 2
        C[t] = c / d

    return C

def get_c_beta(emd, num, span=[0.25, 2]):
    """
    Computes the heat capacity num times by multiplying theta by equaly spaced betas in span)

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param num: int
    The number of heat capacities to compute, all with different betas.
    :param span: list
    The span for betas
    :return: numpy.ndarray
    The heat capacities computed with num different betas.
    """
    betas = numpy.linspace(span[0], span[1], num)
    c_betas = numpy.zeros((len(betas), emd.T))
    for i, beta in enumerate(betas):
        c_betas[i, :] = compute_c(emd, beta)
    return c_betas


def get_entropy(emd):
    """
    Computes the entropy of the network(S_pair) and the pairwise contributions (S_ratio)

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :return: numpy.ndarray
    The entropy and pairwise contributions
    """
    energies.get_energies(emd)
    S_pair = emd.S2
    S_ind = emd.S1
    S0 = emd.N * numpy.log(2)
    S_ratio = (S_ind - S_pair) / (S0 - S_pair) * 100

    return S_pair, S_ratio


def get_theta_samples(emd, size):
    """
    Gets size number of thetas sampled form the theta distribution

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param size: int
    The number of sample you wish to generate
    :return: numpy.ndarray
    size number of thetas sampled form the theta distribution
    """

    T, D = emd.theta_s.shape
    thetas = numpy.zeros((T, D, size))
    thetas[:, :, 0] = emd.theta_s
    s = emd.sigma_s

    if s.shape != (T, D):
        for t in range(T):
            theta = emd.theta_s[t]
            sigma = s[t]
            thetas[t, :, 1:] = numpy.random.multivariate_normal(theta, sigma, size - 1).T

    else:
        for t, d in itertools.product(range(T), range(D)):
            theta = emd.theta_s[t, d]
            sigma = numpy.sqrt(s[t, d])
            thetas[t, d, 1:] = numpy.random.normal(theta, sigma, size - 1)

    return thetas
