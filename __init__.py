"""
Master file of the State-Space Analysis of Spike Correlations.

TODO some sort of automatic versioning system

Changes: All adjustments to incorporate the approximation methods
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

---

This code was updated to enable the user to specify the mean and the
covariance of the prior and to enable or disable the hyperparameters
optimization (m-step).

Copyright (C) 2018

Authors of the update: Jimmy Gaudreault (jimmy.gaudreault@polymtl.ca)
                       Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)

---

This code was extended to enable users to optimize the autoregressive parameter
and noise covariance (a scalar or a diagonal and full matrix) in a state model.

Copyright (C) 2019

Author of the extensions: Magalie Tatischeff (magalietati@gmail.com)

---

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
import pdb
import sys
from tqdm import tqdm

__version__ = '0.1.0'

import container
import exp_max
import probability
import max_posterior
import synthesis
import transforms
import pseudo_likelihood
import mean_field
import energies
import thermodynamics
import util
import bethe_approximation


def run(spikes, order=2, window=1, map_function='cg', \
        state_cov=0.01, state_ar=None, max_iter=100,
        param_est='exact', param_est_eta='exact',\
        theta_o=0, sigma_o=0.1, mstep=True, EM_Info=True,
        stationary=False):
    """
    Master-function of the State-Space Analysis of Spike Correlation package.
    Uses the expectation-maximisation algorithm to find the probability
    distributions of natural parameters of spike-train interactions over time.
    Calls slave functions to perform the expectation and maximisation steps
    repeatedly until the data likelihood reaches an asymptotic value.

    Note that the execution of some slave functions to this master function are
    of exponential complexity with respect to the `order' parameter.

    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :param float state_cov:
        The covariance matrix of state transition noise Q. Depending the choice 
        of the matrix form of Q, the code execute different optimization strategies.
        If state_cov is a list and scalar, Q = state_cov I. state_cov is updated.
        If state_cov is a float vecotr, Q = diag(state_cov), and the diagonal is updated.
        If state_cov is a float matrix, Q = state_cov, and the whoel matrix is updated.
        If state_cov is a list [lambda_1, lambda2], then lambda_1 is asigned to the 
        first order parameters and lambda2 the second/higher order parameters:
        Q = [lambda_1,...,lambda_1, lambda2,..,lambda2]
        and lambda_1 and lambda2 are updated.
    :param float state_ar:
        The matrix of the first-order autoregressive parameter F in the state model.
    :param string map_function:
        Name of the function to use for maximum a-posteriori estimation of the
        natural parameters at each timestep. Refer to max_posterior.py.
    :param float lmbda1:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the first order theta parameters.
    :param float lmbda2:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the second order theta parameters.
    :param int max_iter:
        Maximum number of iterations for which to run the EM algorithm.
    :param str param_est:
        Parameter whether exact likelihood ('exact') or pseudo likelihood
        ('pseudo') should be used
    :param str param_est_eta:
        Eta parameters are either calculated exactly ('exact'), by mean
        field TAP approximation ('mf'), or Bethe approximation (belief
        propagation-'bethe_BP', CCCP-'bethe_CCCP', hybrid-'bethe_hybrid')
    :param stationary:
        To fit stationary model. Set 'all' to have stationary thetas. (
        Default='None')
    :param numpy.ndarray theta_o:
        Prior mean at the first time bin (one-step predictor)
    :param numpy.ndarray sigma_o:
        Prior covariance at the first time bin (one-step predictor)
    :param boolean mstep:
        The m-step of the EM algorithm is performed only if this parameter
        is true. (Default='True')
    :param bool stationary:
        If True, fit a time-independent model by pooling all T×R
        observations into a single time step. Spikes (T, R, N) are reshaped
        to (1, T*R, N) and state_cov is forced to None (Q=0, no update).
        Default is False.

    :returns:
        Results encapsulated in a container.EMData object, containing the
        smoothed posterior probability distributions of the natural parameters
        of the spike-train interactions at each timestep, conditional upon the
        given spikes.
    """
    # Ensure NaNs are caught
    numpy.seterr(invalid='raise')

    # Stationary mode: pool all time bins as trials, fit single time step
    if stationary:
        T, R, N = spikes.shape
        spikes = spikes.reshape(1, T * R, N)
        state_cov = None
        state_ar = None

    # Get Number of cells
    N = spikes.shape[2]
    # Initialise the EM-data container
    emd = container.EMData(spikes, order, window, param_est, param_est_eta,
                           map_function, state_cov, state_ar, theta_o, sigma_o)

    # Set up loop guards for the EM algorithm
    lmc = emd.marg_llk(emd)
    emd.mllk_list = []
    
    # Create progress bar if EM_Info is True
    if EM_Info == True:
        pbar = tqdm(total=max_iter, desc='EM', 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Conv: {postfix}')
    
    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > emd.CONVERGED):
        if EM_Info == True:
            # Update progress bar with convergence info
            pbar.set_postfix_str(f'{emd.convergence:.6f} > {emd.CONVERGED:.6f}')
            pbar.update(1)
            
        # Perform EM
        exp_max.e_step(emd)
        if mstep == True:
            exp_max.m_step(emd)
        # Update previous and current log marginal values
        lmp = lmc
        lmc = emd.marg_llk(emd)
        emd.mllk_list.append(lmc)
        emd.mllk = lmc

        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = (lmp - lmc) / lmp

    if EM_Info == True:
        pbar.close()
        print('Log marginal likelihood = %.6f' % (emd.mllk))
    return emd
