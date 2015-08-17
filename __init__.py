"""
Master file of the State-Space Analysis of Spike Correlations.
 
TODO some sort of automatic versioning system
TODO complete the utilities module
 
---
 
State-Space Analysis of Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012)
Copyright (C) 2014  Thomas Sharp (thomas.sharp@riken.jp)
 
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
 
import container
import exp_max
import probability
import max_posterior
import synthesis
import transforms
import pseudo_likelihood
import mean_field
import energies
import bethe_approximation
 
 
def run(spikes, order, window=1, map_function='nr', lmbda=200, max_iter=100,
        param_est='exact', param_est_eta='exact'):
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
    :param string map_function:
        Name of the function to use for maximum a-posterior estimation of the
        natural parameters at each timestep. Refer to max_posterior.py.
    :param float lmdbda:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix.
    :param int max_iter:
        Maximum number of iterations for which to run the EM algorithm.
    :param str param_est:
        Parameter whether exact likelihood ('exact') or pseudo likelihood
        ('pseudo') should be used
 
    :returns:
        Results encapsulated in a container.EMData object, containing the
        smoothed posterior probability distributions of the natural parameters
        of the spike-train interactions at each timestep, conditional upon the
        given spikes.
    """
    # Ensure NaNs are caught
    numpy.seterr(invalid='raise')
    # Get Number of cells
    N = spikes.shape[2]
    # Initialise the coordinate-transform maps
    if param_est == 'exact':
        transforms.initialise(N, order)
        map_func = max_posterior.functions[map_function]
        #marg_llk_fun = probability.log_marginal
    elif param_est == 'pseudo':
        pseudo_likelihood.compute_Fx_s(spikes, order)
        map_func = pseudo_likelihood.functions[map_function]
        mean_field.create_eta_FI_map(N, order)
    elif param_est == 'mf':
        mean_field.create_eta_FI_map_second_order(N, order)
        map_func = mean_field.functions[map_function]
    elif param_est == 'bethe_BP':
        mean_field.create_eta_FI_map(N, order)
        map_func = bethe_approximation.conjugate_gradient_BP

    if param_est_eta == 'exact':
        marg_llk_fun = probability.log_marginal
    elif param_est_eta == 'mf':
        marg_llk_fun = mean_field.log_marginal
    elif param_est_eta == 'bethe_BP':
        marg_llk_fun = bethe_approximation.log_marginal_BP
    elif param_est_eta == 'bethe_CCCP':
        marg_llk_fun = bethe_approximation.log_marginal_CCCP
    elif param_est_eta == 'bethe_hybrid':
        pass
    # Initialise the EM-data container
    emd = container.EMData(spikes, order, window, map_func, marg_llk_fun, lmbda)
    # Solves backward problem. For zero rates in the beginning small number is added
    if emd.order == 2:
        try:
            y_init = numpy.mean(emd.y, axis=0)
            y_init[y_init == 0] = numpy.spacing(1)
            emd.theta_o[0] = mean_field.backward_problem(y_init, emd.N, 'TAP')
        except numpy.linalg.linalg.LinAlgError:
            emd.theta_o[0][:emd.N] = energies.compute_ind_theta(y_init[:emd.N])
 
    # Set up loop guards for the EM algorithm
    lmp = -numpy.inf
    lmc = emd.marg_llk(emd)
    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > exp_max.CONVERGED):
        print emd.convergence
        # Perform EM
        exp_max.e_step(emd)
        exp_max.m_step(emd)
        # Update previous and current log marginal values
        lmp = lmc
        lmc = emd.marg_llk(emd)
        emd.mllk = lmc
        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = numpy.absolute((lmp - lmc) / lmp)
    return emd