"""
Functions for generating synthetic spike data.

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

import transforms


def generate_thetas(N, O, T):
    D = transforms.compute_D(N, O)
    MU = numpy.tile(-2,(T, D))
    MU[:,N:] = 0
    # Create covariance matrix
    X = numpy.tile(numpy.arange(T),(T,1))
    K = .5*numpy.exp( -.001 *.5 * (X - X.transpose())**2 )
    # Generate Gaussian processes
    L = numpy.linalg.cholesky(K + 1e-12* numpy.eye(T) )
    theta = MU + numpy.dot(L, numpy.random.randn(T, D))
    return theta


def generate_stationary_thetas(N, O, T):
    th1, th2 = -3., 10.
    D = transforms.compute_D(N, O)
    th = numpy.zeros([T,D])
    th[:,:N] = th1
    th[:,N:] = th2/N*(1 + 0.5*numpy.sqrt(N)*numpy.random.randn(T,D-N))
    idx = numpy.triu_indices(N,1)
    theta_array = numpy.zeros([N,N])
    theta_array[idx[0],idx[1]] = th[0,N:]
    theta_array[idx[1],idx[0]] = th[0,N:]
    mean_thetas = numpy.mean(theta_array, axis=0)
    theta_array -= numpy.tile(mean_thetas, [N, 1])
    th[:,N:] = theta_array[idx[0],idx[1]]
    return th


def generate_spikes(p, R, seed=None):
    """
    Draws spike patterns for each of `R' trial runs from the probability mass
    specified in `p'. `p' must have T rows, one independent probability mass for
    each timestep, and 2^C columns, where C is the number of cells (maximum of
    8) involved in the spike pattern.

    :param numpy.ndarray p:
        Probability mass of spike patterns for each timestep.
    :param int R:
        Number of spike patterns to generate for each timestep.

    :returns:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c, as a
        numpy.ndarray
    """
    # Set metadata
    T, N = p.shape[0], numpy.int(numpy.log2(p.shape[1]))
    # Initialise random seed
    numpy.random.seed(seed)
    # Set up spike patterns
    fx = transforms.enumerate_patterns(N)
    # Set up the output array (time, trials, cells)
    T, C = p.shape[0], numpy.log2(p.shape[1])
    spikes = numpy.zeros((T, R, N))
    # Iterate over each probability
    for i in range(T):
        # Draw random values from the probability distribution
        idx = random_weighted(p[i], R)
        # Extract spike patterns for each trial
        spikes[i,:,:] = fx[idx,:]

    return spikes


def generate_spikes_gibbs(theta, N, O, R, **kwargs):
    """Generates spike trains for the model given the thetas with
    `Gibbs-Sampling <https://en.wikipedia.org/wiki/Gibbs_sampling>`_.

    :param numpy.ndarray theta:
        parameters used for sampling for each time bin
    :param int N:
        Number of cells
    :param int O:
        Order of interaction
    :param int R:
        Number of runs that are generated.

    :returns:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c, as a
        numpy.ndarray
    """
    # Set numpy seed (should be removed at some point)
    seed = kwargs.get('seed', None)
    numpy.random.seed(seed)
    # Set pre-trials
    pre_R = kwargs.get('pre_n', 100)
    # Get number of bins
    T = theta.shape[0]
    # Initialize array for spike data
    X = numpy.zeros([T, R+pre_R, N], dtype=numpy.uint8)
    # Gets subsets
    subsets = transforms.enumerate_subsets(N, O)
    # Get number of natural parameters
    D = len(subsets)
    # Initialize subset map
    subset_map = numpy.zeros([D, N])
    # Get map of relevant patterns (d,c)
    for i in range(len(subsets)):
        subset_map[i, subsets[i]] = 1
    # Count how many cells must be active for each theta
    subset_count = numpy.sum(subset_map, axis=1)
    # Draw random numbers from uniform distribution
    rand_numbers = numpy.random.rand(T, R+pre_R, N)
    # Iterate over all time bins
    for t in range(T):
        # Iterate through all Runs
        cur_theta = theta[t]
        for l in range(1, R+pre_R):
            # Iterate through all cells
            for i in range(N):
                # Construct pattern from trial before and
                # from neurons that have been seen in this trial
                pattern = numpy.array([numpy.hstack([X[t, l, :i],
                                                     X[t, l-1, i:]])])
                # set x^(i,t) to "1" and compute f(X) for those
                pattern[:, i] = 1
                fx1 = (numpy.dot(pattern, subset_map.T) == subset_count)[0]
                # Set x^(i,t) to "0" and compute f(X) for those
                pattern[:, i] = 0
                fx0 = (numpy.dot(pattern, subset_map.T) == subset_count)[0]
                # compute p( x^(i,l) = 1 || X^(1:i-1,t),X^(i+1:N,l-1) )
                prob_spike = 0.5*(1 + numpy.tanh(0.5*(numpy.sum(cur_theta[fx1])
                                                - numpy.sum(cur_theta[fx0]))))
                # if smaller than probability X^(i,l) -> 1
                X[t, l, i] = numpy.greater_equal(prob_spike,
                                                 rand_numbers[t, l, i])
    # Return spike data
    return X[:, pre_R:, :]


def random_weighted(p, R):
    """
    Draws `R' integers from the probability mass over the integers `p'.

    :param numpy.ndarray p:
        Probability mass.
    :param int R:
        Sample size to draw from `p'.

    :returns:
        `R' random numbers drawn from distribution `p', as a numpy.ndarray.
    """
    # Take a cumulative sum of the probability mass
    cs = numpy.cumsum(p)
    # Draw uniform random numbers for each timestep or trial
    rnd = numpy.random.random(R)
    # For each random value, find the index of the first weight above it
    idx = numpy.zeros(R, dtype=numpy.int)
    for i in range(R):
        idx[i] = numpy.sum(cs < rnd[i])

    return idx

