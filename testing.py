"""
Functions for testing results.

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
import unittest
import time

import __init__
import synthesis
import transforms

# Test Parameters
DEFAULT_T = 20  # Number of time steps
DEFAULT_R = 20  # Number of trials
DEFAULT_THETA_BASE = -3.  # Base value for theta parameters
DEFAULT_SPIKE_SEED = 1  # Random seed for spike generation
DEFAULT_WAVE_SEED = 1  # Random seed for wave generation
DEFAULT_CONVERGENCE_THRESHOLD = 0.05  # Threshold for KL divergence
DEFAULT_MLLK_TOLERANCE = 1e-6  # Tolerance for log marginal likelihood comparison

# Expected Log Marginal Likelihood Values
EXPECTED_MLLK_FIRST_ORDER = -349.668974
EXPECTED_MLLK_SECOND_ORDER = -363.021487
EXPECTED_MLLK_THIRD_ORDER = -229.173379
EXPECTED_MLLK_STATE_MODEL_DIAG = -563.760601
EXPECTED_MLLK_STATE_MODEL_FULL = -561.667173
EXPECTED_MLLK_STATE_MODEL_AUTOREG = -554.251761
EXPECTED_MLLK_GRADIENT_CG = -563.838955
EXPECTED_MLLK_GRADIENT_BFGS = -563.836771
EXPECTED_MLLK_PSEUDOLIKELIHOOD_CG = -1134.685715
EXPECTED_MLLK_PSEUDOLIKELIHOOD_BFGS = -1157.434769
EXPECTED_MLLK_SINGLE_TIME_BIN_EXACT = -189.855470
EXPECTED_MLLK_SINGLE_TIME_BIN_CG = -191.870697
EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS = -189.688459

def klic(p_theta, q_theta, N):
    """
    Computes the Kullback-Leibler divergence for each timestep of two
    natural-parameter distributions.

    Arguments:
        p_theta -- Mean of the actual natural parameters
        q_theta -- Mean of the estimated natural parameters
        N -- Number of cells from which the natural parameters were generated
    Returns:
        Kullback-Leibler divergence for each timestep
    """
    # Get metadata and patterns
    T, D = p_theta.shape
    fx = transforms.enumerate_patterns(N)
    # Compute divergence for each timestep
    kld = numpy.zeros(T)
    for i in numpy.arange(T):
        # Compute normalisations for current timestep
        phi_q = transforms.compute_psi(q_theta[i,:])
        phi_p = transforms.compute_psi(p_theta[i,:])
        # Compute log probability for each pattern
        log_prob_q = numpy.array(transforms.p_map.dot(q_theta[i,:]) - phi_q)
        log_prob_p = numpy.array(transforms.p_map.dot(p_theta[i,:]) - phi_p)
        # Take the KLD for this timestep
        kld[i] = numpy.sum(numpy.exp(log_prob_q) * log_prob_q -\
                           numpy.exp(log_prob_q) * log_prob_p)

    return kld

class TestEstimator(unittest.TestCase):
    """
    Tests for the SSLL estimator.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.T = DEFAULT_T
        self.R = DEFAULT_R
        self.theta_base = DEFAULT_THETA_BASE
        self.spike_seed = DEFAULT_SPIKE_SEED
        self.wave_seed = DEFAULT_WAVE_SEED

    def run_ssll(self, theta, N, O, map_fun='cg',
                 state_cov_val=0.01, state_ar_val=None,
                 param_est_val='exact', param_est_eta='exact'):
        """
        Run the SSLL algorithm with given parameters.

        Arguments:
            theta -- Input theta values
            N -- Number of neurons
            O -- Order of interactions
            map_fun -- Mapping function to use
            state_cov_val -- State covariance value
            state_ar_val -- State autoregressive value
            param_est_val -- Parameter estimation method
            param_est_eta -- Eta estimation method
        Returns:
            EMData object containing results
        """
        # Initialise the library for computing pattern probabilities
        transforms.initialise(N, O)
        # Compute probability from theta values
        p = numpy.zeros((self.T, 2**N))
        for i in numpy.arange(self.T):
            p[i,:] = transforms.compute_p(theta[i,:])
        # Generate spikes according to those probabilities
        spikes = synthesis.generate_spikes(p, self.R, seed=self.spike_seed)
        # Run the algorithm!
        emd = __init__.run(spikes, O, map_function=map_fun,
                           state_cov=state_cov_val, state_ar=state_ar_val,
                           param_est=param_est_val, param_est_eta=param_est_eta)
        # Compute the KL divergence between real and estimated parameters
        kld = klic(theta, emd.theta_s, emd.N)
        # Check that KL divergence is OK
        self.assertFalse(numpy.any(kld[50:-50] > DEFAULT_CONVERGENCE_THRESHOLD),
                        "KL divergence exceeds threshold")
        return emd

    def wave(self, A, f, phi, T):
        """
        Generate a wave function.

        Arguments:
            A -- Amplitude
            f -- Frequency
            phi -- Phase
            T -- Time period
        Returns:
            Wave values
        """
        rng = numpy.arange(0, T, 1e-3)
        wave = A * numpy.sin(2 * numpy.pi * f * rng + phi)
        return wave

    def test_1_first_order_time_varying(self):
        print("Test First-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in [4]: #2**numpy.arange(3):  # [1, 2, 4]
            print(N)
            # Create a regular set of theta parameters for each timestep
            theta = numpy.ones((self.T, N)) * self.theta_base
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N + 1)
            cells = numpy.random.choice(numpy.arange(N), n_random)
            for i in numpy.arange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            emd = self.run_ssll(theta, N, 1)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_FIRST_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_FIRST_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_2_second_order_time_varying(self):
        print("\nTest Second-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in [4]: #2**numpy.arange(1, 3):  # [2, 4]
            print(N)
            # Compute dimensionality of natural-parameter distribution
            D = transforms.compute_D(N, 2)
            # Create a regular set of theta parameters for each timestep
            theta = numpy.zeros((self.T, D))
            theta[:,:N] = self.theta_base
            theta[:,N:] = -1.
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N / 2)
            cells = numpy.random.choice(numpy.arange(N), n_random)
            for i in numpy.arange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Add time-varying components for some interactions
            n_random = numpy.random.randint(0, D - N)
            interactions = numpy.random.choice(numpy.arange(N, D), n_random)
            for i in numpy.arange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(1, 2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = interactions[i]
                theta[:,idx] = self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            emd = self.run_ssll(theta, N, 2)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SECOND_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SECOND_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_3_third_order_time_varying(self):
        print("\nTest Third-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in [3]: #3**numpy.arange(1,2):  # [3]
            print(N)
            # Compute dimensionality of natural-parameter distribution
            D = transforms.compute_D(N, 3)
            # Create a regular set of theta parameters for each timestep
            theta = numpy.zeros((self.T, D))
            theta[:,:N] = self.theta_base
            theta[:,N:] = -1.
            # Add time-varying components for some neurons
            numpy.random.seed(self.wave_seed)
            n_random = numpy.random.randint(0, N / 2)
            cells = numpy.random.choice(numpy.arange(N), n_random)
            for i in numpy.arange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = cells[i]
                theta[:,idx] = self.theta_base + \
                    self.wave(A, f, phi, self.T * 1e-3)
            # Add time-varying components for some interactions
            n_random = numpy.random.randint(0, D - N)
            interactions = numpy.random.choice(numpy.arange(N, D), n_random)
            for i in numpy.arange(n_random):
                # Draw random phase, amplitude and frequency
                phi = numpy.random.uniform(0, 2 * numpy.pi)
                A = numpy.random.uniform(1, 2)
                f = 1 / (numpy.random.uniform(self.T / 5., 5 * self.T) * 1e-3)
                idx = interactions[i]
                theta[:,idx] = self.wave(A, f, phi, self.T * 1e-3)
            # Run the actual test
            emd = self.run_ssll(theta, N, 3)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_THIRD_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_THIRD_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_4_state_models_covariance(self):
        print("\nTest Different State Models (N=4, O=2, Time-Varying Interactions).")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        N, O = 4, 2
        D = transforms.compute_D(N, 2)
        # Create time-varying theta parameters
        theta = synthesis.generate_thetas(N, O, self.T)
        # Run the algorithm!
        # A diagonal covariance matrix
        tc = time.time()
        emd = self.run_ssll(theta, N, O, state_cov_val=0.01*numpy.ones(D))
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_STATE_MODEL_DIAG)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_STATE_MODEL_DIAG) > DEFAULT_MLLK_TOLERANCE)
        print('diag cov in %f s' %(time.time() - tc))
        # A full covariance matrix
        tc = time.time()
        emd = self.run_ssll(theta, N, O, state_cov_val=0.01*numpy.identity(D))
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_STATE_MODEL_FULL)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_STATE_MODEL_FULL) > DEFAULT_MLLK_TOLERANCE)
        print('full cov in %f s' %(time.time() - tc))
        # An autoregressive matrix
        tc = time.time()
        emd = self.run_ssll(theta, N, O, state_ar_val=1.*numpy.identity(D))
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_STATE_MODEL_AUTOREG)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_STATE_MODEL_AUTOREG) > DEFAULT_MLLK_TOLERANCE)
        print('autoreg in %f s' %(time.time() - tc))
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_5_gradient_optimization(self):
        print("\nTest Gradient Algorithms (N=6, O=2, Time-Varying Interactions).")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        N, O = 4, 2
        # Create time-varying theta parameters
        theta = synthesis.generate_thetas(N, O, self.T)
        # Run the algorithm!
        # Conjugate Gradient
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='cg')
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_GRADIENT_CG)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_GRADIENT_CG) > DEFAULT_MLLK_TOLERANCE)
        print('cg in %f s' %(time.time() - tc))
        # BFGS
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='bf')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_GRADIENT_BFGS)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_GRADIENT_BFGS) > DEFAULT_MLLK_TOLERANCE)
        print('bfgs in %f s' %(time.time() - tc))
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_6_pseudolikelihood(self):
        print("Test Psuedolikelihood Algorithm (N=3, O=2, Time-Varying Interactions).")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        N, O = 8, 2
        # Create time-varying theta parameters
        theta = synthesis.generate_thetas(N, O, self.T)
        # Run the algorithm!
        # CG Mean field
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='cg',
                            param_est_val='pseudo', param_est_eta='mf')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_PSEUDOLIKELIHOOD_CG)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_PSEUDOLIKELIHOOD_CG) > DEFAULT_MLLK_TOLERANCE)
        print('cg in %f s' %(time.time() - tc))
        # BFGS Bethe_bybrid
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='bf',
                            param_est_val='pseudo', param_est_eta='bethe_hybrid')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_PSEUDOLIKELIHOOD_BFGS)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_PSEUDOLIKELIHOOD_BFGS) > DEFAULT_MLLK_TOLERANCE)
        print('bfgs in %f s' %(time.time() - tc))
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_7_single_time_bin(self):
        print("Test One Time Bin for (N=3, O=2).")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        N, O = 3, 2
        self.T = 1
        self.R = 300
        # Create time-varying theta parameters
        theta = synthesis.generate_stationary_thetas(N, O, self.T)

        # Exact
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='cg')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TIME_BIN_EXACT)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SINGLE_TIME_BIN_EXACT) > DEFAULT_MLLK_TOLERANCE)
        print('bfgs in %f s' %(time.time() - tc))

        # CG Mean field
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='cg',
                            param_est_val='pseudo', param_est_eta='mf')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TIME_BIN_CG)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SINGLE_TIME_BIN_CG) > DEFAULT_MLLK_TOLERANCE)
        print('cg in %f s' %(time.time() - tc))

        # BFGS Bethe_bybrid
        tc = time.time()
        emd = self.run_ssll(theta, N, O, map_fun='bf',
                            param_est_val='pseudo', param_est_eta='bethe_hybrid')
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS) > DEFAULT_MLLK_TOLERANCE)
        print('bfgs in %f s' %(time.time() - tc))
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

if __name__ == '__main__':
    unittest.main()
