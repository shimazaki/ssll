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

                           
Updated to extended testing framework.

Copyright (C) 2025
Authors of the extensions: Hideaki Shimazaki (h.shimazaki@i.kyoto-u.ac.jp)

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
import energies
import synthesis
import thermodynamics
import transforms

# Test Parameters
DEFAULT_T = 20  # Number of time steps
DEFAULT_R = 20  # Number of trials
DEFAULT_SPIKE_SEED = 1  # Random seed for spike generation
DEFAULT_THETA_SEED = 42  # Random seed for theta generation
DEFAULT_CONVERGENCE_THRESHOLD = 0.05  # Threshold for KL divergence
DEFAULT_MLLK_TOLERANCE = 1e-6  # Tolerance for log marginal likelihood comparison


SPIKE_GENERATION_TEST_NEURONS = [5]  # Number of neurons for spike generation test
EXPECTED_SPIKE_COUNT = 211  # Total number of spikes expected for 5 neurons
EXPECTED_SPIKE_COUNT_GIBBS = 246  # Total number of spikes expected for Gibbs sampling
EXPECTED_SPIKE_COUNT_GIBBS_PARALLEL = 246  # Total number of spikes expected for Gibbs sampling parallel

# Test Configuration
FIRST_ORDER_TEST_NEURONS = [3]  # Number of neurons for first-order test
SECOND_ORDER_TEST_NEURONS = [4]  # Number of neurons for second-order test
THIRD_ORDER_TEST_NEURONS = [3]  # Number of neurons for third-order test
STATE_MODEL_TEST_NEURONS = [4]  # Number of neurons for state model test
GRADIENT_TEST_NEURONS = [4]  # Number of neurons for gradient test
PSEUDOLIKELIHOOD_TEST_NEURONS = [4]  # Number of neurons for pseudolikelihood test
SINGLE_TIME_BIN_TEST_NEURONS = [3]  # Number of neurons for single time bin test

# Expected Log Marginal Likelihood Values
EXPECTED_MLLK_FIRST_ORDER = -421.674518
EXPECTED_MLLK_SECOND_ORDER = -575.257230
EXPECTED_MLLK_THIRD_ORDER = -401.902792
EXPECTED_MLLK_STATE_MODEL_DIAG = -574.967673
EXPECTED_MLLK_STATE_MODEL_FULL = -573.224739
EXPECTED_MLLK_STATE_MODEL_AUTOREG = -560.376159
EXPECTED_MLLK_GRADIENT_NR = -575.258048
EXPECTED_MLLK_GRADIENT_CG = -575.257230
EXPECTED_MLLK_GRADIENT_BFGS = -575.251454
EXPECTED_MLLK_PSEUDOLIKELIHOOD_CG = -570.582877
EXPECTED_MLLK_PSEUDOLIKELIHOOD_BFGS = -578.793619
EXPECTED_MLLK_SINGLE_TIME_BIN_EXACT = -149.619925
EXPECTED_MLLK_SINGLE_TIME_BIN_CG = -151.303684
EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_BP = -149.582678
EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_CCCP = -149.585147

# Edge Case Expected Values
EXPECTED_MLLK_SINGLE_NEURON = -133.413675
EXPECTED_MLLK_SINGLE_TRIAL = -17.040421

# Thermodynamics Expected Values (N=4, O=2, T=20, R=20, seed=42/1, numpy.random.seed(0), samples=50)
THERMO_RANDOM_SEED = 0
THERMO_SAMPLES = 50
THERMO_THRESHOLD = 90
THERMO_TOLERANCE = 1e-4
EXPECTED_S_PAIR_FIRST = 1.464306
EXPECTED_S_PAIR_LAST = 1.390805
EXPECTED_S_RATIO_FIRST = 0.866118
EXPECTED_C_FIRST = 1.574292
EXPECTED_C_LAST = 1.602107
EXPECTED_P_SILENCE_FIRST = 0.586762
EXPECTED_P_SILENCE_LAST = 0.610721
EXPECTED_C_BETA_MID_FIRST = 1.635892

def klic(p_theta, q_theta, N):
    """
    Computes the Kullback-Leibler divergence for each timestep of two
    natural-parameter distributions.

    Args:
        p_theta: Mean of the actual natural parameters
        q_theta: Mean of the estimated natural parameters
        N: Number of cells from which the natural parameters were generated
    Returns:
        Kullback-Leibler divergence for each timestep
    """
    # Get metadata and patterns
    T, D = p_theta.shape
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
        self.spike_seed = DEFAULT_SPIKE_SEED

    def run_ssll(self, theta, N, O, map_fun='cg',
                 state_cov_val=0.01, state_ar_val=None,
                 param_est_val='exact', param_est_eta='exact'):
        """
        Run the SSLL algorithm with given parameters.

        Args:
            theta: Input theta values
            N: Number of neurons
            O: Order of interactions
            map_fun: Mapping function to use
            state_cov_val: State covariance value
            state_ar_val: State autoregressive value
            param_est_val: Parameter estimation method
            param_est_eta: Eta estimation method
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
    
    def test_0_spike_generation(self):
        print("Test Spike Generation.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in SPIKE_GENERATION_TEST_NEURONS:
            O = 2
            # Initialize transforms library
            transforms.initialise(N, O)
            # Generate synthetic data
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
            # Compute probability from theta values
            p = numpy.zeros((self.T, 2**N))
            for i in numpy.arange(self.T):
                p[i,:] = transforms.compute_p(theta[i,:])

            # Generate spikes
            spikes = synthesis.generate_spikes(p, self.R, seed=self.spike_seed)
            spike_count = numpy.sum(spikes)
            print('Spike Count = %d (expected %d)' % (spike_count, EXPECTED_SPIKE_COUNT))
            self.assertEqual(spike_count, EXPECTED_SPIKE_COUNT)

            # Generate spikes using Gibbs sampling
            spikes = synthesis.generate_spikes_gibbs(theta, N, O, self.R, seed=self.spike_seed)
            spike_count = numpy.sum(spikes)
            print('Spike Count = %d (expected %d)' % (spike_count, EXPECTED_SPIKE_COUNT_GIBBS))
            self.assertEqual(spike_count, EXPECTED_SPIKE_COUNT_GIBBS)
    
            # Generate spikes using Gibbs sampling parallel
            spikes = synthesis.generate_spikes_gibbs_parallel(theta, N, O, self.R, seed=self.spike_seed)
            spike_count = numpy.sum(spikes)
            print('Spike Count = %d (expected %d)' % (spike_count, EXPECTED_SPIKE_COUNT_GIBBS_PARALLEL))
            self.assertEqual(spike_count, EXPECTED_SPIKE_COUNT_GIBBS_PARALLEL)    

    def test_1_first_order_time_varying(self):
        print("Test First-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in FIRST_ORDER_TEST_NEURONS:
            O = 1
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
            # Run the actual test
            emd = self.run_ssll(theta, N, O)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_FIRST_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_FIRST_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_2_second_order_time_varying(self):
        print("\nTest Second-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in SECOND_ORDER_TEST_NEURONS:
            O = 2
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
            # Run the actual test
            emd = self.run_ssll(theta, N, O)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SECOND_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SECOND_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_3_third_order_time_varying(self):
        print("\nTest Third-Order Time-Varying Interactions.")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in THIRD_ORDER_TEST_NEURONS:
            O = 3
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
            # Run the actual test
            emd = self.run_ssll(theta, N, O)
        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_THIRD_ORDER)
        self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_THIRD_ORDER) > DEFAULT_MLLK_TOLERANCE)
        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_4_state_models_covariance(self):
        print("\nTest Different State Models (N=4, O=2, Time-Varying Interactions).")
        start_cpu_time = time.process_time()
        # Repeat test for different numbers of neurons
        for N in STATE_MODEL_TEST_NEURONS:
            O = 2
            D = transforms.compute_D(N, 2)
            # Create time-varying theta parameters
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
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
        for N in GRADIENT_TEST_NEURONS:
            O = 2
            # Create time-varying theta parameters
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
            # Run the algorithm!

            # Newton-Raphson
            tc = time.time()
            emd = self.run_ssll(theta, N, O, map_fun='nr')
            print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_GRADIENT_NR)
            self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_GRADIENT_NR) > DEFAULT_MLLK_TOLERANCE)
            print('nr in %f s' %(time.time() - tc))

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
        for N in PSEUDOLIKELIHOOD_TEST_NEURONS:
            O = 2
            # Create time-varying theta parameters
            theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
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
        for N in SINGLE_TIME_BIN_TEST_NEURONS:
            O = 2
            self.T = 1
            self.R = 300
            # Create time-varying theta parameters
            theta = synthesis.generate_stationary_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)

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
            print('cg mf in %f s' %(time.time() - tc))

            # BFGS Bethe BP
            tc = time.time()
            emd = self.run_ssll(theta, N, O, map_fun='bf',
                                param_est_val='pseudo', param_est_eta='bethe_BP')
            # Check the consistency with the expected result.
            print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_BP)
            self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_BP) > DEFAULT_MLLK_TOLERANCE)
            print('bfgs bethe bp in %f s' %(time.time() - tc))
 
            # BFGS Bethe CCP
            tc = time.time()
            emd = self.run_ssll(theta, N, O, map_fun='bf',
                                param_est_val='pseudo', param_est_eta='bethe_CCCP')
            # Check the consistency with the expected result.
            print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_CCCP)
            self.assertFalse(numpy.absolute(emd.mllk-EXPECTED_MLLK_SINGLE_TIME_BIN_BFGS_CCCP) > DEFAULT_MLLK_TOLERANCE)
            print('bfgs bethe cccp in %f s' %(time.time() - tc))
            end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_8_edge_cases(self):
        print("Test Edge Cases.")
        start_cpu_time = time.process_time()

        # Single neuron (N=1, O=1)
        N, O = 1, 1
        theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
        emd = self.run_ssll(theta, N, O)
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_NEURON)
        self.assertFalse(numpy.absolute(emd.mllk - EXPECTED_MLLK_SINGLE_NEURON) > DEFAULT_MLLK_TOLERANCE)
        print('single neuron OK')

        # Single trial (R=1)
        N, O = 3, 2
        self.R = 1
        theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
        transforms.initialise(N, O)
        p = numpy.zeros((self.T, 2**N))
        for i in numpy.arange(self.T):
            p[i,:] = transforms.compute_p(theta[i,:])
        spikes = synthesis.generate_spikes(p, self.R, seed=self.spike_seed)
        emd = __init__.run(spikes, O, EM_Info=False)
        print('Log marginal likelihood = %.6f (expected)' % EXPECTED_MLLK_SINGLE_TRIAL)
        self.assertFalse(numpy.absolute(emd.mllk - EXPECTED_MLLK_SINGLE_TRIAL) > DEFAULT_MLLK_TOLERANCE)
        print('single trial OK')

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_9_thermodynamics(self):
        print("Test Thermodynamics (N=4, O=2).")
        start_cpu_time = time.process_time()

        # Fit a model (same as test_2)
        N, O = 4, 2
        theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)
        emd = self.run_ssll(theta, N, O)
        energies.get_energies(emd)

        # --- get_theta_samples ---
        thetas = thermodynamics.get_theta_samples(emd, 10)
        self.assertEqual(thetas.shape, (self.T, emd.D, 10))
        # First sample should be the MAP estimate
        self.assertTrue(numpy.allclose(thetas[:, :, 0], emd.theta_s),
                        "First theta sample should equal theta_s")
        print('get_theta_samples OK')

        # --- get_entropy (deterministic) ---
        S_pair, S_ratio = thermodynamics.get_entropy(emd)
        self.assertEqual(S_pair.shape, (self.T,))
        self.assertAlmostEqual(S_pair[0], EXPECTED_S_PAIR_FIRST, places=4)
        self.assertAlmostEqual(S_ratio[0], EXPECTED_S_RATIO_FIRST, places=4)
        print('get_entropy OK (S_pair[0]=%.6f, S_ratio[0]=%.6f)' % (S_pair[0], S_ratio[0]))

        # --- compute_heat_capacity (deterministic) ---
        C = thermodynamics.compute_heat_capacity(emd)
        self.assertEqual(C.shape, (self.T,))
        self.assertAlmostEqual(C[0], EXPECTED_C_FIRST, places=4)
        self.assertAlmostEqual(C[-1], EXPECTED_C_LAST, places=4)
        print('compute_heat_capacity OK (C[0]=%.6f, C[-1]=%.6f)' % (C[0], C[-1]))

        # --- compute_entropy_b (with sampling) ---
        numpy.random.seed(THERMO_RANDOM_SEED)
        S_pair_b, S_pair_bounds, S_ratio_b, S_ratio_bounds = \
            thermodynamics.compute_entropy_b(emd, THERMO_SAMPLES, THERMO_THRESHOLD)
        self.assertEqual(S_pair_b.shape, (self.T,))
        self.assertEqual(S_pair_bounds.shape, (self.T, 2))
        self.assertAlmostEqual(S_pair_b[0], EXPECTED_S_PAIR_FIRST, places=4)
        self.assertAlmostEqual(S_pair_b[-1], EXPECTED_S_PAIR_LAST, places=4)
        # Bounds should bracket the MAP estimate
        self.assertTrue(S_pair_bounds[0, 0] <= S_pair_b[0] <= S_pair_bounds[0, 1],
                        "Entropy bounds should bracket MAP estimate")
        print('compute_entropy_b OK (S_pair[0]=%.6f, bounds=[%.4f, %.4f])' %
              (S_pair_b[0], S_pair_bounds[0, 0], S_pair_bounds[0, 1]))

        # --- compute_heat_capacity_b (with sampling) ---
        numpy.random.seed(THERMO_RANDOM_SEED)
        C_b, C_bounds = thermodynamics.compute_heat_capacity_b(emd, THERMO_SAMPLES, THERMO_THRESHOLD)
        self.assertEqual(C_b.shape, (self.T,))
        self.assertEqual(C_bounds.shape, (self.T, 2))
        self.assertAlmostEqual(C_b[0], EXPECTED_C_FIRST, places=4)
        self.assertAlmostEqual(C_b[-1], EXPECTED_C_LAST, places=4)
        self.assertTrue(C_bounds[0, 0] <= C_b[0] <= C_bounds[0, 1],
                        "Heat capacity bounds should bracket MAP estimate")
        print('compute_heat_capacity_b OK (C[0]=%.6f, bounds=[%.4f, %.4f])' %
              (C_b[0], C_bounds[0, 0], C_bounds[0, 1]))

        # --- compute_p_silence_b (with sampling) ---
        numpy.random.seed(THERMO_RANDOM_SEED)
        p_s, p_s_bounds = thermodynamics.compute_p_silence_b(emd, THERMO_SAMPLES, THERMO_THRESHOLD)
        self.assertEqual(p_s.shape, (self.T,))
        self.assertEqual(p_s_bounds.shape, (self.T, 2))
        self.assertAlmostEqual(p_s[0], EXPECTED_P_SILENCE_FIRST, places=4)
        self.assertAlmostEqual(p_s[-1], EXPECTED_P_SILENCE_LAST, places=4)
        self.assertTrue(p_s_bounds[0, 0] <= p_s[0] <= p_s_bounds[0, 1],
                        "p_silence bounds should bracket MAP estimate")
        print('compute_p_silence_b OK (p_silence[0]=%.6f, bounds=[%.4f, %.4f])' %
              (p_s[0], p_s_bounds[0, 0], p_s_bounds[0, 1]))

        # --- get_heat_capacity_beta ---
        c_betas = thermodynamics.get_heat_capacity_beta(emd, 5)
        self.assertEqual(c_betas.shape, (5, self.T))
        self.assertAlmostEqual(c_betas[2, 0], EXPECTED_C_BETA_MID_FIRST, places=4)
        print('get_heat_capacity_beta OK (c_betas[2,0]=%.6f)' % c_betas[2, 0])

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_a_jax_tap_solver(self):
        """Test JAX TAP solver matches numpy version when JAX is available."""
        print("Test JAX TAP solver.")
        import mean_field

        if not mean_field.HAS_JAX:
            print('JAX not available, skipping')
            return

        # Generate test theta with pairwise interactions
        N, O = 5, 2
        theta = synthesis.generate_thetas(N, O, self.T, seed=DEFAULT_THETA_SEED)

        # Run JAX version
        eta_jax = mean_field.forward_problem_hessian(theta[0], N)

        # Run numpy version (force fallback)
        mean_field.HAS_JAX = False
        eta_np = mean_field.forward_problem_hessian(theta[0], N)
        mean_field.HAS_JAX = True

        # They should match to high precision
        max_diff = numpy.max(numpy.abs(eta_jax - eta_np))
        print('JAX vs numpy TAP max diff: %.2e' % max_diff)
        self.assertTrue(max_diff < 1e-10,
                        'JAX and numpy TAP results differ by %.2e' % max_diff)

        # Also test across all timesteps
        for t in range(self.T):
            eta_j = mean_field.forward_problem_hessian(theta[t], N)
            mean_field.HAS_JAX = False
            eta_n = mean_field.forward_problem_hessian(theta[t], N)
            mean_field.HAS_JAX = True
            self.assertTrue(numpy.max(numpy.abs(eta_j - eta_n)) < 1e-10,
                            'Mismatch at t=%d' % t)
        print('All %d timesteps match OK' % self.T)
        print('JAX TAP solver test passed')


if __name__ == '__main__':
    unittest.main()
