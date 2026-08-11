"""
Boltzmann learning (Monte Carlo) estimation for the State-Space Analysis of
Spike Correlations.

This module adds a third inference path, `param_est='mc'`, alongside the
exact (`'exact'`, 2**N enumeration, N <= ~12) and pseudo-likelihood
(`'pseudo'`, with TAP/Bethe eta) paths. The objective is the SAME exact
likelihood as the `'exact'` path,

    l_t(theta) = R * (y_t . theta - psi(theta))
                 - (theta - theta_o)' Sigma_o^-1 (theta - theta_o) / 2,

but the expectation parameters eta(theta) that appear in its gradient are
estimated by Gibbs sampling instead of 2**N enumeration, so the fit remains
asymptotically unbiased at population sizes where enumeration is infeasible
and pseudo-likelihood/TAP carry approximation bias.

The per-timestep MAP problem is solved by persistent contrastive divergence
(PCD): a population of Gibbs chains is kept in equilibrium with the slowly
moving theta and advanced only a few sweeps per gradient step. Because the
filter sweeps time bins sequentially and theta varies smoothly in time, the
chains carry over from one time bin to the next with no per-bin burn-in;
a single burn-in is performed at the first time bin of each E-step. The
stochastic gradient is optimized with Adam, convergence is judged on an
exponential moving average of the gradient residual (single-step residuals
are dominated by Monte Carlo noise), and the returned theta is a Polyak
average over a trailing window of iterates, which cancels the per-step
sampling noise.

The posterior covariance required by the Kalman smoother uses the diagonal
of the Fisher information. All sufficient-statistic features of the
pairwise model are binary (products of binary variables), so the Fisher
diagonal is exactly eta * (1 - eta) -- no second-moment accumulation is
needed. The diagonal-covariance filter/smoother branch shared with the
pseudo-likelihood path is reused unchanged.

The log marginal likelihood ('mc' entry of container.log_marginal_functions)
evaluates psi(theta) exactly for N <= 15 and by annealed importance sampling
(energies.ais_estimator) for larger N, with fixed per-bin seeds so that the
EM convergence criterion compares deterministic quantities across iterations.

Currently supports second-order (pairwise) interactions only, matching the
restriction of energies.ais_estimator.

Copyright (C) 2026

Author: Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)

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

import warnings

import numpy
from scipy.special import expit

import energies
import transforms

try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# Tunable parameters of the MC fit. Users may modify this dict before
# calling ssll.run (e.g. boltzmann_learning.MC_PARAMS['n_chains'] = 256).
MC_PARAMS = {
    # --- sampler ---
    # Cost per gradient step scales with n_sweeps * N (Python-level site
    # updates; the chain dimension is vectorized and nearly free), while
    # the Monte Carlo error scales with 1/sqrt(n_chains * n_sweeps).
    # So prefer many chains x few sweeps at a fixed sample budget.
    'n_chains': 256,     # persistent Gibbs chains
    'n_sweeps': 4,       # full Gibbs sweeps per gradient step
    'sampler': 'auto',   # 'auto' (numba when importable), 'numba', 'numpy'
    'burnin': 100,       # one-time burn-in sweeps at the first time bin
    'seed': 0,           # base RNG seed (reset at t=0 of every E-step)
    # --- Adam / convergence ---
    'lr': 0.05,          # Adam step size
    'beta1': 0.9,
    'beta2': 0.99,
    'adam_eps': 1e-8,
    'max_iter': 1000,    # maximum gradient steps per time bin
    'min_iter': 50,      # minimum gradient steps before convergence allowed
    'ema': 0.05,         # EMA coefficient for the signed-gradient average
    'tol': 5e-3,         # convergence on max|EMA(dlpo)|/R. The EMA is taken
                         # of the SIGNED gradient vector, whose MC noise
                         # averages towards zero (an EMA of max|dlpo| would
                         # instead converge to the expected maximum of the
                         # noise, a positive floor, and never reach tol).
    'polyak_iters': 50,  # trailing Adam steps averaged into the returned theta
    # --- final eta / Fisher-diagonal estimate at the converged theta ---
    'final_burnin': 5,
    'final_sweeps': 25,
    # --- eta_s along the smoothed trajectory (compute_eta_trajectory) ---
    'eta_burnin': 10,
    'eta_sweeps': 30,
    # --- AIS settings for the log marginal likelihood at N > 15 ---
    'ais_chains': 200,
    'ais_anneals': 200,
}

# Time bin currently processed by the filter; set by max_posterior.run
# (same mechanism as pseudo_likelihood.time_bin).
time_bin = 0

# Module state: persistent chains and RNG, carried across time bins within
# an E-step and re-initialised at time_bin == 0 so that every E-step is
# deterministic given theta (MC noise then cancels in EM convergence ratios).
_N = None
_order = None
_chains = None
_rng = None


def initialise(N, O):
    """
    Prepares the module for a MC (Boltzmann-learning) fit of an N-cell,
    order-O model. Called by container.EMData for param_est='mc'.

    :param int N:
        Total cells in the spike data.
    :param int O:
        Order of spike-train interactions. Only O=2 is supported.
    """
    global _N, _order, _chains
    if O != 2:
        raise NotImplementedError(
            "param_est='mc' currently supports order=2 (pairwise) only")
    _N = N
    _order = O
    _chains = None


# Cache of upper-triangle index pairs, keyed by N (triu_indices is
# surprisingly costly when rebuilt at every gradient step)
_TRIU_CACHE = {}


def _triu(N):
    if N not in _TRIU_CACHE:
        _TRIU_CACHE[N] = numpy.triu_indices(N, k=1)
    return _TRIU_CACHE[N]


def theta_to_h_J(theta, N):
    """
    Splits a natural-parameter vector into the field vector h and the
    symmetric coupling matrix J (zero diagonal).

    :param numpy.ndarray theta:
        (D,) natural parameters, D = N + N(N-1)/2, coupling entries ordered
        as numpy.triu_indices(N, k=1) (identical to
        transforms.enumerate_subsets order for O=2).
    :param int N:
        Number of cells.

    :returns:
        Tuple (h, J) with h (N,) and J (N, N).
    """
    h = theta[:N]
    J = numpy.zeros((N, N))
    ii, jj = _triu(N)
    J[ii, jj] = theta[N:]
    J += J.T
    return h, J


def gibbs_sample_eta(theta, N, chains, rng, n_sweeps, accumulate=True):
    """
    Advances the Gibbs chains n_sweeps systematic-scan sweeps at fixed theta
    and (optionally) accumulates the Monte Carlo estimate of the expectation
    parameters eta over all post-sweep chain states.

    The chains array is updated IN PLACE (this is what makes the chains
    persistent across calls).

    :param numpy.ndarray theta:
        (D,) natural parameters.
    :param int N:
        Number of cells.
    :param numpy.ndarray chains:
        (C, N) binary float array of chain states, modified in place.
    :param numpy.random.Generator rng:
        Random number generator.
    :param int n_sweeps:
        Number of full Gibbs sweeps.
    :param bool accumulate:
        If True, return the eta estimate; if False, only advance the chains
        (burn-in) and return None.

    :returns:
        (D,) eta estimate averaged over n_sweeps * C samples, or None.

    Two implementations are provided. The default ('auto') uses a numba
    JIT kernel with per-chain incremental effective fields (fields are
    only updated when a spin actually flips, escaping the O(C*N) full
    field recomputation per site of the vectorized path); it falls back
    to the pure-numpy chain-vectorized path when numba is not
    installed. Both are valid systematic-scan Gibbs samplers and both
    are deterministic given the rng state, but they consume different
    random streams, so switching sampler changes results within the
    Monte Carlo error.
    """
    sampler = MC_PARAMS.get('sampler', 'auto')
    if sampler == 'numba' and not _HAVE_NUMBA:
        raise ImportError("MC_PARAMS['sampler']='numba' but numba is not "
                          "installed")
    if _HAVE_NUMBA and sampler in ('auto', 'numba'):
        h, J = theta_to_h_J(theta, N)
        # One integer from the rng seeds the kernel's own generator, so
        # the call remains deterministic given the rng state.
        seed = int(rng.integers(1, 2 ** 31 - 1))
        eta1, eta2 = _gibbs_kernel_numba(h, J, chains, int(n_sweeps), seed,
                                         accumulate)
        if not accumulate:
            return None
        S = n_sweeps * chains.shape[0]
        ii, jj = _triu(N)
        eta = numpy.empty(N + ii.size)
        eta[:N] = eta1 / S
        eta[N:] = eta2[ii, jj] / S
        return eta
    return _gibbs_sample_eta_numpy(theta, N, chains, rng, n_sweeps,
                                   accumulate)


if _HAVE_NUMBA:
    @numba.njit(cache=True)
    def _gibbs_kernel_numba(h, J, chains, n_sweeps, seed, accumulate):
        """
        Systematic-scan Gibbs sweeps with per-chain incremental effective
        fields. F[c, k] = h[k] + sum_j J[k, j] * x[c, j] is maintained
        across site updates and only touched when a spin flips (~2p(1-p)
        of updates), instead of being recomputed from the full chain
        state at every site. eta accumulation visits only active spins.
        """
        numpy.random.seed(seed)
        C, N = chains.shape
        # fresh fields at call start: bounds any float drift to one call
        F = numpy.empty((C, N))
        for c in range(C):
            for k in range(N):
                s = h[k]
                for j in range(N):
                    s += J[k, j] * chains[c, j]
                F[c, k] = s
        eta1 = numpy.zeros(N)
        eta2 = numpy.zeros((N, N))
        active = numpy.empty(N, dtype=numpy.int64)
        for _ in range(n_sweeps):
            for c in range(C):
                for i in range(N):
                    p_on = 1.0 / (1.0 + numpy.exp(-F[c, i]))
                    x_new = 1.0 if numpy.random.random() < p_on else 0.0
                    d = x_new - chains[c, i]
                    if d != 0.0:
                        chains[c, i] = x_new
                        for k in range(N):
                            F[c, k] += J[i, k] * d
                if accumulate:
                    # collect active spins; pairs only over the active set
                    n_act = 0
                    for i in range(N):
                        if chains[c, i] != 0.0:
                            active[n_act] = i
                            n_act += 1
                    # active is ascending, so i < j always holds here
                    for a in range(n_act):
                        i = active[a]
                        eta1[i] += 1.0
                        for b in range(a + 1, n_act):
                            eta2[i, active[b]] += 1.0
        return eta1, eta2


if _HAVE_NUMBA:
    @numba.njit(cache=True)
    def _ais_kernel_numba(h1, J1, n_chains, n_anneals, seed):
        """
        AIS log-weights for the matched-H bridge from the independent
        model (h = h1, J = 0) to (h1, J1) along the linear path
        alpha*J1, one systematic Gibbs sweep per anneal step (the same
        bridge as energies.ais_estimator with dh = 0).

        The quadratic energy Q[c] = 0.5 x' J1 x and the coupling fields
        G[c, k] = sum_j J1[k, j] x[c, j] are maintained incrementally on
        spin flips, so each anneal step costs O(S) for the weight update
        plus flip-limited field updates, instead of the S*N^2 dense
        recomputation of the numpy version.
        """
        numpy.random.seed(seed)
        N = h1.shape[0]
        x = numpy.empty((n_chains, N))
        G = numpy.zeros((n_chains, N))
        Q = numpy.zeros(n_chains)
        log_w = numpy.zeros(n_chains)
        # exact independent-model samples at alpha = 0
        for c in range(n_chains):
            for i in range(N):
                p0 = 1.0 / (1.0 + numpy.exp(-h1[i]))
                x[c, i] = 1.0 if numpy.random.random() < p0 else 0.0
        for c in range(n_chains):
            for k in range(N):
                s = 0.0
                for j in range(N):
                    s += J1[k, j] * x[c, j]
                G[c, k] = s
            q = 0.0
            for i in range(N):
                q += x[c, i] * G[c, i]
            Q[c] = 0.5 * q
        da = 1.0 / n_anneals
        for t in range(1, n_anneals + 1):
            a = t * da
            for c in range(n_chains):
                # weight increment at the previous state, then sweep at
                # alpha = a (same order as energies.ais_estimator)
                log_w[c] += da * Q[c]
                for i in range(N):
                    field = h1[i] + a * G[c, i]
                    p_on = 1.0 / (1.0 + numpy.exp(-field))
                    x_new = 1.0 if numpy.random.random() < p_on else 0.0
                    d = x_new - x[c, i]
                    if d != 0.0:
                        x[c, i] = x_new
                        Q[c] += d * G[c, i]
                        for k in range(N):
                            G[c, k] += J1[i, k] * d
        return log_w


def _gibbs_sample_eta_numpy(theta, N, chains, rng, n_sweeps,
                            accumulate=True):
    """Pure-numpy fallback sampler, vectorized over chains."""
    h, J = theta_to_h_J(theta, N)
    C = chains.shape[0]
    # One batched draw replaces the n_sweeps * N per-site rng.random(C)
    # calls. numpy's Generator fills the array in C-order, so
    # rand[s, i] is bit-identical to the s*N+i-th sequential random(C)
    # call — the sampled stream (and thus every result) is unchanged.
    rand = rng.random((n_sweeps, N, C))
    if accumulate:
        eta1_sum = numpy.zeros(N)
        eta2_sum = numpy.zeros((N, N))
    for s in range(n_sweeps):
        rand_s = rand[s]
        for i in range(N):
            # J is symmetric, so row J[i] equals column J[:, i] but is
            # contiguous, which keeps the gemv on the fast path
            field = h[i] + chains.dot(J[i])
            chains[:, i] = (rand_s[i] < expit(field))
        if accumulate:
            eta1_sum += chains.sum(axis=0)
            eta2_sum += chains.T.dot(chains)
    if not accumulate:
        return None
    S = n_sweeps * C
    ii, jj = _triu(N)
    eta = numpy.empty(N + ii.size)
    eta[:N] = eta1_sum / S
    eta[N:] = eta2_sum[ii, jj] / S
    return eta


def _reset_chains(y_t, R):
    """
    Re-seeds the module RNG and re-initialises the persistent chains from
    independent Bernoulli draws matched to the first-order empirical rates,
    then burns them in at the current starting theta. Called at time_bin 0
    of every E-step so each E-step is deterministic.
    """
    global _chains, _rng
    _rng = numpy.random.default_rng(MC_PARAMS['seed'])
    rates = numpy.clip(y_t[:_N], 1.0 / (2 * R), 1.0 - 1.0 / (2 * R))
    _chains = (_rng.random((MC_PARAMS['n_chains'], _N))
               < rates[None, :]).astype(numpy.float64)


def mc_map(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, *args):
    """
    MAP estimate of the natural parameters at one time bin by Boltzmann
    learning: stochastic gradient ascent (Adam) on the exact log posterior
    with Gibbs-sampled eta, using persistent chains (PCD).

    Signature and return convention match the estimators in max_posterior.py
    and pseudo_likelihood.py. Covariances are DIAGONAL (D,) vectors, as in
    the pseudo-likelihood branch.

    :param numpy.ndarray y_t:
        (D,) empirical rates of the spike patterns at this time bin.
    :param numpy.ndarray X_t:
        (R, N) binary spike matrix at this time bin (unused; the sufficient
        statistics y_t are all the likelihood needs).
    :param int R:
        Number of trials.
    :param numpy.ndarray theta_0:
        (D,) starting point for theta (smoothed estimate of the previous EM
        iteration).
    :param numpy.ndarray theta_o:
        (D,) one-step prediction mean.
    :param numpy.ndarray sigma_o:
        (D,) one-step prediction variances (diagonal).
    :param numpy.ndarray sigma_o_i:
        (D,) inverse one-step prediction variances (diagonal).

    :returns:
        Tuple (theta_f, sigma_f) with the posterior mode (D,) and the
        diagonal (D,) of the posterior covariance.
    """
    global _chains

    p = MC_PARAMS
    theta = numpy.array(theta_0, dtype=float)

    # Fresh chains + burn-in at the start of every E-step (time bin 0);
    # afterwards the chains persist across time bins, staying close to
    # equilibrium because theta moves smoothly along the filter sweep.
    if time_bin == 0 or _chains is None:
        _reset_chains(y_t, R)
        gibbs_sample_eta(theta, _N, _chains, _rng, p['burnin'],
                         accumulate=False)

    D = theta.shape[0]
    m = numpy.zeros(D)   # Adam first moment
    v = numpy.zeros(D)   # Adam second moment
    ema_g = numpy.zeros(D)  # bias-corrected EMA of the signed gradient
    resid = numpy.inf
    converged = False

    def adam_step(theta, dlpo, m, v, t):
        m = p['beta1'] * m + (1 - p['beta1']) * dlpo
        v = p['beta2'] * v + (1 - p['beta2']) * dlpo * dlpo
        m_hat = m / (1 - p['beta1'] ** t)
        v_hat = v / (1 - p['beta2'] ** t)
        theta = theta + p['lr'] * m_hat / (numpy.sqrt(v_hat) + p['adam_eps'])
        return theta, m, v

    # --- Phase 1: Adam until the EMA of the signed gradient reaches tol,
    # judged on the same scale as the exact path (max|.| / R). The signed
    # average tends to zero at the optimum, so the criterion is reachable
    # despite the per-step MC noise. ---
    for it in range(1, p['max_iter'] + 1):
        eta = gibbs_sample_eta(theta, _N, _chains, _rng, p['n_sweeps'])
        dlpo = R * (y_t - eta) - sigma_o_i * (theta - theta_o)
        ema_g = (1 - p['ema']) * ema_g + p['ema'] * dlpo
        # Bias correction as in Adam: early iterates are shrunk towards 0
        resid = numpy.amax(numpy.absolute(ema_g)) / \
            (1 - (1 - p['ema']) ** it) / R
        if resid < p['tol'] and it >= p['min_iter']:
            converged = True
            break
        theta, m, v = adam_step(theta, dlpo, m, v, it)

    if not converged:
        warnings.warn(
            'mc_map: gradient EMA %.3e did not reach tol %.3e within %d '
            'iterations at time bin %d; continuing with the Polyak average.'
            % (resid, p['tol'], p['max_iter'], time_bin),
            RuntimeWarning, stacklevel=2)

    # --- Phase 2: Polyak averaging. The iterates now oscillate around the
    # optimum with MC noise; the mean over a trailing window cancels it. ---
    theta_sum = numpy.zeros(D)
    for jt in range(1, p['polyak_iters'] + 1):
        eta = gibbs_sample_eta(theta, _N, _chains, _rng, p['n_sweeps'])
        dlpo = R * (y_t - eta) - sigma_o_i * (theta - theta_o)
        theta, m, v = adam_step(theta, dlpo, m, v, it + jt)
        theta_sum += theta
    theta_f = theta_sum / p['polyak_iters']

    # --- Phase 3: Fisher diagonal at the converged theta. Features are
    # binary, so diag(G) = eta (1 - eta) exactly. ---
    gibbs_sample_eta(theta_f, _N, _chains, _rng, p['final_burnin'],
                     accumulate=False)
    eta_f = gibbs_sample_eta(theta_f, _N, _chains, _rng, p['final_sweeps'])
    fisher_diag = eta_f * (1 - eta_f)
    sigma_f = 1.0 / (R * fisher_diag + sigma_o_i)

    return theta_f, sigma_f


def compute_eta_trajectory(theta_array, N):
    """
    Monte Carlo estimate of the expectation parameters eta for every theta
    of a (smoothed) trajectory, sweeping the time bins sequentially with
    persistent chains so that each bin needs only a short re-equilibration.

    Called by exp_max.e_step for param_est_eta='mc'. Deterministic: uses a
    fresh, fixed-seed RNG on every call.

    :param numpy.ndarray theta_array:
        (T, D) natural parameters (e.g. emd.theta_s).
    :param int N:
        Number of cells.

    :returns:
        (T, D) numpy.ndarray of eta estimates.
    """
    p = MC_PARAMS
    T = theta_array.shape[0]
    rng = numpy.random.default_rng(p['seed'] + 1)
    rates = expit(theta_array[0, :N])
    chains = (rng.random((p['n_chains'], N))
              < rates[None, :]).astype(numpy.float64)
    eta = numpy.empty_like(theta_array)
    for t in range(T):
        burnin = p['burnin'] if t == 0 else p['eta_burnin']
        gibbs_sample_eta(theta_array[t], N, chains, rng, burnin,
                         accumulate=False)
        eta[t] = gibbs_sample_eta(theta_array[t], N, chains, rng,
                                  p['eta_sweeps'])
    return eta


def compute_psi_trajectory(theta_array, N):
    """
    Log-partition function psi for every theta of a trajectory. Exact
    (2**N enumeration) for N <= 15; annealed importance sampling
    (energies.ais_estimator, fixed per-bin seeds) for larger N, with a
    closed-form shortcut for bins whose couplings are all zero (e.g. the
    zero initialisation of theta_f before the first E-step).

    :param numpy.ndarray theta_array:
        (T, D) natural parameters.
    :param int N:
        Number of cells.

    :returns:
        (T,) numpy.ndarray of psi values.
    """
    p = MC_PARAMS
    T = theta_array.shape[0]
    if N <= 15:
        if transforms.p_map is None or \
                transforms.p_map.shape != (2 ** N, theta_array.shape[1]):
            transforms.initialise(N, 2)
        return transforms.compute_psi_vec(theta_array)
    psi = numpy.empty(T)
    sampler = p.get('sampler', 'auto')
    use_numba = _HAVE_NUMBA and sampler in ('auto', 'numba')
    theta0 = numpy.zeros(theta_array.shape[1])
    for t in range(T):
        theta_t = theta_array[t]
        theta0[:N] = theta_t[:N]
        psi0 = float(numpy.sum(numpy.log(1 + numpy.exp(theta0[:N]))))
        if numpy.allclose(theta_t[N:], 0.0):
            psi[t] = psi0
        elif use_numba:
            # same matched-H AIS bridge as energies.ais_estimator, in a
            # numba kernel with incremental energy bookkeeping (different
            # random stream, so psi differs within the AIS error)
            from scipy.special import logsumexp
            h1, J1 = theta_to_h_J(theta_t, N)
            log_w = _ais_kernel_numba(numpy.ascontiguousarray(h1), J1,
                                      p['ais_chains'], p['ais_anneals'],
                                      t + 1)
            psi[t] = psi0 + float(logsumexp(log_w)) - \
                numpy.log(p['ais_chains'])
        else:
            psi[t] = energies.ais_estimator(theta0.copy(), psi0, theta_t, N,
                                            2, S=p['ais_chains'],
                                            T=p['ais_anneals'], seed=t)
    return psi


def log_marginal(emd, period=None):
    """
    Log marginal likelihood of the observed spike-pattern rates for the MC
    path (equation 45 of Shimazaki et al. 2012, with the diagonal-covariance
    convention of the approximate paths and psi from
    compute_psi_trajectory).

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param tuple period:
        Timestep range over which to compute the probability.

    :returns:
        Log marginal probability as a float.
    """
    if period is None:
        period = (0, emd.theta_f.shape[0])
    a, b = 0.0, 0.0
    psi = compute_psi_trajectory(emd.theta_f[period[0]:period[1]], emd.N)
    for k, i in enumerate(range(period[0], period[1])):
        a += emd.R * (numpy.dot(emd.theta_f[i], emd.y[i]) - psi[k])
        theta_d = emd.theta_f[i] - emd.theta_o[i]
        b -= numpy.dot(theta_d, emd.sigma_o_inv[i] * theta_d)
        b += numpy.sum(numpy.log(emd.sigma_f[i])) + \
            numpy.sum(numpy.log(emd.sigma_o_inv[i]))
    return a + b / 2


# Named function pointers to MAP estimators. The MC path has a single
# estimator; all map_function keys resolve to it so that ssll.run's default
# ('cg') works unchanged.
functions = {'nr': mc_map,
             'cg': mc_map,
             'bf': mc_map}
