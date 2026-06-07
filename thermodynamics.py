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
import synthesis
import transforms

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:

    @_numba.njit(cache=True, fastmath=True, parallel=True)
    def _gibbs_pairwise_sweeps_nb(x, theta_1, J, burn_in, seed):
        """Inner Gibbs sweep for pairwise model — parallel over chains.

        x: (T, R, N) float64, updated in place.
        theta_1: (T, N) float64.
        J: (T, N, N) float64, symmetric, zero diag.
        burn_in: int — number of full Gibbs sweeps.
        seed: int — seed for numba's internal RNG (one global stream).

        Uses numba's internal RNG so we avoid materializing a
        (burn_in, N, T, R) float64 uniform buffer (which can be larger than
        the entire downstream computation). Parallelizes the outer (t, r)
        chain dimension across threads — chains are independent so the
        per-thread RNG splits do not affect correctness of the MC estimator.
        """
        numpy.random.seed(seed)
        T, R, N = x.shape
        TR = T * R
        for sweep in range(burn_in):
            for i in range(N):
                for tr in _numba.prange(TR):
                    t = tr // R
                    r = tr - t * R
                    h = theta_1[t, i]
                    for j in range(N):
                        h += x[t, r, j] * J[t, i, j]
                    if h >= 0.0:
                        p = 1.0 / (1.0 + numpy.exp(-h))
                    else:
                        e = numpy.exp(h)
                        p = e / (1.0 + e)
                    x[t, r, i] = 1.0 if numpy.random.random() < p else 0.0


def _psi(theta, N, O, method):
    """Compute psi for a (T, D) theta array using the requested method.

    method: 'auto' (energies.compute_psi: exact if N<=15, OT if N>15),
            'exact' (transforms.compute_psi, enumerates 2**N), or
            'approx' (Ogata-Tanemura, applicable to any N).
    """
    if method == 'auto':
        return energies.compute_psi(theta, N, O)
    if method == 'exact':
        transforms.initialise(N, O)
        return transforms.compute_psi_vec(theta)
    if method == 'approx':
        T = theta.shape[0]
        theta0 = numpy.copy(theta)
        theta0[:, N:] = 0
        psi0 = energies.compute_ind_psi(theta0[:, :N])
        psi = numpy.empty(T)
        for i in range(T):
            psi[i] = energies.ot_estimator(theta0[i], psi0[i], theta[i], N, O, N)
        return psi
    raise ValueError("method must be 'auto', 'exact', or 'approx'")


def _gibbs_pairwise_batch(theta, N, R, burn_in, seed):
    """Vectorized parallel-chain Gibbs sampler for the pairwise (O=2) model.

    Runs R independent chains per time bin, batched across all T bins. After
    ``burn_in`` full sweeps it returns one sample per chain, so each (t, r)
    sample is independent. Cost per sweep is one batched matmul of shape
    (T, R, N) x (T, N, N), i.e. O(T*R*N^2) flops with no Python inner loop.

    :param numpy.ndarray theta: (T, D) parameter array, D = N + N*(N-1)/2.
    :param int N: number of cells.
    :param int R: number of parallel chains (samples) per time bin.
    :param int burn_in: number of full Gibbs sweeps before sampling.
    :param int seed: RNG seed.
    :return: numpy.ndarray (T, R, N) of uint8 spike samples.
    """
    T, D = theta.shape
    theta_1 = theta[:, :N]                                   # (T, N)
    iu, ju = numpy.triu_indices(N, 1)
    J = numpy.zeros((T, N, N))
    J[:, iu, ju] = theta[:, N:]
    J[:, ju, iu] = theta[:, N:]                              # (T, N, N), symmetric, zero diag

    rng = numpy.random.default_rng(seed)
    x = (rng.random((T, R, N)) < 0.5).astype(numpy.float64)
    if _HAVE_NUMBA:
        # Numba inner loop: fused matvec + sample, no per-neuron numpy call.
        # numpy.random.seed inside numba is independent of `rng` (used for the
        # initial x state above), so derive a 32-bit seed deterministically.
        nb_seed = 0 if seed is None else (int(seed) ^ 0xDEADBEEF) & 0xFFFFFFFF
        _gibbs_pairwise_sweeps_nb(x, theta_1, J, burn_in, nb_seed)
    else:
        # Numpy fallback: N batched matvecs of shape (T, R, N) x (T, N) per
        # sweep, no (T, R, N) temporary on the inner loop.
        for _ in range(burn_in):
            rand = rng.random((N, T, R))
            for i in range(N):
                h_i = theta_1[:, i, None] + numpy.einsum('trj,tj->tr', x, J[:, i, :])
                p_i = 1.0 / (1.0 + numpy.exp(-h_i))
                x[:, :, i] = (rand[i] < p_i).astype(numpy.float64)
    return x.astype(numpy.uint8)


def _heat_capacity_sampling(theta_eff, N, O, R, pre_n, sample_steps, seed,
                            parallel=False, num_proc=1):
    """Sampling-based heat capacity via the fluctuation-dissipation identity.

    For ``g(s) := psi(s * theta_eff)``, ``g''(s=1)`` equals
    ``Var_{x ~ p(.|theta_eff)}[theta_eff . f(x)]``, where ``f(x)`` is the
    order-O sufficient-statistic vector (subset-indicator features used by the
    rest of the library). This matches the quantity returned by the
    finite-difference path in :func:`compute_heat_capacity`.

    For O=2 (pairwise) uses the batched parallel-chain sampler
    :func:`_gibbs_pairwise_batch`; for O>2 falls back to the per-bin
    single-chain Gibbs sampler in :mod:`synthesis`.

    :param numpy.ndarray theta_eff:
        (T, D) array. Pass ``beta * theta_s`` when probing inverse temperature
        ``beta``.
    :param int N: number of cells.
    :param int O: model interaction order.
    :param int R: number of Gibbs samples per time bin.
    :param int pre_n: burn-in sweeps per time bin.
    :param int sample_steps: thinning between retained samples (O>2 path only).
    :param int seed: RNG seed (per-bin seeds are derived from this).
    :param bool parallel: if True and O>2, use the multiprocessing fallback.
    :param int num_proc: pool size for the O>2 multiprocessing fallback.
    :return: numpy.ndarray of shape (T,) — heat capacity per time bin.
    """
    T = theta_eff.shape[0]
    if O == 2:
        X = _gibbs_pairwise_batch(theta_eff, N, R, burn_in=pre_n, seed=seed)
        # Energy E_{t,r} = theta_1 . x + theta_2 . (x_i * x_j over pairs)
        theta_1 = theta_eff[:, :N]                                  # (T, N)
        theta_2 = theta_eff[:, N:]                                  # (T, N(N-1)/2)
        iu, ju = numpy.triu_indices(N, 1)
        X_f = X.astype(numpy.float64)
        E_lin = numpy.einsum('ti,tri->tr', theta_1, X_f)
        E_pair = numpy.einsum('tk,trk->tr',
                              theta_2, X_f[:, :, iu] * X_f[:, :, ju])
        E = E_lin + E_pair                                          # (T, R)
        return E.var(axis=1, ddof=1)
    if parallel:
        X = synthesis.generate_spikes_gibbs_parallel(
            theta_eff, N, O, R, seed=seed, pre_n=pre_n,
            sample_steps=sample_steps, num_proc=num_proc)
    else:
        X = synthesis.generate_spikes_gibbs(
            theta_eff, N, O, R, seed=seed, pre_n=pre_n,
            sample_steps=sample_steps)
    subsets = transforms.enumerate_subsets(N, O)
    D = len(subsets)
    subset_map = numpy.zeros((D, N))
    for i in range(D):
        subset_map[i, subsets[i]] = 1
    subset_count = subset_map.sum(axis=1)
    C = numpy.empty(T)
    for t in range(T):
        active = (subset_map @ X[t].T == subset_count[:, None]).astype(numpy.float64)
        E = theta_eff[t] @ active
        C[t] = E.var(ddof=1)
    return C


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


def compute_heat_capacity_b(emd, samples, threshold, beta=1, method='auto',
                            n_samples=1000, pre_n=100, sample_steps=1, seed=None):
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
    :param method: str
    'auto' (default): exact for N<=15, Ogata-Tanemura for N>15.
    'exact': always enumerate 2**N. 'approx': always use Ogata-Tanemura.
    'sampling': Gibbs Monte Carlo via the fluctuation-dissipation identity
    (see :func:`_heat_capacity_sampling`).
    :param n_samples: int
    Gibbs samples per (theta_sample, time bin) when ``method='sampling'``.
    :param pre_n: int
    burn-in sweeps per bin when ``method='sampling'``.
    :param sample_steps: int
    thinning between retained Gibbs samples.
    :param seed: int or None
    RNG seed for the Gibbs sampler.
    :return: numpy.ndarray, numpy.ndarray
    The heat capacity and the bounds based on the threshold
    """
    T, D = emd.theta_s.shape

    thetas = beta * get_theta_samples(emd, samples)  # (T, D, samples)

    if method == 'sampling':
        C = numpy.empty((T, samples))
        for s in range(samples):
            s_seed = None if seed is None else seed + s
            C[:, s] = _heat_capacity_sampling(
                thetas[:, :, s], emd.N, emd.order, R=n_samples,
                pre_n=pre_n, sample_steps=sample_steps, seed=s_seed)
    else:
        # Reshape (T, D, samples) -> (samples*T, D) so psi runs in one batched call.
        th_stack = numpy.moveaxis(thetas, 2, 0).reshape(samples * T, D)
        epsilon = 1e-3
        psi = _psi(th_stack, emd.N, emd.order, method)
        tmp1 = _psi(th_stack * (1 + epsilon), emd.N, emd.order, method)
        tmp2 = _psi(th_stack * (1 - epsilon), emd.N, emd.order, method)
        C = ((tmp1 - 2 * psi + tmp2) / (epsilon ** 2)).reshape(samples, T).T

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


def compute_heat_capacity(emd, beta=1, method='auto',
                          n_samples=1000, pre_n=100, sample_steps=1, seed=None):
    """
    Computes the heat capacity

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param beta: float
    the value of beta used to slightly vary the theta parameters.
    :param method: str
    'auto' (default): exact for N<=15, Ogata-Tanemura for N>15.
    'exact': always enumerate 2**N. 'approx': always use Ogata-Tanemura.
    'sampling': Gibbs Monte Carlo — applicable to any N. Computes
    ``Var_{p(x|beta*theta)}[(beta*theta) . f(x)]`` directly via the
    fluctuation-dissipation identity.
    :param n_samples: int
    Gibbs samples per time bin when ``method='sampling'``.
    :param pre_n: int
    burn-in sweeps per bin when ``method='sampling'``.
    :param sample_steps: int
    thinning between retained Gibbs samples.
    :param seed: int or None
    RNG seed for the Gibbs sampler.
    :return: numpy.ndarray, numpy.ndarray
    The heat capacity (if you wants bounding heat capacity, use compute_heat_capacity_b)
    """
    theta = beta * emd.theta_s
    if method == 'sampling':
        return _heat_capacity_sampling(theta, emd.N, emd.order, R=n_samples,
                                       pre_n=pre_n, sample_steps=sample_steps,
                                       seed=seed)
    epsilon = 1e-3
    psi = _psi(theta, emd.N, emd.order, method)
    tmp1 = _psi(theta * (1 + epsilon), emd.N, emd.order, method)
    tmp2 = _psi(theta * (1 - epsilon), emd.N, emd.order, method)
    C = (tmp1 - 2 * psi + tmp2) / (epsilon ** 2)

    return C

def get_heat_capacity_beta(emd, num, span=[0.25, 2], method='auto',
                           n_samples=1000, pre_n=100, sample_steps=1, seed=None):
    """
    Computes the heat capacity num times by multiplying theta by equaly spaced betas in span)

    :param emd: container.EMData
    Object used for encapsulating data used in the expectation maximisation algorithm.
    :param num: int
    The number of heat capacities to compute, all with different betas.
    :param span: list
    The span for betas
    :param method: str
    'auto' (default), 'exact', 'approx', or 'sampling' — see compute_heat_capacity.
    :param n_samples: int
    Gibbs samples per (beta, time bin) when ``method='sampling'``.
    :param pre_n: int
    burn-in sweeps per bin when ``method='sampling'``.
    :param sample_steps: int
    thinning between retained Gibbs samples.
    :param seed: int or None
    RNG seed for the Gibbs sampler.
    :return: numpy.ndarray
    The heat capacities computed with num different betas.
    """
    betas = numpy.linspace(span[0], span[1], num)
    T, D = emd.theta_s.shape
    if method == 'sampling':
        C = numpy.empty((num, T))
        for k, b in enumerate(betas):
            k_seed = None if seed is None else seed + k
            C[k] = _heat_capacity_sampling(
                b * emd.theta_s, emd.N, emd.order, R=n_samples,
                pre_n=pre_n, sample_steps=sample_steps, seed=k_seed)
        return C
    epsilon = 1e-3
    # Build a (num*T, D) stack so psi only needs to be evaluated three times
    # across all betas (psi, +eps, -eps) — same total inner work, one batched call.
    theta_stack = (betas[:, None, None] * emd.theta_s[None, :, :]).reshape(num * T, D)
    psi = _psi(theta_stack, emd.N, emd.order, method)
    tmp1 = _psi(theta_stack * (1 + epsilon), emd.N, emd.order, method)
    tmp2 = _psi(theta_stack * (1 - epsilon), emd.N, emd.order, method)
    C = ((tmp1 - 2 * psi + tmp2) / (epsilon ** 2)).reshape(num, T)
    return C


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
