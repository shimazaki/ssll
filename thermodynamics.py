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

try:
    import jax as _jax
    import jax.numpy as _jnp
    _HAVE_JAX = True
except ImportError:
    _HAVE_JAX = False


if _HAVE_JAX:

    def _gibbs_pairwise_sweeps_jax(x0, theta_1, J, key, burn_in):
        """JIT'd parallel-chain Gibbs sampler for the pairwise model.

        x0: (T, R, N) float, initial state.
        theta_1: (T, N) float.
        J: (T, N, N) float, symmetric, zero diag.
        key: jax.random.PRNGKey.
        burn_in: static int — full Gibbs sweeps before return.

        Returns x of the same shape as x0 after burn_in sweeps. burn_in is
        a static argument so XLA can specialize the scan length.
        """
        T, R, N = x0.shape

        def neuron_step(x, idx_and_key):
            i, k = idx_and_key
            J_row = _jax.lax.dynamic_index_in_dim(J, i, axis=1, keepdims=False)  # (T, N)
            theta_i = _jax.lax.dynamic_index_in_dim(theta_1, i, axis=1, keepdims=False)  # (T,)
            h = theta_i[:, None] + _jnp.einsum('trj,tj->tr', x, J_row)
            u = _jax.random.uniform(k, (T, R), dtype=x.dtype)
            x_new = (u < _jax.nn.sigmoid(h)).astype(x.dtype)
            x = x.at[:, :, i].set(x_new)
            return x, None

        def sweep_step(x, sweep_key):
            keys = _jax.random.split(sweep_key, N)
            x, _ = _jax.lax.scan(neuron_step, x, (_jnp.arange(N), keys))
            return x, None

        sweep_keys = _jax.random.split(key, burn_in)
        x_final, _ = _jax.lax.scan(sweep_step, x0, sweep_keys)
        return x_final

    _gibbs_pairwise_sweeps_jax = _jax.jit(
        _gibbs_pairwise_sweeps_jax, static_argnames=('burn_in',))


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
    if _HAVE_JAX:
        # JAX path: one JIT'd scan over (sweep, neuron). Whole burn-in runs as
        # a single XLA program; on GPU the (T*R)-wide chains saturate SIMT.
        jax_seed = 0 if seed is None else int(seed) & 0xFFFFFFFF
        key = _jax.random.PRNGKey(jax_seed)
        x_j = _jnp.asarray(x)
        theta_1_j = _jnp.asarray(theta_1)
        J_j = _jnp.asarray(J)
        x_out = _gibbs_pairwise_sweeps_jax(x_j, theta_1_j, J_j, key, burn_in)
        # Block until the device finishes and copy back.
        x = numpy.asarray(x_out.block_until_ready())
    elif _HAVE_NUMBA:
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
        # Energy E_{t,r} = theta_1 . x + 0.5 * x . J . x  (J symmetric, zero diag).
        # Rewriting as a contraction over N avoids materializing the
        # (T, R, N(N-1)/2) pair-product tensor, which would dominate memory
        # at large N or when callers fuse betas into the T axis.
        theta_1 = theta_eff[:, :N]                                  # (T, N)
        iu, ju = numpy.triu_indices(N, 1)
        J = numpy.zeros((T, N, N))
        J[:, iu, ju] = theta_eff[:, N:]
        J[:, ju, iu] = theta_eff[:, N:]
        X_f = X.astype(numpy.float64)
        E_lin = numpy.einsum('ti,tri->tr', theta_1, X_f)
        JX = numpy.einsum('tij,trj->tri', J, X_f)                   # (T, R, N)
        E_pair = 0.5 * numpy.einsum('tri,tri->tr', X_f, JX)
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
    T = emd.T
    thetas = get_theta_samples(emd, samples)            # (T, D, samples)

    # Stack (samples, T) into a single (samples*T, D) batch dim so
    # compute_eta / compute_psi run once over the whole posterior fan-out
    # instead of once per posterior sample. Same total numerical work as
    # the loop, but eliminates the per-sample Python overhead and lets the
    # batched paths (transforms.compute_psi_vec for N<=15, the per-bin
    # TAP/OT loop for N>15) see a (samples*T)-long T axis at once.
    th_stack = numpy.moveaxis(thetas, 2, 0).reshape(samples * T, -1)
    eta_stack, emd.eta_sampled = energies.compute_eta(th_stack, N, O)
    psi_stack = energies.compute_psi(th_stack, N, O)
    eta1_stack = eta_stack[:, :N]
    theta1_stack = energies.compute_ind_theta(eta1_stack)
    psi1_stack = energies.compute_ind_psi(theta1_stack)
    S1_stack = energies.compute_entropy(theta1_stack, eta1_stack, psi1_stack, 1)
    S_pair_stack = energies.compute_entropy(th_stack, eta_stack, psi_stack, 2)
    S_ratio_stack = (S1_stack - S_pair_stack) / (S0 - S_pair_stack) * 100

    S_pair_all = S_pair_stack.reshape(samples, T).T     # (T, samples)
    S_ratio_all = S_ratio_stack.reshape(samples, T).T

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
        # Stack (samples, T) into one big batch axis so the kernel sees a
        # single (samples*T)-wide problem and runs as one GPU/numba launch.
        th_stack = numpy.moveaxis(thetas, 2, 0).reshape(samples * T, D)
        C_flat = _heat_capacity_sampling(
            th_stack, emd.N, emd.order, R=n_samples,
            pre_n=pre_n, sample_steps=sample_steps, seed=seed)
        C = C_flat.reshape(samples, T).T
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
        # Stack betas into the T axis so the kernel sees one (num*T)-wide
        # problem and the whole sweep runs as a single launch.
        theta_stack = (betas[:, None, None]
                       * emd.theta_s[None, :, :]).reshape(num * T, D)
        C_flat = _heat_capacity_sampling(
            theta_stack, emd.N, emd.order, R=n_samples,
            pre_n=pre_n, sample_steps=sample_steps, seed=seed)
        return C_flat.reshape(num, T)
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
