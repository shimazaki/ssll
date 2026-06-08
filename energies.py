"""
Some useful functions.

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
import mean_field
import synthesis
import transforms


def get_energies(emd):
    """ Wrapper function to get all energies.

    :param ssll.container emd:
        ssll-container object for that energies should be computed
    """
    N, O = emd.N, emd.order
    theta = emd.theta_s
    eta, emd.eta_sampled = compute_eta(theta, N, O)
    psi = compute_psi(theta, N, O)
    eta1 = eta[:,:N]
    theta1 = compute_ind_theta(eta1)
    psi1 = compute_ind_psi(theta1)
    emd.U1 = compute_internal_energy(theta1, eta1)
    emd.S1 = compute_entropy(theta1, eta1, psi1, 1)
    emd.eta = eta
    emd.psi = psi
    emd.U2 = compute_internal_energy(theta, eta)
    emd.S2 = compute_entropy(theta, eta, psi, 2)
    emd.S_ratio = (emd.S1 - emd.S2)/emd.S1
    emd.dkl = compute_dkl(eta, emd.theta_s, psi, theta1, psi1, N)
    emd.llk1 = compute_likelihood(emd.y[:,:N], theta1, psi1, emd.R)
    emd.llk2 = compute_likelihood(emd.y, theta, psi, emd.R)


def compute_ind_eta(theta):
    """ Computes analytically eta from theta for independent model.

    :param numpy.ndarray theta:
        (t, c) array with natural parameters
    :return numpy.ndarray:
        (t, c) array with expectation parameters parameters
    """
    eta = 1./(1. + numpy.exp(-theta))
    return eta


def compute_ind_theta(eta):
    """ Computes analytically theta from eta for independent model.

    :param numpy.ndarray eta:
        (t, c) array with expectation parameters
    :return numpy.ndarray:
        (t, c) array with natural parameters parameters
    """
    theta = numpy.log(eta/(1. - eta))
    return theta


def compute_ind_psi(theta):
    """ Computes analytically psi from theta for independent model.

    :param numpy.ndarray theta:
        (t, c) array with natural parameters
    :return numpy.ndarray:
        (t,) with solution for log-partition function
    """
    return numpy.sum(numpy.log(1. + numpy.exp(theta)), axis=1)


def compute_eta(theta, N, O, R=1000):
    """ Computes eta from given theta.

    :param numpy.ndarray theta:
        (t, d) array with natural parameters
    :param int N:
        number of cells
    :param int O:
        order of model
    :param int R:
        trials that should be sampled to estimate eta
    :return numpy.ndarray, list:
        (t, d) array with natural parameters parameters and a list with indices of bins, for which has been sampled

    Details: Tries to estimate eta by solving the forward problem from TAP. However, if it fails we fall back to
    sampling. For networks with less then 15 neurons exact solution is computed and for first order analytical solution
    is used.
    """
    T, D = theta.shape
    eta = numpy.empty(theta.shape)
    bins_to_sample = []
    if O == 1:
        eta = compute_ind_eta(theta[:,:N])
    elif O == 2:
        # if few cells compute exact rates
        if N > 15:
            for i in range(T):
                # try to solve forward problem
                try:
                    eta[i] = mean_field.forward_problem(theta[i], N, 'TAP')
                # if it fails remember bin for sampling
                except Exception:
                    bins_to_sample.append(i)
            if len(bins_to_sample) != 0:
                theta_to_sample = numpy.empty([len(bins_to_sample), D])
                for idx, bin2sampl in enumerate(bins_to_sample):
                    theta_to_sample[idx] = theta[bin2sampl]
                spikes = synthesis.generate_spikes_gibbs_parallel(theta_to_sample, N, O, R, sample_steps=100)
                eta_from_sample = transforms.compute_y(spikes, O)
                for idx, bin2sampl in enumerate(bins_to_sample):
                    eta[bin2sampl] = eta_from_sample[idx]

        # if large ensemble approximate
        else:
            transforms.initialise(N, O)
            P = transforms.compute_p_vec(theta)  # (T, 2**N)
            for i in range(T):
                eta[i] = transforms.compute_eta(P[i])

    return eta, bins_to_sample


def compute_psi(theta, N, O, R=1000, estimator='ais'):
    """ Computes psi from given theta.

    :param numpy.ndarray theta:
        (t, d) array with natural parameters
    :param int N:
        number of cells
    :param int O:
        order of model
    :param int R:
        trials that should be sampled to estimate eta (legacy arg, kept
        for backwards compatibility; unused by the current estimators)
    :param str estimator:
        which approximate estimator to use when N > 15.  One of:
            'ais' (default) -- annealed importance sampling, ais_estimator;
            'ot'            -- Ogata-Tanemura with TAP eta, ot_estimator.
        For N <= 15 the exact 2^N enumeration is always used regardless of
        this argument.
    :return numpy.ndarray:
        (t,) array with log-partition

    For first order the analytical solution is used. For networks with 15
    units or less the exact solution is computed.  For N > 15 the requested
    approximate estimator is invoked along the linear path from the
    matched-H independent Bernoulli model to theta.
    """
    T = theta.shape[0]
    psi = numpy.empty(T)

    if O == 1:
        psi = compute_ind_psi(theta[:, :N])
    if O == 2:
        # if few cells compute exact result
        if N > 15:
            theta0 = numpy.copy(theta)
            theta0[:, N:] = 0
            psi0 = compute_ind_psi(theta0[:, :N])
            if estimator == 'ais':
                for i in range(T):
                    psi[i] = ais_estimator(theta0[i], psi0[i], theta[i], N, O,
                                           seed=i)
            elif estimator == 'ot':
                for i in range(T):
                    psi[i] = ot_estimator(theta0[i], psi0[i], theta[i], N, O, N)
            else:
                raise ValueError(
                    "compute_psi: unknown estimator %r (expected 'ais' or 'ot')"
                    % (estimator,))
        # else exact
        else:
            transforms.initialise(N, 2)
            psi = transforms.compute_psi_vec(theta)
    return psi


def ais_estimator(th0, psi0, th1, N, O, S=500, T=1000, seed=0):
    """Annealed importance sampling estimator of psi(th1) (Neal 2001).

    Bridges from a tractable reference distribution at th0 (typically the
    matched-H independent Bernoulli model, with theta_J = 0 and psi0 in
    closed form) to th1 along the linear path

        theta(alpha) = th0 + alpha * (th1 - th0),    alpha in [0, 1],

    with T annealing steps and one full Gibbs sweep per step.  S
    independent chains are propagated in parallel, initialised by exact
    samples from the independent model at th0.  Returns

        psi_hat = psi0 + logsumexp(log_w) - log(S),

    which is an unbiased estimator of psi(th1) - psi(th0) plus psi0.

    Unlike ot_estimator, AIS uses no Plefka anchor at intermediate
    theta(alpha), so its accuracy does not degrade when the rescaled
    model crosses the critical line of the inner mean-field solver.

    :param numpy.ndarray th0:
        (d,) array, reference natural parameters (theta_J = 0).
    :param float psi0:
        psi corresponding to th0 (closed-form independent log-partition).
    :param numpy.ndarray th1:
        (d,) array, target natural parameters.
    :param int N:
        number of cells.
    :param int O:
        order of interactions (currently O=2 only).
    :param int S:
        number of independent AIS chains.
    :param int T:
        number of annealing steps along the path.
    :param int seed:
        RNG seed for chain initialisation and Gibbs proposals.
    :returns float:
        AIS estimate of psi(th1).
    """
    if O != 2:
        raise NotImplementedError("ais_estimator currently supports O=2 only")
    from scipy.special import logsumexp

    rng = numpy.random.default_rng(seed)
    h0 = th0[:N]
    h1 = th1[:N]
    # symmetric coupling matrix for theta1 (zero diagonal), built from upper triangle
    triu = numpy.triu_indices(N, k=1)
    J0 = numpy.zeros((N, N))
    J1 = numpy.zeros((N, N))
    J0[triu] = th0[N:]; J0 += J0.T
    J1[triu] = th1[N:]; J1 += J1.T

    dh = h1 - h0      # field increment along the path (zero for matched-H bridge)
    dJ = J1 - J0      # coupling increment

    # Initial samples: exact independent Bernoulli draws from the th0 model.
    # We assume th0_J = 0 (legacy convention shared with ot_estimator); if not,
    # the user is responsible for an exact-samplable reference distribution.
    p0 = 1.0 / (1.0 + numpy.exp(-h0))
    x = (rng.random((S, N)) < p0[None, :]).astype(numpy.float64)

    alphas = numpy.linspace(0.0, 1.0, T + 1)
    log_w = numpy.zeros(S)

    for t in range(1, T + 1):
        da = alphas[t] - alphas[t - 1]
        # log p_alpha(x) - log p_{alpha-da}(x) = da * [dh . x + 0.5 x^T dJ x]
        lin = x.dot(dh)
        quad = 0.5 * numpy.einsum('si,si->s', x, x.dot(dJ))
        log_w += da * (lin + quad)
        # Gibbs sweep at alpha = alphas[t]
        a_now = alphas[t]
        h_eff = h0 + a_now * dh
        J_eff = J0 + a_now * dJ
        for i in range(N):
            field = h_eff[i] + x.dot(J_eff[:, i])
            p_i = 1.0 / (1.0 + numpy.exp(-field))
            x[:, i] = (rng.random(S) < p_i).astype(numpy.float64)

    return float(psi0 + logsumexp(log_w) - numpy.log(S))


def ot_estimator(th0, psi0, th1, N, O, K, expansion='TAP'):
    """ Uses the Ogata-Tanemura Estimator for estimation (Huang, 2001)

    :param numpy.ndarray th0:
        (1,d) array with theta distribution where psi is known
    :param float psi0
        psi corresponding to th0
    :param th1:
        thetas for which one wants to compute psi
    :param int N:
        number of cells
    :param int O:
        order of interactions
    :param int K:
        points of integration

    :returns
        estimation of psi to th1

    Tries to solve the forward problem at each point and samples if it fails.
    """

    # compute difference between th0 and th1
    dth = th1 - th0
    # points of integration
    int_points = numpy.linspace(0,1,K)
    # All K integration thetas at once: (K, D)
    th_batch = th0[None, :] + int_points[:, None] * dth[None, :]
    eta_batch = mean_field.forward_problem_hessian_batch(th_batch, N)
    # negative derivative of energy function at each integration point
    avg_dUs = eta_batch.dot(dth)
    points_to_sample = []

    # weights for trapezoidal intergration rule
    w = numpy.ones(K)/K
    w[0] /= K
    w[-1] /= K
    # compute estimation of psi
    return psi0 + numpy.dot(w, avg_dUs)


def compute_internal_energy(theta, eta):
    """ Computes the internal energy of the system.

    :param numpy.ndarray theta:
        (t, d) array with natural parameters
    :param numpy.ndarray eta:
        (t, d) array with expectation parameters
    :return numpy.ndarray:
        (t,) array with internal energy at each time bin
    """
    U = -numpy.sum(theta*eta, axis=1)
    return U


def compute_entropy(theta, eta, psi, O):
    """ Computes the entropy of the system.

    :param numpy.ndarray theta:
        (t, d) array with natural parameters
    :param numpy.ndarray eta:
        (t, d) array with expectation parameters
    :param numpy.ndarray psi:
        (t,) array with log-partition function
    :param int O:
        order of model
    :return numpy.ndarray:
        (t,) array with entropy at each time bin
    """

    if O == 1:
        S = -numpy.sum(eta*numpy.log(eta) + (1 - eta)*numpy.log(1 - eta), axis=1)
    else:
        U = compute_internal_energy(theta, eta)
        F = -psi
        S = U - F

    return S

def compute_dkl(eta2, theta2, psi2, theta1, psi1, N):
    """ Computes Kullback Leibler Divergence between pairwise and independent
    model.

    :param numpy.ndarray eta2:
        (t, d) array containing expectations of the pairwise model.
    :param numpy.ndarray  theta2:
        (t,d) array containing theta parameters of the pairwise model
    :param numpy.ndarray  psi2:
        (t) array containing the log partition values of pairwise model.
    :param numpy.ndarray  theta1:
        (t,c) array containing theta parameters of the independent model
    :param numpy.ndarray  psi1:
        (t) array containing log partition values for the independent model
    :param int N:
        number of cells

    :return:
        (t) array with Kullback Leibler Divergen
    """
    dtheta = numpy.copy(theta2)
    dtheta[:,:N] = theta2[:,:N] - theta1
    dkl = numpy.sum(eta2*dtheta, axis=1) - (psi2 - psi1)
    return dkl

def compute_likelihood(y, theta, psi, R):
    """ Computes the likelihood of data for a model

    :param numpy.ndarray y:
        (t,d) array containing empirical expectations of data
    :param numpy.ndarray theta:
        (t,d) array containing theta parameters of model
    :param numpy.ndarray psi:
        (t) array of log partition function
    :param int R:
        number of trials
    :return:
    """
    llk = R*(numpy.sum(y*theta, axis=1) - psi)
    return llk
