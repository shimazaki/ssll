__author__ = 'christian'


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
    psi, emd.psi_sampled = compute_psi(theta, N, O)
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
                eta_from_sample = transforms.compute_y(spikes, O, 1)
                for idx, bin2sampl in enumerate(bins_to_sample):
                    eta[bin2sampl] = eta_from_sample[idx]

        # if large ensemble approximate
        else:
            transforms.initialise(N, O)
            for i in range(T):
                p = transforms.compute_p(theta[i])
                eta[i] = transforms.compute_eta(p)

    return eta, bins_to_sample


def compute_psi(theta, N, O, R=1000):
    """ Computes psi from given theta.

    :param numpy.ndarray theta:
        (t, d) array with natural parameters
    :param int N:
        number of cells
    :param int O:
        order of model
    :param int R:
        trials that should be sampled to estimate eta
    :return numpy.ndarray, list:
        (t, d) array with log-partition and a list with indices of bins, for which has been sampled

    For first order the analytical solution is used. For networks with 15 units and less the exact solution is computed.
    Otherwise, the Ogata-Tanemura-Estimator is used. It tries to solve the forward problem and samples where it fails.
    """
    T = theta.shape[0]
    bins_sampled = []
    psi = numpy.empty(T)

    if O == 1:
        psi = compute_ind_psi(theta[:,:N])
    if O == 2:
        # if few cells compute exact result
        if N > 15:
            theta0 = numpy.copy(theta)
            theta0[:,N:] = 0
            psi0 = compute_ind_psi(theta0[:,:N])
            for i in range(T):
                psi[i], sampled = ot_estimator(theta0[i], psi0[i], theta[i], N, O, N)
                # save bin if sampled
                if sampled:
                    bins_sampled.append(i)
        # else approximate
        else:
            transforms.initialise(N, 2)
            for i in range(T):
                psi[i] = transforms.compute_psi(theta[i])
    return psi, bins_sampled


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
    sampled = False
    # compute difference between th0 and th1
    dth = th1 - th0
    # points of integration
    int_points = numpy.linspace(0,1,K)
    # array for negative derivatives of Energy function
    avg_dUs = numpy.empty(K)
    # iterate over all integration points
    points_to_sample = []
    for i, int_point in enumerate(int_points):
        # theta point that needs to be evaluated
        th_tmp = th0 + int_point*dth
        # Sample Data
        try:
            eta = mean_field.forward_problem(th_tmp, N, expansion)
        except Exception:
            sampled = True
            points_to_sample.append(i)
        # negative derivative of energy function
        dU = numpy.dot(dth, eta)
        # compute mean
        avg_dUs[i] = numpy.mean(dU)
    if len(points_to_sample) != 0:
        th_to_sample = numpy.empty([len(points_to_sample), th0.shape[0]])

        # get remaining thetas and sample for them
        for j, point2sample in enumerate(points_to_sample):
            th_to_sample[j] = th0 + int_points[point2sample]*dth

        spikes = synthesis.generate_spikes_gibbs_parallel(th_to_sample, N, O, R=1000)
        eta_from_sample = transforms.compute_y(spikes, O, 1)

        for idx, point2sample in enumerate(points_to_sample):
            # negative derivative of energy function
            dU = numpy.dot(dth, eta_from_sample[idx])
            # compute mean
            avg_dUs[point2sample] = numpy.mean(dU)

    # weights for trapezoidal intergration rule
    w = numpy.ones(K)/K
    w[0] /= K
    w[-1] /= K
    # compute estimation of psi
    return psi0 + numpy.dot(w, avg_dUs), sampled


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
    dtheta = numpy.copy(theta2)
    dtheta[:,:N] = theta2[:,:N] - theta1
    dkl = numpy.sum(eta2*dtheta, axis=1) - (psi2 - psi1)
    return dkl

def compute_likelihood(y, theta, psi, R):
    llk = R*(numpy.sum(y*theta, axis=1) - psi)
    return llk