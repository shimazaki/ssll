__author__ = 'Christian Donner'

import numpy
import itertools
from scipy.optimize import fsolve
import max_posterior,energies

eta_FI_map = None

def self_consistent_eq(eta, theta1, theta2, expansion='TAP'):
    """ Generates self-consistent equations for forward problem.

    :param numpy.ndarray eta:
        (c,) vector with individual rates for each cell
    :param numpy.ndarray theta1:
        (c,) vector with first order thetas
    :param numpy.ndarray theta2:
        (c, c) array with second order thetas (theta_ij in row i and column j)
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean field
        and 'TAP' for second order approximation with Osanger correction. (default='TAP')

    :returns:
        list of c equations that have to be solved for getting the first order etas.
    """
    # TAP equations
    if expansion == 'TAP':
        equations = numpy.log(eta) - numpy.log(1 - eta) - theta1 - numpy.dot(theta2, eta) - \
                    .5*numpy.dot((.5 - eta)[:,numpy.newaxis]*theta2**2, (eta - eta**2))
    # Naive Mean field equations
    elif expansion == 'naive':
        equations = numpy.log(eta)- numpy.log(1 - eta) - theta1 - numpy.dot(theta2, eta)

    return equations


def self_consistent_eq_Hinv(eta, theta1, theta2, expansion='TAP'):
    """ Generates self-consistent equations for forward problem.

    :param numpy.ndarray eta:
        (c,) vector with individual rates for each cell
    :param numpy.ndarray theta1:
        (c,) vector with first order thetas
    :param numpy.ndarray theta2:
        (c, c) array with second order thetas (theta_ij in row i and column j)
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean field
        and 'TAP' for second order approximation with Osanger correction. (default='TAP')

    :returns:
        list of c equations that have to be solved for getting the first order etas.
    """
    # TAP equations
    if expansion == 'TAP':
        H_diag =  1./eta + 1./(1 - eta) + .5*numpy.dot(theta2**2, (eta - eta**2))
    # Naive Mean field equations
    elif expansion == 'naive':
        H_diag =  1./eta + 1./(1 - eta)
    Hinv = numpy.diag(1./H_diag)
    return Hinv


def forward_problem_hessian(theta, N):
    """ Gets the etas for given thetas.

    :param numpy.ndarray theta:
        (d,)-dimensional array containing all thetas
    :param int N:
        Number of cells
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean field
        and 'TAP' for second order approximation with Osanger correction.

    :returns:
        (d,) numpy.ndarray with all etas.
    """
    # Initialize eta vector
    eta = numpy.empty(theta.shape)
    eta_max = 0.5*numpy.ones(N)
    # Extract first order thetas
    theta1 = theta[:N]
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Write second order thetas into matrix
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    conv = numpy.inf
    # Solve self-consistent equations and calculate approximation of fisher matrix
    iter_num = 0
    while conv > 1e-4 and iter_num < 500:
        deta = self_consistent_eq(eta_max, theta1=theta1, theta2=theta2, expansion='TAP')
        Hinv = self_consistent_eq_Hinv(eta_max, theta1=theta1, theta2=theta2, expansion='TAP')
        eta_max -= .1*numpy.dot(Hinv, deta)
        conv = numpy.amax(numpy.absolute(deta))
        iter_num += 1
        eta_max[eta_max <= 0.] = numpy.spacing(1)
        eta_max[eta_max >= 1.] = 1. - numpy.spacing(1)
        if iter_num == 500:
            raise Exception('Self consistent equations could not be solved!')

    G_inv = - theta2 - theta2**2*numpy.outer(0.5 - eta_max[:N], 0.5 - eta_max[:N])
    G_inv[diag_idx] = 1./eta_max + 1./(1.-eta_max) + .5*numpy.dot(theta2**2, (eta_max - eta_max**2))
    G = numpy.linalg.inv(G_inv)
    # Compute second order eta
    eta2 = G + numpy.outer(eta_max[:N], eta_max[:N])
    eta[N:] = eta2[triu_idx]
    eta[:N] = eta_max
    eta[eta < 0.] = numpy.spacing(1)
    eta[eta > 1.] = 1. - numpy.spacing(1)
    return eta


def forward_problem(theta, N, expansion):
    """ Gets the etas for given thetas.

    :param numpy.ndarray theta:
        (d,)-dimensional array containing all thetas
    :param int N:
        Number of cells
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean field
        and 'TAP' for second order approximation with Osanger correction.

    :returns:
        (d,) numpy.ndarray with all etas.
    """
    # Initialize eta vector
    eta = numpy.empty(theta.shape)
    # Extract first order thetas
    theta1 = theta[:N]
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Write second order thetas into matrix
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Solve self-consistent equations and calculate approximation of fisher matrix
    if expansion == 'TAP':
        f = lambda x: self_consistent_eq(x, theta1=theta1, theta2=theta2,
                                         expansion='TAP')
        try:
            eta[:N] = fsolve(f, 0.1*numpy.ones(N))
        except Warning:
            raise Exception('scipy.fsolve did not compute reliable result!')
        G_inv = - theta2 - theta2**2*numpy.outer(0.5 - eta[:N], 0.5 - eta[:N])
    elif expansion == 'naive':
        f = lambda x: self_consistent_eq(x, theta1=theta1, theta2=theta2,
                                         expansion='naive')
        try:
            eta[:N] = fsolve(f, 0.1*numpy.ones(N))
        except Warning:
            raise Exception('scipy.fsolve did not compute reliable result!')
        G_inv = - theta2

    # Compute Inverse of Fisher
    G_inv[diag_idx] = 1./(eta[:N]*(1-eta[:N]))
    G = numpy.linalg.inv(G_inv)
    # Compute second order eta
    eta2 = G + numpy.outer(eta[:N], eta[:N])
    eta[N:] = eta2[triu_idx]
    return eta


def backward_problem(y_t, N, expansion, diag_weight_trick=True):
    """ Calculates thetas for given etas.

    :param numpy.ndarray y_t:
        (d,) dimensional vector containing rates
    :param numpy.ndarray X_t:
        (t,r) dimesional binary array with spikes
    :param int R:
        Number of trials
    :param str expansion:
        String that indicates order of approximantion. 'naive' for naive mean
        field and 'TAP' for second order approximation with Osanger correction.

    """


    # Compute indices
    triu_idx = numpy.triu_indices(N, k=1)
    diag_idx = numpy.diag_indices(N)
    # Compute covariance matrix and invert
    G = compute_fisher_info_from_eta(y_t, N)
    G_inv = numpy.linalg.inv(G[:N,:N])

    # Solve backward problem for indicated approximation
    if expansion == 'TAP':
        # Write the rate into a matrix for dot-products
        y_mat = numpy.zeros([N, N])
        y_mat[triu_idx] = y_t[N:]
        y_mat[triu_idx[1],triu_idx[0]] = y_t[N:]
        # Compute quadratic coefficient of the solution for theta_ij
        quadratic_term = ((.5 - y_mat)*(.5 - y_mat.T)).flatten()
        # Compute linear coefficient of the solution for theta_ij
        linear_term = numpy.ones(quadratic_term.shape, dtype=float)
        # Compute offset of the solution for theta_ijtheta_TAP_wD
        offset = G_inv.flatten()
        # Solve for theta_ij
        theta2_solution = solve_quadratic_problem(quadratic_term, linear_term,
                                                  offset)
        # Bring back to matrix form
        theta2_est = theta2_solution.reshape([N, N])
        theta2_est[diag_idx] = 0
        # Calculate Diagonal
        if diag_weight_trick:
            theta2_est[diag_idx] = compute_diagonal(y_t[:N], theta2_est, G_inv[diag_idx])
        # Initialize array for solution of theta
        theta = numpy.empty(y_t.shape)
        # Fill in theta_ij
        diag_weight = numpy.ones(theta2_est.shape)
        theta[N:] = theta2_est[triu_idx]
        # Compute theta_i
        theta[:N] = numpy.log(y_t[:N]/(1 - y_t[:N])) - \
                        numpy.dot(theta2_est, y_t[:N]) - \
                        0.5*(0.5-y_t[:N])*\
                        numpy.dot(theta2_est**2, y_t[:N]*(1 - y_t[:N]))

    return theta


def compute_diagonal(eta, theta2, G_inv_diag):
    """ Computes the diagonal for the second order theta matrix.

    :param numpy.ndarray eta:
        (c,) vector with all first order rates.
    :param numpy.ndarray theta3:
        (c,c) array with all second order thetas.
    :param G_inv_diag:
        (c,) vector with the diagonal of the Fisher Info.

    :returns:
        (c,) array with solution for theta_ii
    """
    return - 1./(eta*(1 - eta)) - .5*numpy.dot(theta2**2,eta*(1 - eta)) + G_inv_diag


def solve_quadratic_problem(a, b, c):
    """ Solves a quadratic equation of form:

    ax^2 + bx + c = 0

    Selects the solution closest to the naive mean field solution.
    If solution is complex, naive approximation is returned.

    :param numpy.ndarray a:
        d-dimensional vector, where d is the number of equations.
        Vector contains coefficients of quadratic term.
    :param numpy.ndarray b:
        d-dimensional vector, containing coefficients of linear term.
    :param numpy.ndarray c:
        d-dimensional vector, offset.

    :returns:
        x that is closest to naive solution or, if complex, naive mean field approx.
    """
    D = a.shape[0]
    # Get solution without quadratic term
    naive_x = -c/b
    # Compute term below root
    term_in_root = b**2 - 4.*a*c
    # Check where solution is non complex
    non_complex = term_in_root >= 0
    non_complex_idx = numpy.where(non_complex)[0]
    is_complex_idx = numpy.where(numpy.logical_not(non_complex))[0]
    # Initialize array for two solutions
    x_12 = numpy.zeros([D, 2])
    # Compute two solutions
    x_12[non_complex_idx, 0] = (-b[non_complex_idx] - \
                                     numpy.sqrt(term_in_root[non_complex_idx]))/(2.*a[non_complex_idx])
    x_12[non_complex_idx, 1] = (-b[non_complex_idx] + \
                                     numpy.sqrt(term_in_root[non_complex_idx]))/(2.*a[non_complex_idx])
    # Find closest solution
    diff2naive = numpy.absolute(x_12 - naive_x[:, numpy.newaxis])
    closest_x = numpy.argmin(diff2naive, axis=1)
    sol1 = numpy.where(closest_x == 0)[0]
    sol2 = numpy.where(closest_x)[0]
    x = numpy.zeros(D)
    x[sol1] = x_12[sol1, 0]
    x[sol2] = x_12[sol2, 1]
    # Take naive solution where complex
    x[is_complex_idx] = naive_x[is_complex_idx]
    # Return solution
    return x


def compute_psi(theta, eta, N):
    """ Computes TAP approximation of log-partition function.

    :param numpy.ndarray theta:
        (d,) dimensional vector with natural parameters theta.
    :param numpy.ndarray eta:
        (d,) dimensional vector with expectation parameters eta.
    :param int N:
        Number of cells.

    :returns:
        TAP-approximation of log-partition function
    """
    # Get indices
    triu_idx = numpy.triu_indices(N, k=1)
    # Insert second order theta into matrix
    theta2 = numpy.zeros([N,N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Dot product of theta and eta
    psi_trans = numpy.dot(theta[:N], eta[:N])
    # Entropy of independent model
    psi_0 = - numpy.sum(eta[:N]*numpy.log(eta[:N]) + (1 - eta[:N])*numpy.log(1 - eta[:N]))
    # First derivative
    psi_1 = .5*numpy.sum(numpy.dot(theta2, numpy.outer(eta[:N],eta[:N])))
    # Second derivative
    psi_2 = .125*numpy.sum(numpy.dot(theta2, numpy.outer(eta[:N] - eta[:N]**2,eta[:N] - eta[:N]**2)))
    # Return sum of all
    return psi_trans + psi_0 + psi_1 + psi_2


def log_likelihood_mf(eta, theta, R, N):
    """ Compute log-likelihood with TAP estimation of log-partition function.

    :param numpy.ndarray theta:
        (d,) dimensional vector with natural parameters theta.
    :param eta:
        (d,) dimensional vector with expectation parameters eta.
    :param numpy.ndarray y:
        (d,) dimensional vector with empirical rates.
    :param int N:
        Number of cells.

    """
    # Compute TAP estimation of psi
    th0 = numpy.zeros(theta.shape)
    th0[:N] = theta[:N]
    psi0 = numpy.sum(numpy.log(1.+numpy.exp(th0[:N])))
    psi = energies.ot_estimator(th0, psi0, theta, N, 2, N)
    # Return log-likelihood
    return R*(numpy.dot(theta, eta) - psi)


def log_marginal(emd, period=None):
    """
    Computes the log marginal probability of the observed spike-pattern rates
    by marginalising over the natural-parameter distributions. See equation 45
    of the source paper for details.

    This is just a wrapper function for `log_marginal_raw`. It unpacks data
    from the EMD container pbject and calls that function.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    # Unwrap the parameters and call the raw function
    log_p = log_marginal_raw(emd.theta_f, emd.theta_o, emd.sigma_f, emd.sigma_o_inv,
        emd.y, emd.R, emd.N, period)

    return log_p


def log_marginal_raw(theta_f, theta_o, sigma_f, sigma_o_inv, y, R, N, period=None):
    """
    Computes the log marginal probability of the observed spike-pattern rates
    by marginalising over the natural-parameter distributions. See equation 45
    of the source paper for details.

    From within SSLL, this function should be accessed by calling
    `log_marginal` with the EMD container as a parameter. This raw function is
    designed to be called from outside SSLL, when a complete EMD container
    might not be available.

    See the container.py for a full description of the parameter properties.

    :param period tuple:
        Timestep range over which to compute probability.

    :returns:
        Log marginal probability of the synchrony estimate as a float.
    """
    if period == None: period = (0, theta_f.shape[0])
    # Initialise
    log_p = 0
    # Iterate over each timestep and compute...
    a, b = 0, 0
    for i in range(period[0], period[1]):
        a += log_likelihood_mf(y[i], theta_f[i], R, N)
        theta_d = theta_f[i] - theta_o[i]
        b -= numpy.dot(theta_d, sigma_o_inv[i]*theta_d)
        b += numpy.sum(numpy.log(sigma_f[i])) +\
             numpy.sum(numpy.log(sigma_o_inv[i]))
    log_p = a + b / 2

    return log_p


def compute_fisher_info_from_eta(eta, N):
    """ Creates Fisher-Information matrix from eta-vector.

    :param numpy.ndarray eta:
        vector with rates and coincidence rates
    :param int N:
        number of cells

    :returns:
        Fisher-information matrix as numpy.ndarray
    """
    # Initialize matrix for the first part of the fisher matrix
    G1 = numpy.zeros([N, N])
    # Get upper triangle indices
    triu_idx = numpy.triu_indices(N, k=1)
    # Construct first part from eta
    G1[triu_idx] = eta[N:]
    G1 += G1.T
    G1 += numpy.diag(eta[:N])
    # Second part of fisher information matrix
    G2 = numpy.outer(eta[:N], eta[:N])
    # Final matrix
    G = G1 - G2

    return G


def compute_full_G(eta, theta, N):
    """ Computes Fisher Matrix with all information up to order 3 using eta_FI_map.

    :param numpy.ndarray eta:
        (d,) vector with all rates (up to second order)
    :param numpy.ndarray theta:
        (d,) vector with all natural parameters (up to second order)
    :param int N:
        Number of cells

    :returns:
        Fisher Matrix with all entries that require not more than third-order rates
    """
    eta1 = eta[:N]
    D = N + N*(N-1)/2
    eta3 = estimate_higher_order_eta(eta, N, 3)
    eta4 = estimate_higher_order_eta(eta, N, 4)
    eta_full = numpy.hstack([eta, eta3, eta4])
    G2 = numpy.outer(eta,eta)
    G = numpy.zeros([D,D])
    G[eta_FI_map[0]] = eta_full[numpy.array(eta_FI_map[1],dtype=int)] - G2[eta_FI_map[0]]
    return G

def compute_full_G_second_order(eta, theta, N):
    """ Computes Fisher Matrix with all information up to order 3 using eta_FI_map.

    :param numpy.ndarray eta:
        (d,) vector with all rates (up to second order)
    :param numpy.ndarray theta:
        (d,) vector with all natural parameters (up to second order)
    :param int N:
        Number of cells

    :returns:
        Fisher Matrix with all entries that require not more than third-order rates
    """
    D = N + N*(N-1)/2
    G2 = numpy.outer(eta,eta)
    G = numpy.zeros([D,D])
    G[eta_FI_map[0]] = eta[numpy.array(eta_FI_map[1],dtype=int)] - G2[eta_FI_map[0]]
    return G


def create_eta_FI_map_second_order(N, O=3):
    """ Computes the Index map to fill the FI matrix

    :param int N:
        Number of cells
    :param int O:
        Up to which order Information should be used
    """
    global eta_FI_map
    eta_FI_map = [[numpy.array([], dtype=int),numpy.array([], dtype=int)],numpy.array([], dtype=int)]
    eta1_idx = range(N)
    eta2_idx = range(N, N + N*(N-1)/2)
    #eta3_idx = range(N + N*(N-1)/2, N + N*(N-1)/2 + N*(N-1)*(N-2)/6)
    #eta4_idx = range(N + N*(N-1)/2 + N*(N-1)*(N-2)/6, N + N*(N-1)/2 + N*(N-1)*(N-2)/6 + N*(N-1)*(N-2)*(N-3)/24)
    triu_idx = numpy.triu_indices(N, k=1)
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx])
    d_n2_idx = 0
    #d_n3_idx = 0
    #d_n4_idx = 0
    for n in range(N-1):
        d_n2 = N - (n+1)
        # Horizontals between first and second order
        eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],numpy.tile(numpy.array([n]), d_n2)])
        eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(N+d_n2_idx,N+d_n2_idx+d_n2)])
        eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx[d_n2_idx:d_n2_idx+d_n2]])

        if n != N-1:
            # Diagonals between first and second order (second order etas)
            diag_idx = numpy.diag_indices(d_n2)
            eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], diag_idx[0]+(N-d_n2)])
            eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], diag_idx[1]+N+d_n2_idx])
            eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx[d_n2_idx:d_n2_idx+d_n2]])
            # Off Diagonals (third order etas)
            #triu_idx = numpy.triu_indices(d_n2, k=1)
            #d_n3 = triu_idx[0].shape[0]
            #eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+(N-d_n2)])
            #eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+N+d_n2_idx])
            #eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Also the transposed
            #eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[1]+(N-d_n2)])
            #eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[0]+N+d_n2_idx])
            #eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Entries above the horizontals between first and second order thetas (Third order etas)
            #if n < N-2:
            #    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], numpy.tile(numpy.array([n]),N*(N-1)/2 - d_n2_idx-d_n2) ])
            #    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(N+d_n2_idx+d_n2, N + N*(N-1)/2)])
            #    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Off Diagonals Close to main diagonal (Third order etas)
            #eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+N+d_n2_idx])
            #eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+N+d_n2_idx])
            #eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Higher order part
            #G_x_offset = N+d_n2_idx
            #G_y_offset = N+d_n2_idx+d_n2
            #d_n2_tmp = d_n2 - 1
            #d_n2_idx_tmp = 0
            #for pair_idx in range(N-n-2):
            #    # Horizontal (third order eta)
            #    eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n3+d_n2_idx_tmp+d_n2_tmp]
            #    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], numpy.tile(numpy.array([pair_idx+G_x_offset]),d_n2_tmp)])
            #    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(d_n2_idx_tmp+G_y_offset,d_n2_idx_tmp + d_n2_tmp+G_y_offset)])
            #    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n2_idx_tmp+d_n2_tmp]])
            #    # Diagonals (thrid order)
            #    diag_idx = numpy.diag_indices(d_n2_tmp)
            #    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], diag_idx[0]+pair_idx+1+G_x_offset])
            #    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], diag_idx[1]+d_n2_idx_tmp+G_y_offset])
            #    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n2_idx_tmp+d_n2_tmp]])
            #    # Off diagonals (fourth order)
            #    triu_idx = numpy.triu_indices(d_n2_tmp, k=1)
            #    num_of_quadruplets = len(triu_idx[0])
            #    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+pair_idx+1+G_x_offset])
            #    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+d_n2_idx_tmp+G_y_offset])
            #    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1], eta4_idx[d_n4_idx:d_n4_idx + num_of_quadruplets]])
            #    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[1]+pair_idx+1+G_x_offset])
            #    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[0]+d_n2_idx_tmp+G_y_offset])
            #    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1], eta4_idx[d_n4_idx:d_n4_idx + num_of_quadruplets]])
            #    d_n4_idx += num_of_quadruplets
            #    d_n2_idx_tmp += d_n2_tmp
            #    d_n2_tmp -= 1

        #d_n3_idx += d_n3
        d_n2_idx += d_n2
    # Transpose
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],eta_FI_map[0][1]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1],eta_FI_map[0][0][:eta_FI_map[0][1].shape[0]]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta_FI_map[1]])
    # Main Diagonal
    diag_idx = numpy.diag_indices(N + N*(N-1)/2)
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],diag_idx[0]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1],diag_idx[1]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],range(N + N*(N-1)/2)])


def create_eta_FI_map(N, O=3):
    """ Computes the Index map to fill the FI matrix

    :param int N:
        Number of cells
    :param int O:
        Up to which order Information should be used
    """
    global eta_FI_map
    eta_FI_map = [[numpy.array([], dtype=int),numpy.array([], dtype=int)],numpy.array([], dtype=int)]
    eta1_idx = range(N)
    eta2_idx = range(N, N + N*(N-1)/2)
    eta3_idx = range(N + N*(N-1)/2, N + N*(N-1)/2 + N*(N-1)*(N-2)/6)
    eta4_idx = range(N + N*(N-1)/2 + N*(N-1)*(N-2)/6, N + N*(N-1)/2 + N*(N-1)*(N-2)/6 + N*(N-1)*(N-2)*(N-3)/24)
    triu_idx = numpy.triu_indices(N, k=1)
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx])
    d_n2_idx = 0
    d_n3_idx = 0
    d_n4_idx = 0
    for n in range(N-1):
        d_n2 = N - (n+1)
        # Horizontals between first and second order
        eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],numpy.tile(numpy.array([n]), d_n2)])
        eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(N+d_n2_idx,N+d_n2_idx+d_n2)])
        eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx[d_n2_idx:d_n2_idx+d_n2]])

        if n != N-1:
            # Diagonals between first and second order (second order etas)
            diag_idx = numpy.diag_indices(d_n2)
            eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], diag_idx[0]+(N-d_n2)])
            eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], diag_idx[1]+N+d_n2_idx])
            eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta2_idx[d_n2_idx:d_n2_idx+d_n2]])
            # Off Diagonals (third order etas)
            triu_idx = numpy.triu_indices(d_n2, k=1)
            d_n3 = triu_idx[0].shape[0]
            eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+(N-d_n2)])
            eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+N+d_n2_idx])
            eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Also the transposed
            eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[1]+(N-d_n2)])
            eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[0]+N+d_n2_idx])
            eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Entries above the horizontals between first and second order thetas (Third order etas)
            if n < N-2:
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], numpy.tile(numpy.array([n]),N*(N-1)/2 - d_n2_idx-d_n2) ])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(N+d_n2_idx+d_n2, N + N*(N-1)/2)])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Off Diagonals Close to main diagonal (Third order etas)
            eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+N+d_n2_idx])
            eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+N+d_n2_idx])
            eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx:d_n3_idx+d_n3]])
            # Higher order part
            G_x_offset = N+d_n2_idx
            G_y_offset = N+d_n2_idx+d_n2
            d_n2_tmp = d_n2 - 1
            d_n2_idx_tmp = 0
            for pair_idx in range(N-n-2):
                # Horizontal (third order eta)

                eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n3+d_n2_idx_tmp+d_n2_tmp]
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], numpy.tile(numpy.array([pair_idx+G_x_offset]),d_n2_tmp)])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], range(d_n2_idx_tmp+G_y_offset,d_n2_idx_tmp + d_n2_tmp+G_y_offset)])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n2_idx_tmp+d_n2_tmp]])
                # Diagonals (thrid order)
                diag_idx = numpy.diag_indices(d_n2_tmp)
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], diag_idx[0]+pair_idx+1+G_x_offset])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], diag_idx[1]+d_n2_idx_tmp+G_y_offset])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta3_idx[d_n3_idx+d_n2_idx_tmp:d_n3_idx+d_n2_idx_tmp+d_n2_tmp]])
                # Off diagonals (fourth order)
                triu_idx = numpy.triu_indices(d_n2_tmp, k=1)
                num_of_quadruplets = len(triu_idx[0])
                #print n, G_x_offset, pair_idx, G_y_offset, num_of_quadruplets, d_n4_idx, d_n2_tmp
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[0]+pair_idx+1+G_x_offset])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[1]+d_n2_idx_tmp+G_y_offset])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1], eta4_idx[d_n4_idx:d_n4_idx + num_of_quadruplets]])
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], triu_idx[1]+pair_idx+1+G_x_offset])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], triu_idx[0]+d_n2_idx_tmp+G_y_offset])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1], eta4_idx[d_n4_idx:d_n4_idx + num_of_quadruplets]])
                # above horizontals (fourth order)
                eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0], numpy.tile(numpy.array([G_x_offset + pair_idx], dtype=int), num_of_quadruplets)])
                eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1], numpy.arange(N+N*(N-1)/2 - num_of_quadruplets, N+N*(N-1)/2)])
                eta_FI_map[1] = numpy.concatenate([eta_FI_map[1], eta4_idx[d_n4_idx:d_n4_idx + num_of_quadruplets]])
                d_n4_idx += num_of_quadruplets
                d_n2_idx_tmp += d_n2_tmp
                d_n2_tmp -= 1

        d_n3_idx += d_n3
        d_n2_idx += d_n2
    # Transpose
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],eta_FI_map[0][1]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1],eta_FI_map[0][0][:eta_FI_map[0][1].shape[0]]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],eta_FI_map[1]])
    # Main Diagonal
    diag_idx = numpy.diag_indices(N + N*(N-1)/2)
    eta_FI_map[0][0] = numpy.concatenate([eta_FI_map[0][0],diag_idx[0]])
    eta_FI_map[0][1] = numpy.concatenate([eta_FI_map[0][1],diag_idx[1]])
    eta_FI_map[1] = numpy.concatenate([eta_FI_map[1],range(N + N*(N-1)/2)])


def estimate_higher_order_eta(eta, N, order):
    subpops = list(itertools.combinations(range(N), order))
    pairs_in_subpops = []
    for i in subpops:
        pairs_in_subpops.append(list(itertools.combinations(i, 2)))
    pair_array = numpy.array(pairs_in_subpops)
    sub_pops_array = numpy.array(subpops)
    eta2 = numpy.zeros([N,N])
    triu_idx = numpy.triu_indices(N,1)
    eta2[triu_idx] = eta[N:]
    eta2 += eta2.T
    log_eta_a = numpy.sum(numpy.log(eta2[pair_array[:,:,0],pair_array[:,:,1]]), axis=1)
    eta1 = eta[:N]
    log_eta_b = numpy.sum(numpy.log(eta1[sub_pops_array])*(order-2), axis=1)
    return numpy.exp(log_eta_a - log_eta_b)


def compute_higher_order_etas(eta1, theta2, O):
    """ Approximates higher order thetas by mean-field approximation.

    :param numpy.ndarray eta1:
        (c,) vector with first-order rates
    :param numpy.ndarray theta2:
        (d-c,) vector with second order thetas
    :param int O:
        Order for that the rates should be computed.

    :returns:
        numpy.ndarray with approximatin of higher order rates
    """

    # Initialize all necessary parameters
    N = eta1.shape[0]
    triu_idx = numpy.triu_indices(N, k=1)
    theta2_mat = numpy.zeros([N,N])
    theta2_mat[triu_idx] = theta2
    theta2_mat += theta2_mat.T
    # Get all subpopulations for that the rates should be computed
    subpopulations = list(itertools.combinations(range(N),O))
    # Get Connections within the subpopulation (PROBABLY MORE ELEGANT WAY POSSIBLE)
    pairs_in_subpopulations = []
    for i in subpopulations:
        pairs_in_subpopulations.append(list(itertools.combinations(i, 2)))
    # Get Indices for the pairs in an NxN matrix
    pair_idx = numpy.ravel_multi_index(numpy.array(pairs_in_subpopulations).T, [N,N])
    # Compute independent rates of the subpopulations
    ind_rates = numpy.prod(eta1[subpopulations], axis = 1)
    # Compute the terms of the first derivative responsible for pairs within subpopulation
    terms_within_subpopulation = theta2_mat*(1-numpy.outer(eta1, eta1))
    # Extract the values for each pair within each subpopulation and sum over it
    # (Note that 0.5 is dropped because we consider each pair just once!)
    first_div1 = numpy.sum(terms_within_subpopulation.flatten()[pair_idx], axis=0)
    # Product of theta_ij and eta_j
    theta2_eta_j = theta2_mat*eta1[:,numpy.newaxis]
    # Get the eta_i's for each subpopulation
    eta_i = eta1[subpopulations]
    # Get the connections for each neuron in each subpopulation to units outside the population
    theta2_eta_j_pairs =  theta2_eta_j[subpopulations,:]
    # neighbors of subpopulation
    terms_neighboring_subpopulation = theta2_eta_j_pairs*eta_i[:,:,numpy.newaxis]
    # Compute first derivative
    first_derivative = (first_div1 + numpy.sum(numpy.sum(terms_neighboring_subpopulation,axis=2),axis=1))*ind_rates
    # Compute approximation of higher order rates and return
    higher_order_rates = ind_rates + first_derivative
    return higher_order_rates


def mean_field_nr(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, epsilon=.1, diag_weight_trick=False):
    """ Newton Raphson algorithm with TAP approximation.

    :param numpy.ndarray y_t:
        (d,) dimensional vector with empirical rates of data.
    :param numpy.ndarray X_t:
        (r,c) dimensional array with spike trains for each trial and each cell.
    :param int R:
        Number of trials.
    :param numpy.ndarray theta0:
        Starting point for theta.
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix
    :param float epsilon:
        Learning rate for gradient steps. (Default=1e-1)
    :param bool diag_weight_trick:
        if diagonal weight should be used. (Default=False)

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    """

    # Convergence limit for the rate
    GA_CONVERGENCE = 1e-4
    MAX_GA_ITERATIONS = 1000.
    # Get number of Neurons
    N = X_t.shape[1]
    # Initial value
    max_dlpo = numpy.inf
    # Solve backward problem without prior
    theta_max = backward_problem(y_t, N, 'TAP', diag_weight_trick=False)
    # Set current eta
    eta = numpy.copy(y_t)
    # Initialize iteration counter
    iterations = 0
    # Until gradient converges
    while max_dlpo > GA_CONVERGENCE and iterations<MAX_GA_ITERATIONS:
        # Count iterations
        iterations += 1
        # Get gradient of posterior
        dlpo = R*(y_t - eta) - numpy.dot(sigma_o_i, theta_max - theta_o)
        # Compute Fisher info
        G = compute_full_G(eta, theta_max, N)
        ddlpo = - R*G - sigma_o_i
        # Get theta change
        dth = -numpy.dot(numpy.linalg.inv(ddlpo), dlpo)
        # Update theta
        theta_max += epsilon*dth
        # Get new eta by solving forward problem
        eta = forward_problem_hessian(theta_max, N, 'TAP')
        max_dlpo = numpy.amax(numpy.absolute(dlpo))/R
        # Break if max. number of iterations is reached.
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The Mean Field NR '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')
    # Calculate Fisher info
    G = compute_full_G(eta,theta_max,N)
    ddlpo = -R*G - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)
    # Return
    return theta_max, -ddlpo_i


def mean_field_cg(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, diag_weight_trick=False):
    """ Conjugate Gradient algorithm with TAP approximation.

    :param numpy.ndarray y_t:
        (d,) dimensional vector with empirical rates of data.
    :param numpy.ndarray X_t:
        (r,c) dimensional array with spike trains for each trial and each cell.
    :param int R:
        Number of trials.
    :param numpy.ndarray theta0:
        Starting point for theta.
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix
    :param bool diag_weight_trick:
        if diagonal weight should be used. (Default=False)

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    """

    # Convergence limit for the rate
    GA_CONVERGENCE = 1e-4
    MAX_GA_ITERATIONS = 1000.
    # Get number of Neurons
    N = X_t.shape[1]
    # Initialize iteration counter
    iterations = 0
    # Solve backward problem without prior
    theta_max = backward_problem(y_t, N, 'TAP', diag_weight_trick=False)
    # Get Gradient (just prior beacuse the one of llk is zero)
    dlpo = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # Change of theta
    d_th = dlpo
    # Search direction
    s = d_th
    # Get initial eta
    eta = numpy.copy(y_t)
    # Convergence criterion
    max_dlpo = numpy.amax(numpy.absolute(dlpo))/R
    # Until rate converged
    while max_dlpo > GA_CONVERGENCE and iterations<MAX_GA_ITERATIONS:
        # Save old theta
        d_th_prev = d_th
        # Count iterations
        iterations += 1
        # Get new theta direction
        d_th = dlpo
        # Get Beta
        beta = max_posterior.compute_beta(d_th, d_th_prev, s, 'HS')
        # Update search direction
        s = d_th + beta*s
        # Update theta by line search
        theta_max = mean_field_line_search(theta_max, N, R, s, dlpo, sigma_o_i, eta)
        # Get eta by solving forward problem
        eta = forward_problem_hessian(theta_max, N, 'TAP')
        # Get gradient of posterior
        dlpo = R*(y_t - eta) - numpy.dot(sigma_o_i, theta_max - theta_o)
        # Get maximal Gradient entry
        max_dlpo = numpy.amax(numpy.absolute(dlpo))/R
        # If maximal number of iterations is reached break
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The Mean Field CG '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Calculate Fisher info
    G = compute_full_G(eta, theta_max, N)
    ddlpo = -R*G - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo + 1e-13*numpy.identity(ddlpo.shape[0]))
    # Return
    return theta_max, -ddlpo_i


def mean_field_bfgs(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, diag_weight_trick=False):
    """ BFGS algorithm with TAP approximation.

    :param numpy.ndarray y_t:
        (d,) dimensional vector with empirical rates of data.
    :param numpy.ndarray X_t:
        (r,c) dimensional array with spike trains for each trial and each cell.
    :param int R:
        Number of trials.
    :param numpy.ndarray theta0:
        Starting point for theta.
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix
    :param bool diag_weight_trick:
        if diagonal weight should be used. (Default=False)

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    """

    # Convergence limit for the rate
    GA_CONVERGENCE = 1e-4
    MAX_GA_ITERATIONS = 1000.
    # Get number of Neurons
    N = X_t.shape[1]
    # Initialize iteration counter
    iterations = 0
    # Solve backward problem without prior
    theta_max = backward_problem(y_t, N, 'TAP', diag_weight_trick=False)
    # Get Gradient (just prior beacuse the one of llk is zero)
    dlpo = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # Initialize the estimate of the inverse fisher info
    ddlpo_i_e = numpy.identity(theta_max.shape[0])
    # Change of theta
    d_th = dlpo
    # Search direction
    s = d_th
    # Get initial eta
    eta = numpy.copy(y_t)
    # Convergence criterion
    max_dlpo = numpy.amax(numpy.absolute(dlpo))/R
    # Until rate converged
    while max_dlpo > GA_CONVERGENCE and iterations<MAX_GA_ITERATIONS:
        # Compute direction for line search
        s_dir = numpy.dot(dlpo, ddlpo_i_e)
        # Set theta to old theta
        theta_prev = numpy.copy(theta_max)
        # Set current log posterior gradient to previous
        dlpo_prev = dlpo
        # Update theta by line search
        theta_max = mean_field_line_search(theta_max, N, R, s_dir, dlpo, sigma_o_i, eta)
        # Get the difference between old and new theta
        d_th = theta_max - theta_prev
        # Get eta by solving forward problem
        eta = forward_problem(theta_max, N, 'TAP')
        # Get gradient of posterior
        dlpo = R*(y_t - eta) - numpy.dot(sigma_o_i, theta_max - theta_o)
        # Difference in log posterior gradients
        dlpo_diff = dlpo_prev - dlpo
        # Project gradient change on theta change
        dlpo_diff_dth = numpy.inner(dlpo_diff, d_th)
        # Compute estimate of covariance matrix with Sherman-Morrison Formula
        a = (dlpo_diff_dth + \
             numpy.dot(dlpo_diff, numpy.dot(ddlpo_i_e, dlpo_diff.T)))*\
            numpy.outer(d_th, d_th)
        b = numpy.inner(d_th, dlpo_diff)**2
        c = numpy.dot(ddlpo_i_e, numpy.outer(dlpo_diff, d_th)) + \
            numpy.outer(d_th, numpy.inner(dlpo_diff, ddlpo_i_e))
        d = dlpo_diff_dth
        ddlpo_i_e += (a/b - c/d)
        # Count iterations
        iterations += 1
        # Get maximal Gradient entry
        max_dlpo = numpy.amax(numpy.absolute(dlpo))/R
        # If maximal number of iterations is reached break
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The Mean Field BFGS '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Calculate Fisher info
    G = compute_full_G(eta, theta_max, N)
    ddlpo = -R*G - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)
    # Return
    return theta_max, -ddlpo_i


def mean_field_line_search(theta, N, R, s, dlpo, sigma_o_i, eta):
    """ Performs the line search for TAP.

    :param numpy.ndarray theta:
        (d,) natural parameters
    :param int R:
        Number of trials
    :param numpy.ndarray s:
        (d,) search direction
    :param numpy.ndarray dlpo:
        (d,) derivative of posterior
    :param numpy.ndarray sigma_o_i:
        (d,d) inverse of one-step covariance
    :param numpy.ndarray eta:
        (d,) all expectation parameters eta

    :returns:
        (d,) numpy.ndarray with optimal theta
    """
    # Project posterior gradient on search direction
    dlpo_s = numpy.dot(dlpo.T, s)
    # Get Fisher info py TAP approximation
    # G = compute_fisher_info_from_eta_TAP(eta, theta, N)
    G = compute_full_G(eta, theta, N)
    # Get Hessian
    ddlpo = -R*G - sigma_o_i
    # Project Hessian on search direction
    ddlpo_s = numpy.dot(s, numpy.dot(ddlpo,s))
    # Compute how much the step should be along search direction
    alpha = -dlpo_s/ddlpo_s
    # Update theta
    theta_new = theta + alpha*s
    # Return
    return theta_new

functions = {'nr': mean_field_nr,
             'cg': mean_field_cg,
             'bf': mean_field_bfgs}