__author__ = 'Christian Donner'

import numpy
import max_posterior
from scipy import sparse
import transforms
import container

CONVERGED = 1e-3
MAX_GA_ITERATIONS = 5000
GA_CONVERGENCE = 1e-4
Fx_s = None
time_bin = -1

def run(spikes, order, window=1, map_function='nr', lmbda=200, max_iter=30):
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

    :returns:
        Results encapsulated in a container.EMData object, containing the
        smoothed posterior probability distributions of the natural parameters
        of the spike-train interactions at each timestep, conditional upon the
        given spikes.
    """
    # Ensure NaNs are caught
    numpy.seterr(invalid='raise')
    # Initialise the EM-data container
    map_func = functions[map_function]
    emd = container.EMData(spikes, order, window, map_func, lmbda)
    # Initialise the coordinate-transform maps
    compute_Fx_s(emd.spikes, emd.order)
    # Set up loop guards for the EM algorithm
    lmp = -numpy.inf
    lmc = pseudo_log_marginal(emd)
    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > CONVERGED):
        # Perform EM
        e_step(emd)
        m_step(emd)
        # Update previous and current log marginal values
        lmp = lmc
        lmc = pseudo_log_marginal(emd)
        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = numpy.absolute((lmp - lmc) / lmp)

    return emd

def e_step(emd):
    """
    Computes the posterior (approximated as a multivariate Gaussian
    distribution) of the natural parameters of observed spike patterns, given
    the state-transition hyperparameters. Firstly performs a `forward'
    iteration, in which the filter posterior density at time t is determined
    from the observed patterns at time t and the one-step prediction density at
    time t-1. Secondly performs a `backward' iteration, in which these
    sequential filter estimates are smoothed over time.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Compute the 'forward' filter density
    e_step_filter(emd)
    # Compute the 'backward' smooth density
    e_step_smooth(emd)


def e_step_filter(emd):
    """
    Computes the one-step-prediction density and the filter density in the
    expectation step.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Iterate forwards over each timestep, computing filter density
    emd.theta_f[0,:], emd.sigma_f[0,:] = run_map(emd, 0)
    for i in range(1, emd.T):
        # Compute one-step prediction density
        emd.theta_o[i,:] = numpy.dot(emd.F, emd.theta_f[i-1,:])
        tmp = numpy.dot(emd.F, emd.sigma_f[i-1,:,:])
        emd.sigma_o[i,:,:] = numpy.dot(tmp, emd.F.T) + emd.Q
        # Compute inverse of one-step prediction covariance
        emd.sigma_o_inv[i,:,:] = numpy.linalg.inv(emd.sigma_o[i,:,:])
        # Get MAP estimate of filter density
        emd.theta_f[i,:], emd.sigma_f[i,:] = run_map(emd, i)


def e_step_smooth(emd):
    """
    Computes smooth density in the expectation step.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Initialise the smoothed theta and sigma values
    emd.theta_s[-1,:] = emd.theta_f[-1,:]
    emd.sigma_s[-1,:,:] = emd.sigma_f[-1,:,:]
    # Iterate backwards over each timestep, computing smooth density
    for i in reversed(range(emd.T - 1)):
        # Compute the A matrix
        a = numpy.dot(emd.sigma_f[i,:,:], emd.F.T)
        A = numpy.dot(a, emd.sigma_o_inv[i+1,:,:])
        # Compute the backward-smoothed means
        tmp = numpy.dot(A, emd.theta_s[i+1,:] - emd.theta_o[i+1,:])
        emd.theta_s[i,:] = emd.theta_f[i,:] + tmp
        # Compute the backward-smoothed covariances
        tmp = numpy.dot(A, emd.sigma_s[i+1,:,:] - emd.sigma_o[i+1,:,:])
        tmp = numpy.dot(tmp, A.T)
        emd.sigma_s[i,:,:] = emd.sigma_f[i,:,:] + tmp
        # Compute the backward-smoothed lag-one covariances
        emd.sigma_s_lag[i+1,:,:] = numpy.dot(A, emd.sigma_s[i+1,:])


def m_step(emd):
    """
    Computes the optimised hyperparameters of the natural parameters of the
    posterior distributions over time. `Q' is the covariance matrix of the
    transition probability distribution. `F' is the autoregressive parameter of
    the state transitions, but it is kept constant in this implementation.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Update the initial mean of the one-step-prediction density
    emd.theta_o[0,:] = emd.theta_s[0,:]
    # Compute the state-transition hyperparameter
    m_step_Q(emd)


def m_step_Q(emd):
    """
    Computes the optimised state-transition covariance hyperparameters `Q' of
    the natural parameters of the posterior distributions over time.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    inv_lmbda = 0
    for i in range(1, emd.T):
        # Computing lag-one covariance locally
        #a = numpy.dot(emd.sigma_f[i-1,:,:], emd.F.T)
        #A = numpy.dot(a, emd.sigma_o_inv[i,:,:])
        #lag_one_covariance = numpy.dot(A, emd.sigma_s[i,:])
        # Loading saved lag-one smoother
        lag_one_covariance = emd.sigma_s_lag[i,:,:]
        tmp = emd.theta_s[i,:] - emd.theta_s[i-1,:]
        inv_lmbda += numpy.trace(emd.sigma_s[i,:,:]) -\
                 2 * numpy.trace(lag_one_covariance)  +\
                 numpy.trace(emd.sigma_s[i-1,:,:])  +\
                 numpy.dot(tmp, tmp)
    emd.Q = inv_lmbda / emd.D / (emd.T - 1) * numpy.identity(emd.D)



def run_map(emd, t):
    """
    Computes the MAP estimate of the natural parameters at some timestep, given
    the observed spike patterns at that timestep and the one-step-prediction
    mean and covariance for the same timestep. This function pass the variables
    at time t to the user-specified gradient ascent alogirhtm.
    """
    # Extract observed patterns and one-step predictions for time t
    # Data at time t
    global time_bin
    time_bin = t
    X_t = emd.spikes[t,:,:]
    #compute_Fx_s(X_t, emd.order)
    R = emd.R
    # Initial values of natural parameters
    theta_0 = emd.theta_o[t,:]
    # Mean and covariance of one-step prediction density
    theta_o = emd.theta_o[t,:]
    sigma_o = emd.sigma_o[t,:,:]
    sigma_o_i = emd.sigma_o_inv[t,:,:]
    # Run the user-specified gradient ascent algorithm
    theta_f, sigma_f = emd.max_posterior(X_t, R, theta_0, theta_o, sigma_o,
                                          sigma_o_i)

    return theta_f, sigma_f


def compute_Fx_s(X, O):
    """ 
    Constructs F(x_s=1, x_\s), feature vectors of interactions up to the 
    'O'th order from observed patterns for conditional likelihood model. 

    :param numpy.array X:
        Two dimensional (r, c) binary array, where the first dimension is runs
        (trials) and the second is the number of cells.
    :param int O:
        Order of interactions.

    :returns Fx_s:
        (r, D) sparse matrix, where D is the model dimension.
    """
    T, R, N = X.shape
    # Initialize Fx_s
    global Fx_s
    # List of lists (for each time bin) of sparse matrices (for each cell)
    Fx_s = []
    # For each time bin
    for i in range(T):
        # Initialize list
        Fx_s.append([])
        # Get spike data
        # For each cell
        for s in range(N):
            # Get spike data
            Xtmp = X[i,:,:].copy()
            # Set current cell to 1
            Xtmp[:,s] = 1
            # Compute Fx with cell active
            Fx1 = compute_Fx(Xtmp, O)
            # Get spike data again
            Xtmp = X[i,:,:].copy()
            # Sett current cell to 0
            Xtmp[:,s] = 0
            # Compute Fx for cell inactive
            Fx2 = compute_Fx(Xtmp, O)
            # Create sparse matrix of difference in active and inactive Fx
            Fx_s[i].append(sparse.coo_matrix(Fx1 - Fx2))


def compute_Fx(X, O):
    """
    Construct feature vectors of interactions up to the 'O'th order from 
    pattern data. 

    :param numpy.array X:
        (r, c) binary array, where the first dimension are runs (trials) 
        and second cells.
    :param int O:
        Order of interactions

    :returns Fx:
        (r, D) matrix of feature vectors, where D is the model 
        dimension.
    """
    # Get spike-matrix metadata
    R, N = X.shape
    # Compute each n-choose-k subset of cell IDs up to the 'O'th order
    subsets = transforms.enumerate_subsets(N, O)
    # Set up the output array
    Fx = numpy.zeros((len(subsets),R))
    # Iterate over each subset
    for i in range(len(subsets)):
        # Select the cells that are in the subset
        sp = X[:,subsets[i]]
        # Find the timesteps in which all subset-cells spike coincidentally
        spc = sp.sum(axis=1) == len(subsets[i])
        # Save the observed spike pattern
        Fx[i,:] = spc

    return Fx


def pseudo_newton(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    """ Newton-Raphson method with pseudo-log-likelihood as objective function.

    :param numpy.ndarray X:
        Two dimensional (r, c) binary array, where the first dimension is runs
        (trials) and the second is the number of cells.
    :param int R:
        Number of runs
    :param numpy.ndarray theta_0:
        Starting point for theta
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    @author: Christian Donner
    """
    # Read out number of cells and natural parameters
    N, D = X.shape[1], theta_0.shape[0]
    # Initialize theta, iteration counter and maximal derivative of posterior
    theta_max = theta_0
    iterations = 0
    max_dlpo = numpy.Inf
    # Intialize array for sum of active thetas (r,c)
    fs = numpy.empty([R, N])

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:

        # Initialize gradient and Hessian arrays
        dllk = numpy.zeros(D)
        ddllk = numpy.zeros([D,D])

        # Iterate over all cells
        for s_i in range(N):
            # Calculate sum of active thetas
            fs[:, s_i] = Fx_s[time_bin][s_i].T.dot(theta_max)
            # Calculate conditional rate
            etas = numpy.exp(fs[:, s_i])/(1 + numpy.exp(fs[:, s_i]))
            # Calculate derivative of conditional rate
            deta = -etas*(1-etas)
            # Calculate derivative for neuron
            dllk += Fx_s[time_bin][s_i].dot(X[:, s_i] - etas)
            # Fill in detas in Fx_s
            Fx_s_deta = sparse.coo_matrix(((deta)[Fx_s[time_bin][s_i].col],
                                  [Fx_s[time_bin][s_i].col,
                                   Fx_s[time_bin][s_i].row]),
                                  [Fx_s[time_bin][s_i].shape[1],
                                   Fx_s[time_bin][s_i].shape[0]])
            # Compute finally Hesian for Neuron
            ddllk += Fx_s[time_bin][s_i].dot(Fx_s_deta)
        # Calculate prior
        dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
        # Calculate posterior
        dlpo = numpy.array(dllk + dlpr)
        # Calculate the Hessian of posterior
        ddlpo = numpy.array(ddllk - sigma_o_i)
        # Compute the inverse
        ddlpo_inv = numpy.linalg.inv(ddlpo)
        # Update theta
        theta_max = theta_max - 0.1*numpy.dot(ddlpo_inv, dlpo)
        # Get maximal entry in gradient and count iteration
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        iterations += 1
        # Throw Exception if did not converge
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior pseudo newton '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute sum of active thetas
    for s_i in range(N):
        fs[:, s_i] = Fx_s[time_bin][s_i].T.dot(theta_max)

    # Compute final Hessian of posterior
    dllk, etas = pseudo_dllk(theta_max, X, fs)
    ddllk = pseudo_ddllk(etas, D)
    ddlpo = ddllk - sigma_o_i
    # Compute inverse
    ddlpo_i = numpy.linalg.inv(ddlpo)
    # Return fitted theta and Fisher Info matrix
    return theta_max, -ddlpo_i


def pseudo_cg(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    """ Fits due to non linear conjugate gradient, where Pseudolikelihood is the
     objective function.

    :param numpy.ndarray X:
        Two dimensional (r, c) binary array, where the first dimension is runs
        (trials) and the second is the number of cells.
    :param int R:
        Number of runs
    :param numpy.ndarray theta_0:
        Starting point for theta
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    @author: Christian Donner
    """

    # Extract parameters
    N = X.shape[1]
    D = theta_0.shape[0]
    # Initialize theta
    theta_max = theta_0
    # Calculate fs = sum(theta_I*F_I(x_s = 1, x_/s))
    fs = numpy.empty([R, N])
    for s_i in range(N):
        fs[:, s_i] = Fx_s[time_bin][s_i].T.dot(theta_max)

    # Initialize stopping criterion variables
    max_dlpo = numpy.Inf
    iterations = 0
    # Get likelihood gradient
    dllk, etas = pseudo_dllk(theta_max, X, fs)
    # Get prior
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # Get posterior
    dlpo = dllk + dlpr
    # Initialize theta gradient
    d_th = dlpo
    # Set initial search direction
    s = dlpo
    # Perform first line search
    theta_max, fs = pseudo_line_search(theta_max, X, s, fs, dlpo, sigma_o_i,
                                       etas)
    # Calculate new likelihood gradient
    dllk, etas = pseudo_dllk(theta_max, X, fs)
    # and new prior
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # and new Posterior
    dlpo = dllk + dlpr

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Set old theta direction
        d_th_prev = d_th
        # Set posterior to new theta direction
        d_th = dlpo
        # Calculate beta
        beta = max_posterior.compute_beta(d_th, d_th_prev)
        # Set new search direction
        s = d_th + beta * s
        # Perform line search in this direction
        theta_max, fs = pseudo_line_search(theta_max, X, s, fs, dlpo, sigma_o_i,
                                           etas)
        # Calculate the new gradient and conditional rates
        dllk, etas = pseudo_dllk(theta_max, X, fs)
        # Calculate prior
        dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
        # Calculate posterior
        dlpo = dllk + dlpr
        # Get maximal entry of posterior gradient an count iterations
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        iterations += 1
        # Throw exceptio if not converged
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The pseudo conjugate gradient '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final Hessian of posterior
    dllk, etas = pseudo_dllk(theta_max, X, fs)
    ddllk = pseudo_ddllk(etas, D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ddlpo_i = numpy.linalg.inv(ddlpo)
    # Return fitted theta and Fisher Info matrix
    return theta_max, -ddlpo_i


def pseudo_bfgs(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    """ Fits due to Broyden-Fletcher-Goldfarb-Shanno algorithm, where
    Pseudolikelihood is the objective function.

    :param numpy.ndarray X:
        Two dimensional (r, c) binary array, where the first dimension is runs
        (trials) and the second is the number of cells.
    :param int R:
        Number of runs
    :param numpy.ndarray theta_0:
        Starting point for theta
    :param numpy.ndarray theta_o:
        One-step prediction for theta
    :param sigma_o:
        One-step prediction covariance matrix
    :param sigma_o_i:
        Inverse one-step prediction covariance matrix

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.

    @author: Christian Donner
    """

    # Get number of cells and natural parameters
    N, D = X.shape[1], theta_0.shape[0]
    # Initialize theta with previous smoothed theta
    theta_max = theta_0
    # Calculate fs = sum(theta_I*F_I(x_s = 1, x_/s))
    fs = numpy.empty([R, N])
    for s_i in range(N):
        fs[:, s_i] = Fx_s[time_bin][s_i].T.dot(theta_max)

    # Initialize the estimate of the inverse fisher info
    ddlpo_i_e = numpy.identity(theta_max.shape[0])
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0
    # Compute derivative of posterior
    dllk, etas = pseudo_dllk(theta_max, X, fs)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:

        # Compute direction for line search
        s_dir = numpy.dot(dlpo, ddlpo_i_e)
        # Set theta to old theta
        theta_prev = numpy.copy(theta_max)
        # Set current log posterior gradient to previous
        dlpo_prev = dlpo
        # Perform line search
        theta_max, fs = pseudo_line_search(theta_max, X, s_dir, fs, dlpo,
                                           sigma_o_i, etas)
        # Get the difference between old and new theta
        d_theta = theta_max - theta_prev
        # Compute derivative of posterior
        dllk, etas = pseudo_dllk(theta_max, X, fs)
        dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
        dlpo = dllk + dlpr
        # Difference in log posterior gradients
        dlpo_diff = dlpo_prev - dlpo
        # Project gradient change on theta change
        dlpo_diff_dth = numpy.inner(dlpo_diff, d_theta)
        # Compute estimate of covariance matrix with Sherman-Morrison Formula
        a = (dlpo_diff_dth + \
             numpy.dot(dlpo_diff, numpy.dot(ddlpo_i_e, dlpo_diff.T)))*\
            numpy.outer(d_theta, d_theta)
        b = numpy.inner(d_theta, dlpo_diff)**2
        c = numpy.dot(ddlpo_i_e, numpy.outer(dlpo_diff, d_theta)) + \
            numpy.outer(d_theta, numpy.inner(dlpo_diff, ddlpo_i_e))
        d = dlpo_diff_dth
        ddlpo_i_e += (a/b - c/d)
        # Get maximal entry of log posterior grad divided by number of trials
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The pseudo bfgs '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final Hessian of posterior
    ddllk = pseudo_ddllk(etas, D)
    ddlpo = ddllk - sigma_o_i
    # Compute inverse
    ddlpo_i = numpy.linalg.inv(ddlpo)
    # Return fitted theta and Fisher Info matrix
    return theta_max, -ddlpo_i


def pseudo_line_search(theta, X, s, fs, dlpo, sigma_o_i, etas):
    """ Performs the line search for pseudo-log-likelihood as objective
    function by quadratic approximation at current theta.

    :param numpy.ndarray theta:
        (d,) natural parameters
    :param numpy.ndarray X:
        (r,c) spike data
    :param numpy.ndarray s:
        (d,) search direction
    :param numpy.ndarray fs:
        (r,c) sum of active thetas for run and cell
    :param numpy.ndarray dlpo:
        (d,) derivative of posterior
    :param numpy.ndarray:
        (d,d) inverse of one-step covariance
    :param numpy.ndarray etas:
        (r,c) conditional rate for each run and cell

    :returns:
        (d,) new theta according to quadratic approximation
        (r, c) new sums of active thetas
    """
    # Extract number of runs and cells
    R, N = X.shape
    # Initialize array for Fx_s projection on search direction (r,c)
    Fx_s_s = numpy.empty([R, N])
    # Iterate of all cells and project Fx_s on search direction
    for s_i in range(N):
        Fx_s_s[:, s_i] = Fx_s[time_bin][s_i].T.dot(s)
    # Project posterior on search direction
    dlpo_s = numpy.dot(dlpo.T, s)
    # Project conditional rate on search direction
    detas = etas*(1-etas)
    # Project one-step covariance matrix on search direction
    sigma_o_i_s = numpy.dot(s, numpy.dot(sigma_o_i, s))
    # Compute projection of pseudo-log-likelihood Hessian on search direction
    ddlpo_s = numpy.tensordot(detas*Fx_s_s, Fx_s_s, ((1,0),(1,0))) + sigma_o_i_s
    # Compute how much the step should be along search direction
    alpha = dlpo_s/ddlpo_s
    # Update sum of active thetas
    fs_new = fs + alpha*Fx_s_s
    # Update theta
    theta_new = theta + alpha*s
    # Return
    return theta_new, fs_new


def pseudo_dllk(theta, X, fs):
    """ Calculculates the gradient of the pseudo-log-likelihood.

    :param numpy.ndarray theta:
        (d,) array of natural parameters
    :param numpy.ndarray X:
        (r,c) array with spike data
    :param numpy.ndarray fs:
        (r,c) array containing sum of 'active thetas' for data

    :returns:
        (d,) numpy.ndarray with gradient
        (r,c) numpy.ndarray with conditional rates
    """
    # Get number of cells
    N = X.shape[1]
    # Initialize gradient array
    dllk = numpy.zeros(theta.shape[0])
    # Calculate conditional rate
    etas = numpy.exp(fs)/(1 + numpy.exp(fs))
    # Iterate over all cells
    for s_i in range(N):
        # Add gradient for each cell
        dllk += Fx_s[time_bin][s_i].dot((X[:,s_i] - etas[:,s_i]))
    # Return
    return dllk, etas


def pseudo_ddllk(etas, D):
    """ Calculates the Hessian for the pseudo-log-likelihood.

    :param numpy.ndarray etas:
        (r,c) array of conditional rate
    :param int D:
        number of natural parameters

    :returns
        (d,d) array with Hessian of pseudo-log-likelihood
    """
    # Get number of cells
    N = etas.shape[1]
    # Intitialize Hessian
    ddllk = numpy.zeros([D,D])
    # iteratate over all cells
    for s_i in range(N):
        # Calculate the derivative of conditional rate wrt. theta
        deta = -etas[:, s_i]*(1-etas[:, s_i])
        # Fill the derivatives where Fx_s one
        Fx_s_deta = sparse.coo_matrix(((deta)[Fx_s[time_bin][s_i].col],
                                [Fx_s[time_bin][s_i].col,
                                 Fx_s[time_bin][s_i].row]),
                                [Fx_s[time_bin][s_i].shape[1],
                                 Fx_s[time_bin][s_i].shape[0]])
        # Compute final Hessian for each cell
        ddllk += Fx_s[time_bin][s_i].dot(Fx_s_deta)
    # Return
    return ddllk


def generate_spikes_gibbs(theta, N, O, R, **kwargs):
    """Generates spike trains for the model given the thetas with
    `Gibbs-Sampling <https://en.wikipedia.org/wiki/Gibbs_sampling>`_.

    :param numpy.ndarray theta:
        parameters used for sampling for each time bin
    :param int N:
        Number of units
    :param int O:
        Order of interaction
    :param int R:
        Number of trials that are generated.

    :returns:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c, as a
        numpy.ndarray
    """
    seed = kwargs.get('seed', 42)
    pre_R = kwargs.get('pre_n', 100)
    numpy.random.seed(seed)
    T = theta.shape[0]
    X = numpy.zeros([T, R+pre_R, N], dtype=numpy.uint8)
    subsets = transforms.enumerate_subsets(N, O)
    D = len(subsets)
    subset_map = numpy.zeros([D, N])

    for i in range(len(subsets)):
        subset_map[i, subsets[i]] = 1

    subset_count = numpy.sum(subset_map, axis=1)
    # draw random number from uniform distribution
    rand_numbers = numpy.random.rand(T, R+pre_R, N)

    for t in range(T):
        # iterate through all trials
        cur_theta = theta[t]
        for l in range(1, R+pre_R):
            # iterate through all neurons
            for i in range(N):
                # construct pattern from trial before and from neurons that have been seen in this trial
                pattern = numpy.array([numpy.hstack([X[t, l, :i], X[t, l-1, i:]])])
                # set x^(i,t) to "1" and compute f(X) for those
                pattern[:, i] = 1
                fx1 = (numpy.dot(pattern, subset_map.T) == subset_count)[0]
                # set x^(i,t) to "0" and compute f(X) for those
                pattern[:, i] = 0
                fx0 = (numpy.dot(pattern, subset_map.T) == subset_count)[0]
                # compute p( x^(i,l) = 1 || X^(1:i-1,t),X^(i+1:N,l-1) )
                prob_spike = 0.5*(1 + numpy.tanh(0.5*(numpy.sum(cur_theta[fx1]) - numpy.sum(cur_theta[fx0]))))
                # if smaller than probability X^(i,l) -> 1
                X[t, l, i] = numpy.greater_equal(prob_spike, rand_numbers[t, l, i])

    return X[:, pre_R:, :]

def pseudo_log_marginal(emd, period=None):
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
    log_p = pseudo_log_marginal_raw(emd.theta_f, emd.theta_o, emd.sigma_f, emd.sigma_o_inv,
        emd.spikes, emd.R, period)

    return log_p

def pseudo_log_marginal_raw(theta_f, theta_o, sigma_f, sigma_o_inv, X, R, period=None):
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
        a += pseudo_log_likelihood(X[i,:,:], theta_f[i,:], i)
        theta_d = theta_f[i,:] - theta_o[i,:]
        b -= numpy.dot(theta_d, numpy.dot(sigma_o_inv[i,:,:], theta_d))
        b += numpy.log(numpy.linalg.det(sigma_f[i,:,:])) +\
             numpy.log(numpy.linalg.det(sigma_o_inv[i,:,:]))
    log_p = a + b / 2

    return log_p


def pseudo_log_likelihood(X_t, theta, t):
    """ Computes the pseudo-log-likelihood for data and theta

    :param numpy.ndarray X_t:
        (r,c) array containing spike data
    :param numpy.ndarray theta:
        (d) array containing natural parameters
    :param int t:
        time bin of data and theta

    :returns float:
        pseudo-log-likelihood
    """
    # Extraxt trial and Cell number
    R, N = X_t.shape
    # Initialize pseudo-log-likelihood
    pseudo_llk = 0
    # Run over all cells
    for s_i in range(N):
        # Calculate fs
        fs = Fx_s[t][s_i].T.dot(theta)
        # and pseudo-log-likelihood for each cell
        pseudo_llk += numpy.sum(X_t[:,s_i]*fs - numpy.log(1 + numpy.exp(fs)))
    # Return
    return pseudo_llk


functions = {'nr': pseudo_newton,
             'cg': pseudo_cg,
             'bf': pseudo_bfgs}

if __name__ == '__main__':
    N, O, T, R = 10, 2, 1, 1000
    import synthesis
    import pylab
    import time
    D = transforms.compute_D(N, O)
    thetas = synthesis.generate_thetas(N, O, T)
    time_bin = 0
    t1 = time.clock()
    spikes = generate_spikes_gibbs(thetas, N, O, R)
    print('sampling done in %f s' %(time.clock() - t1))
    theta_0 = numpy.zeros(D)
    compute_Fx_s(spikes, O)
    t1 = time.clock()
    theta_o = numpy.ones(theta_0.shape[0])*1.
    sigma_o_i = numpy.diag(numpy.ones(D))*0.
    theta_max, sigma = pseudo_cg(spikes[time_bin], R, theta_0, theta_o, 0, sigma_o_i)
    print('fitting done in %f s' %(time.clock() - t1))
    #pylab.plot(thetas[0],theta_max_h, 'bo')
    pylab.plot(thetas[0],theta_max, 'k.')
    pylab.show()