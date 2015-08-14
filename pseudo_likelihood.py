__author__ = 'Christian Donner'

import numpy
import max_posterior
from scipy import sparse
import transforms
import mean_field
import bethe_approximation


MAX_GA_ITERATIONS = 5000
Fx_s = None
time_bin = -1


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


def pseudo_newton(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i):
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
    N, D = X_t.shape[1], theta_0.shape[0]
    # Initialize theta, iteration counter and maximal derivative of posterior
    theta_max = theta_0
    iterations = 0
    max_dlpo = numpy.Inf
    # Intialize array for sum of active thetas (r,c)
    fs = numpy.empty([R, N])

    # Iterate until convergence or failure
    while max_dlpo > max_posterior.GA_CONVERGENCE:

        # Initialize gradient and Hessian arrays
        dllk = numpy.zeros(D)
        ddllk = numpy.zeros([D,D])

        # Iterate over all cells
        for s_i in range(N):
            # Calculate sum of active thetas
            fs[:, s_i] = Fx_s[time_bin][s_i].T.dot(theta_max)
            # Calculate conditional rate
            #tmp = numpy.exp(fs[:, s_i])
            #etas = tmp / (1 + tmp)
            try:
                calc = numpy.less_equal(fs[:,s_i], 709)
            except FloatingPointError:
                print numpy.amax(fs)
            etas = numpy.ones(fs.shape[0])
            etas[calc] = numpy.exp(fs[calc,s_i])/(1.+numpy.exp(fs[calc,s_i]))
            # Calculate derivative of conditional rate
            deta = - etas * (1-etas)
            # Calculate derivative for neuron
            dllk += Fx_s[time_bin][s_i].dot(X_t[:, s_i] - etas)
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

    # Return fitted theta and Fisher Info matrix
    try:
        eta = bethe_approximation.compute_eta_BP(theta_max, N)[0]
    except:
        eta = bethe_approximation.compute_eta_CCCP(theta_max, N)[0]
    ddllk = -R*mean_field.compute_full_G(eta, theta_max, N)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ddlpo_i = numpy.linalg.inv(ddlpo + 1e-13*numpy.identity(ddlpo.shape[0]))
    return theta_max, -ddlpo_i


def pseudo_cg(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i, param_est_eta='bethe_BP'):
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
    R, N = X_t.shape
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
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    # Get prior
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # Get posterior
    dlpo = dllk + dlpr
    # Initialize theta gradient
    d_th = dlpo
    # Set initial search direction
    s = dlpo
    # Perform first line search
    theta_max, fs = pseudo_line_search2(theta_max, X_t, s, fs, dlpo, sigma_o_i,
                                       etas, theta_o)
    # Calculate new likelihood gradient
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    # and new prior
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    # and new Posterior
    dlpo = dllk + dlpr

    # Iterate until convergence or failure
    while max_dlpo > max_posterior.GA_CONVERGENCE:
        # Set old theta direction
        d_th_prev = d_th
        # Set posterior to new theta direction
        d_th = dlpo
        # Calculate beta
        beta = max_posterior.compute_beta(d_th, d_th_prev, 'HS')
        # Set new search direction
        s = d_th + beta * s
        # Perform line search in this direction
        theta_max, fs = pseudo_line_search2(theta_max, X_t, s, fs, dlpo, sigma_o_i,
                                           etas, theta_o)
        # Calculate the new gradient and conditional rates
        dllk, etas = pseudo_dllk(theta_max, X_t, fs)

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
    #eta = mean_field.forward_problem_hessian(theta_max, N, 'TAP')
    try:
        eta = bethe_approximation.compute_eta_BP(theta_max, N)[0]
    except:
        eta = bethe_approximation.compute_eta_CCCP(theta_max, N)[0]
    ddllk = -R*mean_field.compute_full_G(eta, theta_max, N)
    #ddllk = pseudo_ddllk(etas,D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ddlpo_i = numpy.linalg.inv(ddlpo + 1e-13*numpy.identity(ddlpo.shape[0]))
    # Return fitted theta and Fisher Info matrix
    return theta_max, -ddlpo_i


def pseudo_bfgs(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i):
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
    N, D = X_t.shape[1], theta_0.shape[0]
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
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Iterate until convergence or failure
    while max_dlpo > max_posterior.GA_CONVERGENCE:

        # Compute direction for line search
        s_dir = numpy.dot(dlpo, ddlpo_i_e)
        # Set theta to old theta
        theta_prev = numpy.copy(theta_max)
        # Set current log posterior gradient to previous
        dlpo_prev = dlpo
        # Perform line search
        theta_max, fs = pseudo_line_search2(theta_max, X_t, s_dir, fs, dlpo,
                                           sigma_o_i, etas)
        # Get the difference between old and new theta
        d_theta = theta_max - theta_prev
        # Compute derivative of posterior
        dllk, etas = pseudo_dllk(theta_max, X_t, fs)
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

    # Return fitted theta and Fisher Info matrix
    # eta = mean_field.forward_problem_hessian(theta_max, N, 'TAP')
    try:
        eta = bethe_approximation.compute_eta_BP(theta_max, N)[0]
    except:
        eta = bethe_approximation.compute_eta_CCCP(theta_max, N)[0]
    ddllk = -R*mean_field.compute_full_G(eta, theta_max, N)
    #ddllk = pseudo_ddllk(etas,D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ddlpo_i = numpy.linalg.inv(ddlpo + 1e-13*numpy.identity(ddlpo.shape[0]))
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


def pseudo_line_search2(theta, X, s, fs, dlpo, sigma_o_i, etas, theta_o):
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
    num_iter = 0
    conv = numpy.inf
    while conv > 1e-2 and num_iter < 10:
        dlpo_s_old = numpy.absolute(dlpo_s)
        # Project conditional rate on search direction
        detas = etas*(1-etas)
        # Project one-step covariance matrix on search direction
        sigma_o_i_s = numpy.dot(s, numpy.dot(sigma_o_i, s))
        # Compute projection of pseudo-log-likelihood Hessian on search direction
        ddlpo_s = numpy.tensordot(detas*Fx_s_s, Fx_s_s, ((1,0),(1,0))) + sigma_o_i_s
        # Compute how much the step should be along search direction
        alpha = dlpo_s/ddlpo_s
        # Update sum of active thetas
        fs_new = fs + .5*alpha*Fx_s_s
        # Update theta
        theta_new = theta + .5*alpha*s
        dllk, etas = pseudo_dllk(theta_new, X, fs)
        # Calculate prior
        dlpr = -numpy.dot(sigma_o_i, theta_new - theta_o)
        dlpo = dllk + dlpr
        dlpo_s = numpy.dot(dlpo.T, s)
        conv = numpy.absolute(dlpo_s_old-dlpo_s)
        num_iter += 1
    # Return
    return theta_new, fs_new


def compute_cond_eta(theta, t):
    """ Computes conitional rate

    :param numpy.ndarray theta:
        (d) array with thetas at time t
    :param int t:
        time index of theta

    :returns:
        (N,) array whit conditional rates for each neuron
    """
    N = len(Fx_s[t])
    R = Fx_s[t][0].shape[1]
    fs = numpy.empty([R, N])
    for s_i in range(N):
        fs[:, s_i] = Fx_s[t][s_i].T.dot(theta)
    try:
        calc = numpy.less_equal(fs, 709)
    except FloatingPointError:
        print numpy.amax(fs)
    etas = numpy.ones(fs.shape)
    etas[calc] = numpy.exp(fs[calc])/(1.+numpy.exp(fs[calc]))
    return numpy.mean(etas, axis=0)


def pseudo_dllk(theta, X, fs):
    """ Calculates the gradient of the pseudo-log-likelihood.

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
    calc = numpy.less_equal(fs, 709)
    etas = numpy.ones(fs.shape)
    etas[calc] = numpy.exp(fs[calc])/(1.+numpy.exp(fs[calc]))
    #etas[skip_calc] = 1.
    #etas = tmp / (1 + tmp)
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
    log_p = pseudo_log_marginal_raw(emd.theta_f, emd.theta_o, emd.sigma_f,
                                    emd.sigma_o_inv, emd.spikes, emd.R, period)

    return log_p

def pseudo_log_marginal_raw(theta_f, theta_o, sigma_f, sigma_o_inv, X, R,
                            period=None):
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
        A = numpy.dot(sigma_f[i,:,:], sigma_o_inv[i,:,:])
        L = numpy.linalg.cholesky(A)
        b += 2*numpy.sum(numpy.log(numpy.diag(L)))
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
    spikes = synthesis.generate_spikes_gibbs(thetas, N, O, R)
    print('sampling done in %f s' %(time.clock() - t1))
    theta_0 = numpy.zeros(D)
    compute_Fx_s(spikes, O)
    t1 = time.clock()
    theta_o = numpy.ones(theta_0.shape[0])*1.
    sigma_o_i = numpy.diag(numpy.ones(D))*0.
    theta_max, sigma = pseudo_cg(spikes[time_bin], R, theta_0, theta_o, 0,
                                 sigma_o_i)
    print('fitting done in %f s' %(time.clock() - t1))
    #pylab.plot(thetas[0],theta_max_h, 'bo')
    pylab.plot(thetas[0],theta_max, 'k.')
    pylab.show()