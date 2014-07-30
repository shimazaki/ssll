__author__ = 'Christian Donner'

import numpy
from scipy import sparse
import transforms


Fx_s = None

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
    R, N = X.shape
    D = 0
    for i in range(O):
        D += transforms.comb(N, i+1)

    global Fx_s
    Fx_s = []

    for s in range(N):
        Xtmp = X.copy()
        Xtmp[:,s] = 1
        Fx1 = compute_Fx(Xtmp, O)

        Xtmp = X.copy()
        Xtmp[:,s] = 0
        Fx2 = compute_Fx(Xtmp, O)

        Fx_s.append(sparse.coo_matrix(Fx1 - Fx2))

    return Fx_s


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


def get_set_pattern(X):
    """ Computes the set of patterns in data

    :param X numpy.ndarray:
        two dimensional (r, c) binary array, where first dimension are runs
        (trials) and second cells

    :return tuple:
        first numpy.ndarray (s, c) where s is the number of different patterns
        that occur in the data and second dimension is cells
        second numpy.nd.array (s) with the number of how often a pattern occured
    """

    # Get number of runs and cells
    R, N = X.shape
    # Initialise pattern set
    X_unique = numpy.empty([0,N], dtype=numpy.uint8)
    # Initialise array where indices of patterns are stored that already occured
    non_unique = numpy.array([])
    # Initialise array where occurances of patterns are stored
    occurances = numpy.array([])
    # Iterate over all patterns in data
    for l in range(0, R):
        # If pattern not occured before
        if l not in non_unique:
            # Extract current pattern
            tmp_pattern = X[l, :]
            # Find where patterns occured in data
            pattern_non_unique = numpy.nonzero(numpy.prod(tmp_pattern == X,
                                                          axis=1))[0]
            # Store unique pattern
            X_unique = numpy.vstack([X_unique, tmp_pattern])
            # Safe number of occurances
            occurances = numpy.hstack([occurances, pattern_non_unique.shape[0]])
            # Save indices of occured pattern so these are not searched anymore
            non_unique = numpy.hstack([non_unique, pattern_non_unique])
    # return set of patterns and occurances
    return X_unique, numpy.array(occurances)

def pseudo_newton(gradient_fun, D, N, pool):
    """ Computes gradient and fisher info for pseudo likelihood

    :param numpy.ndarray theta_max:
        one dimensional array with current theta
    :param numpy.ndarray X_unique:
        two dimensional (s, c) array with all individual pattern that occured in
        the data
    :param numpy.ndarray dfs:
        three dimensional array (d, s, c) with the derivative of fs. First
        dimension thetas,
        second individual pattern, third cells.
    :param numpy.ndarray dfs_occurance:
        same as dfs second dimension is multiplied with the number of its
        occurances in the data

    :return tuple:
        first one dimensional array with the gradient of pseudolikelihood at
        theta_max
        second two dimensional array with the fisher info of pseudolikelihood at
        theta_max
    """

    results = map(gradient_fun, range(N))
    dllk = numpy.zeros(D)
    ddllk = numpy.zeros([D,D])

    for dllks, ddllks in results:
        dllk += dllks
        ddllk += ddllks

    ddllk = numpy.array(ddllk)
    return dllk, ddllk

def compute_gradient(s, theta_max, X_unique, occurances):

    fs = Fx_s[s].T.dot(theta_max)
    etas = numpy.exp(fs)/(1 + numpy.exp(fs))
    deta = -etas*(1-etas)
    dllks = Fx_s[s].dot((X_unique[:, s] - etas)*occurances)
    deta_occ = sparse.coo_matrix(((deta*occurances)[Fx_s[s].col],
                                  [Fx_s[s].col, Fx_s[s].row]),
                                 [Fx_s[s].shape[1],Fx_s[s].shape[0]])
    ddllks = Fx_s[s].dot(deta_occ)
    return dllks, ddllks

def compute_gradient_diag(s, theta_max, X_unique, occurances):

    fs = Fx_s[s].T.dot(theta_max)
    etas = numpy.exp(fs)/(1 + numpy.exp(fs))
    deta = -etas*(1-etas)
    dllks = Fx_s[s].dot((X_unique[:,s] - etas)*occurances)
    ddllks = numpy.diag(Fx_s[s].dot(deta*occurances))
    return dllks, ddllks

def pseudo_likelihood_christian(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    N, D = X.shape[1], theta_0.shape[0]
    theta_max = theta_0
    iterations = 0
    max_dll = numpy.Inf
    X_unique, occurances = get_set_pattern(X)
    gradient_fun = lambda s: compute_gradient(s, theta_max=theta_max,
                                              X_unique=X_unique,
                                              occurances=occurances)
    #pool = Pool(processes=3)
    pool = None
    while max_dll > 1e-5:

        dll, ddll = pseudo_newton(gradient_fun, D, N, pool)
        #pseudo_gradient_diag(gradient_fun_tmp, D, N, pool)
        # Gradient method
        ddll_inv = numpy.linalg.inv(ddll + 1e-5*numpy.identity(D))
        theta_max = theta_max - 0.1*numpy.dot(ddll_inv, dll)
        max_dll = numpy.amax(numpy.absolute(dll)) / R
        iterations += 1
        if iterations == 1e3:
            max_dll = 0
    #pool.close()
    return theta_max, iterations

def pseudo_likelihood_christian_diag(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    N, D = X.shape[1], theta_0.shape[0]
    theta_max = theta_0
    iterations = 0
    max_dll = numpy.Inf
    X_unique, occurances = get_set_pattern(X)
    gradient_fun = lambda s: compute_gradient_diag(s, theta_max=theta_max,
                                              X_unique=X_unique,
                                              occurances=occurances)
    #pool = Pool(processes=3)
    pool = None
    while max_dll > 1e-5:

        dll, ddll = pseudo_newton(gradient_fun, D, N, pool)
        #pseudo_gradient_diag(gradient_fun_tmp, D, N, pool)
        # Gradient method
        ddll_inv = numpy.linalg.inv(ddll + 1e-5*numpy.identity(D))
        theta_max = theta_max - 0.1*numpy.dot(ddll_inv, dll)
        max_dll = numpy.amax(numpy.absolute(dll)) / R
        iterations += 1
        if iterations == 1e3:
            max_dll = 0
    #pool.close()
    return theta_max, iterations

def pseudo_likelihood_hideaki(X, R, theta_0, theta_o, sigma_o, sigma_o_i):
    N, D = X.shape[1], theta_0.shape[0]
    theta_max = theta_0
    iterations = 0
    max_dll = numpy.Inf
    #pool = Pool(processes=3)
    pool = None
    while max_dll > 1e-5:

        dll, ddll = dll_pseudolikelihood(X, theta_max)
        #pseudo_gradient_diag(gradient_fun_tmp, D, N, pool)
        # Gradient method
        ddll_inv = numpy.linalg.inv(ddll)
        theta_max = theta_max - 0.1*numpy.dot(ddll_inv, dll)
        max_dll = numpy.amax(numpy.absolute(dll)) / R
        iterations += 1
        if iterations == 1e3:
            max_dll = 0
    #pool.close()
    return theta_max, iterations


def dll_pseudolikelihood(X, theta):
    """
    Computes the derivative of the pesudo log-lieklihood.

    :param X:
        (R,N) matrix of binary spike data.
    :param theta:
        The natural parameter at which the derivative is computed.

    :returns:
        The derivative of the pesudo log-lieklihood.
    """
    R, N = numpy.shape(X)

    # construct a matrix of pairwise interactions
    J = numpy.zeros((N,N))
    triu_indices = numpy.triu_indices(N, 1)
    J[triu_indices] = theta[N:]

    # conditional probabilities of a spike, (R,N) matrix
    fs = theta[:N] + numpy.dot(X, J) + numpy.dot(X, J.T)
    Eta = numpy.exp(fs) / ( 1 + numpy.exp(fs) )

    # Gradient of the first order natrual parameters
    dEta = X - Eta
    dtheta_1 = numpy.sum(dEta, 0)
    #print dEta

    # Gradient of the second order natrual parameters
    A = numpy.dot( dEta.T, X )
    A = A + A.T
    dtheta_2 = A[triu_indices]

    # Gradient array
    dll = numpy.hstack((dtheta_1,dtheta_2))

    # Hessian of the first order
    ddEta = Eta * (1 - Eta); # (R,N)
    hessian_1 = - numpy.sum ( ddEta, 0 ) # (N)

    # Hessian of the second order
    B = numpy.dot( ddEta.T, X ) # (R,N).T x (R,N) = (N,N)
    B = B + B.T
    hessian_2 = - B[triu_indices]

    # Hessian matrix
    hessian = numpy.hstack((hessian_1,hessian_2))
    ddll = numpy.diag(hessian)

    return dll, ddll


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

    for t in range(T):
        # iterate through all trials
        for l in range(1, R+pre_R):
            # iterate through all neurons
            for i in range(N):
                # construct pattern from trial before and from neurons that have been seen in this trial
                pattern = numpy.array([numpy.hstack([X[t, l, :i], X[t, l-1, i:]])])
                # set x^(i,t) to "1" and compute f(X) for those
                pattern[:, i] = 1
                fx1 = compute_Fx(pattern, O).T
                # set x^(i,t) to "0" and compute f(X) for those
                pattern[:, i] = 0
                fx0 = compute_Fx(pattern, O).T
                # compute p( x^(i,l) = 1 || X^(1:i-1,t),X^(i+1:N,l-1) )
                prob_spike = 0.5*(1 + numpy.tanh(0.5*(fx1.dot(theta[t].T) - fx0.dot(theta[t].T))))
                # draw random number from uniform distribution
                random_numbers = numpy.random.rand(1)
                # if smaller than probability X^(i,l) -> 1
                X[t, l, i] = numpy.greater_equal(prob_spike, random_numbers)[0]

        return X[:, pre_R:, :]


if __name__=='__main__':
    N, O, T, R = 10, 2, 1, 100
    import synthesis
    import pylab
    D = transforms.compute_D(N, O)
    thetas = synthesis.generate_thetas(N, O, T)
    spikes = generate_spikes_gibbs(thetas, N, O, R)
    theta_0 = numpy.zeros(D)
    set_spikes = get_set_pattern(spikes[0])[0]
    compute_Fx_s(set_spikes, O)
    theta_max_h, iterations_h = pseudo_likelihood_hideaki(spikes[0], R, theta_0, 0, 0, 0)
    theta_max_c, iterations_c = pseudo_likelihood_christian(spikes[0], R, theta_0, 0, 0, 0)
    print(thetas)
    print(theta_max_h, iterations_h)
    print(theta_max_c, iterations_c)
    pylab.plot(thetas[0],theta_max_h, 'bo')
    pylab.plot(thetas[0],theta_max_c, 'r.')
    pylab.show()
