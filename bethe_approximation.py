__author__ = 'christian'

import numpy


def compute_eta_CCCP(theta, N):
    """ CCCP Algorithm to find solution for Bethe free energy [Yuille, 2002 Neural Comp.]

    :param numpy.ndarray theta:
        (d,) dimensional array with natural parameters in it
    :param int N:
        Number of cells
    :returns:
        (d,) dimensional array with approximated etas
    """
    triu_idx = numpy.triu_indices(N, 1)
    diag_idx = numpy.diag_indices(N)
    # Transform thetas
    theta1 = theta[:N]
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Get unnormalized probs
    psi_i = numpy.ones([N, 2])
    psi_i[:,1] = numpy.exp(theta1)
    phi_ij = numpy.ones([N, N, 4])
    phi_ij[:, :, 1] = numpy.exp(theta1[:, numpy.newaxis])
    phi_ij[:, :, 2] = numpy.exp(theta1[:, numpy.newaxis].T)
    phi_ij[:, :, 3] = numpy.exp(theta1[:, numpy.newaxis] + theta1[:, numpy.newaxis].T + theta2)
    phi_ij[diag_idx[0], diag_idx[1], :] = 1
    # Initialize beliefs and Lagrange multipliers
    b_i = .5*numpy.ones([N, 2])
    b_ij = .25*numpy.ones([N, N, 4])
    b_ij[diag_idx[0], diag_idx[1], :] = 0
    lambda_ij = numpy.zeros([N, N, 2])
    gamma_ij = numpy.zeros([N, N])
    # Start CCCP
    eta1, eta2, bethe_energy = outer_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N)
    # Reshape the expectation parameters and return
    eta = numpy.empty(theta.shape)
    eta[:N] = eta1[:, 1]
    eta[N:] = eta2[triu_idx[0], triu_idx[1], 3]
    return eta, bethe_energy


def outer_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Outer loop of CCCP to update beliefs

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with Lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with Lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        first order, second order believes, bethe energy

    """
    # Compute Bethe energy
    bethe_E = bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N)
    conv_crit = numpy.inf
    bethe = [bethe_E]

    while conv_crit > 1e-4:
        # Until convergence update Lagrange multipliers and beliefs
        lambda_ij, gamma_ij = inner_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        b_i, b_ij = update_beliefs(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        # Compute Bethe energy
        bethe_E_old = bethe_E
        bethe_E = bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N)
        bethe.append(bethe_E)
        conv_crit = numpy.absolute((bethe_E_old - bethe_E) / bethe_E_old)

    return b_i, b_ij, bethe_E


def inner_loop(b_i, b_ij, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Inner loop of CCCP to find lagrange multipliers

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    # Compute energy
    dual_E = compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
    conv_crit = numpy.inf

    while conv_crit > 1e-10:
        # Until not converged update lagrange multipliers
        lambda_ij = update_lambda(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        gamma_ij = update_gamma(phi_ij, lambda_ij, N)
        # Compute dual energy
        dual_E_old = dual_E
        dual_E = compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N)
        conv_crit = (dual_E_old - dual_E) / dual_E_old

    # Return
    return lambda_ij, gamma_ij


def update_lambda(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update lambdas.

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """

    # Diagonal indices
    diag_idx = numpy.diag_indices(N)
    p = diag_idx[0]
    # Iterate through each (off-)diagonal

    for i in range(1, N):
        q = (diag_idx[1] + i) % N
        # Update lambda_ij(0)
        exp2lambda0 = (numpy.sum(phi_ij[p, q, :2] *
                                 numpy.exp(-lambda_ij[q, p] - gamma_ij[p, q, numpy.newaxis]), axis=1)) \
                                 /(psi_i[q, 0] * (b_i[q, 0] / psi_i[q, 0]) ** (N - 1) * numpy.exp((N - 1) +
                                 numpy.sum(lambda_ij[:, q, 0], axis=0) - lambda_ij[p, q, 0]))
        lambda_ij[p, q, 0] = .5 * numpy.log(exp2lambda0)
        # Update lambda_ij(1)
        exp2lambda1 = (numpy.sum(phi_ij[p, q, 2:] *
                                 numpy.exp(-lambda_ij[q, p] - gamma_ij[p, q, numpy.newaxis]), axis=1)) \
                                 /(psi_i[q, 1] * (b_i[q, 1] / psi_i[q, 1]) ** (N - 1) * numpy.exp((N - 1) +
                                 numpy.sum(lambda_ij[:, q, 1], axis=0) - lambda_ij[p, q, 1]))
        lambda_ij[p, q, 1] = .5 * numpy.log(exp2lambda1)

    # Return lambdas
    return lambda_ij


def update_gamma(phi_ij, lambda_ij, N):
    """ Update gammas.

    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """

    diag_idx = numpy.diag_indices(N)
    # Compute lambdas for all pair states (0,0),(1,0),(0,1) and (1,1)
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    # Compute new gammas
    gamma_new = numpy.log(numpy.sum(phi_ij * numpy.exp(-1. - lmbd_tmp), axis=2))
    gamma_new[diag_idx] = 0
    # Return
    return gamma_new


def update_beliefs(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update beliefs.

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij
    """
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    b_ij_new = phi_ij * numpy.exp(-1. - lmbd_tmp - gamma_ij[:, :, numpy.newaxis])
    b_i_new = psi_i * numpy.exp(-1. + (N - 1.) + numpy.sum(lambda_ij, axis=0)) * (b_i / psi_i) ** (N - 1)

    return b_i_new, b_ij_new


def compute_dual_energy(b_i, phi_ij, psi_i, lambda_ij, gamma_ij, N):
    """ Update dual energy (maximized by inner loop)

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]

    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param numpy.ndarray lambda_ij:
        [c,c,2] array with lagrange multipliers for conditions sum_j(b_ij)=b_i
    :param numpy.ndarray gamma_ij:
        [c,c] array with lagrange multipliers for conditions sum_ij(b_ij)=1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    # Compute lambdas for all pair states (0,0),(1,0),(0,1) and (1,1)
    lmbd_tmp = numpy.zeros([N, N, 4])
    lmbd_tmp[:, :, 0] = lambda_ij[:, :, 0] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 1] = lambda_ij[:, :, 0] + lambda_ij[:, :, 1].T
    lmbd_tmp[:, :, 2] = lambda_ij[:, :, 1] + lambda_ij[:, :, 0].T
    lmbd_tmp[:, :, 3] = lambda_ij[:, :, 1] + lambda_ij[:, :, 1].T
    # Compute dual energy
    dual_energy = -numpy.sum(phi_ij * numpy.exp(-1. - lmbd_tmp - gamma_ij[:, :, numpy.newaxis]))
    dual_energy -= numpy.sum(
        psi_i * numpy.exp(-1. + (N - 1.) + numpy.sum(lambda_ij, axis=0)) * (b_i / psi_i) ** (N - 1))
    dual_energy -= numpy.sum(gamma_ij)
    return dual_energy


def bethe_free_energy(b_i, b_ij, psi_i, phi_ij, N):
    """ Compute Bethe energy. (Minimized by outer loop)

    :param numpy.ndarray b_i:
        [c,2] array with first order beliefs for silence in [:,0] and firing in [:,1]
    :param numpy.ndarray b_ij:
        [c,c,4] array with second order beliefs for pair patterns (0,0),(1,0),(0,1) and (1,1)
    :param numpy.ndarray phi_ij:
        [c,c,4] array with exp(theta_i x_i + theta_j x_j + theta_ij x_ij) for pair patterns
    :param numpy.ndarray psi_i:
        [c,4] array with exp(theta_i x_i) for x_i = 0 and 1
    :param int N:
        number of cells

    :returns:
        lambda_ij, gamma_ij

    """
    triu_idx = numpy.triu_indices(N, 1)
    # Compute Bethe energy
    bethe_E = numpy.sum(b_ij[triu_idx] * numpy.log(b_ij[triu_idx] / phi_ij[triu_idx])) \
              - ((N - 1.) - 1.) * numpy.sum(b_i * numpy.log(b_i / psi_i))
    return bethe_E


def compute_eta_BP(theta, N, alpha=.5):
    """ Computes the expectation parameters for given theta according to Bethe approximation and belief propagation

    :param numpy.ndarray theta:
        (d,) dimensional array with natural parameters in it
    :param int N:
        Number of cells
    :param float alpha:
        Step size for message update (default=0.5)
    :returns:
        (d,) dimensional array with approximated etas
    """
    # Upper triangle indices
    triu_idx = numpy.triu_indices(N, 1)
    # First order theta in square matrix form
    from_idx, to_idx = numpy.meshgrid(numpy.arange(N), numpy.arange(N))
    theta1 = theta[to_idx]
    # Second order theta in square matrix form
    theta2 = numpy.zeros([N, N])
    theta2[triu_idx] = theta[N:]
    theta2 += theta2.T
    # Calculate unnormalized probabilities for message computation
    psi_i = numpy.exp(theta1)
    psi_i_ij = numpy.exp(theta1 + theta2)
    # Actual belief propogation algorithm
    messages = propagate_beliefs(psi_i, psi_i_ij, N, alpha)
    # Compute beliefs from messages
    b_i, b_ij = compute_beliefs_BP(messages, theta1, theta2, N)
    # Get eta vector
    eta = numpy.empty(theta.shape)
    eta[:N] = b_i
    eta[N:] = b_ij[triu_idx]
    return eta


def propagate_beliefs(psi_i, psi_i_ij, N, alpha=.5):
    """ Actual belief propagation algorithm [Yedidia, 2001]

    :param numpy.ndarray psi_i:
        (c,) dimensional array with exp(theta_i)
    :param numpy.ndarray psi_i_ij:
        (c,c) dimensional array with exp(theta_i + theta_ij)
    :param int N:
        Number of cells
    :param float alpha:
        Step size for message update (default=0.5)
    :return
        (c,c,2) dimensional array with messages from cells to each other about their states
    """

    # Initialize messages
    messages = numpy.ones([N,N,2])
    # Initialize convergence criteria
    message_difference = numpy.inf
    iter_num = 0

    while message_difference > 1e-15 and iter_num <= 1000:
        # Initialize matrix for updated messages
        new_messages = numpy.ones([N,N,2])
        # Compute log of old messages
        log_messages = numpy.log(messages)
        # Marginalize over message sending neurons
        sum_log_messages = numpy.sum(log_messages, axis=0)
        # Compute new messages for neurons being silent
        new_messages[:,:,0] = psi_i*numpy.exp(sum_log_messages[:, 1, numpy.newaxis] - log_messages[:, :, 1].T)\
                                        + numpy.exp(sum_log_messages[:, 0, numpy.newaxis] - log_messages[:, :, 0].T)
        # Compute new messages for neurons firing
        new_messages[:,:,1] = psi_i_ij*numpy.exp(sum_log_messages[:, 1, numpy.newaxis] - log_messages[:, :, 1].T)\
                                        + numpy.exp(sum_log_messages[:, 0, numpy.newaxis]-log_messages[:, :, 0].T)
        # Compute normalization
        k = numpy.sum(new_messages, axis=2)
        new_messages = new_messages/k[:,:,numpy.newaxis]
        # Maximal change in messages
        message_difference = numpy.amax(numpy.absolute(messages - new_messages))
        # Normalize and update messages
        M = (1. - alpha)*messages + alpha*new_messages
        k = numpy.sum(M, axis=2)
        messages = M/k[:, :, numpy.newaxis]
        iter_num += 1
        # Raise exception if not converged
        if iter_num == 1000:
            raise Exception('BP algorithm did not converge!')

    # Return messages
    return messages


def compute_beliefs_BP(messages, theta1, theta2, N):
    """

    :param numpy.ndarray messages:
        (c,c,2) dimensional array with messages from cells to each other about their states
    :param numpy.ndarray theta1:
        (c,c) array with theta_i in rows
    :param numpy.ndarray theta2:
        (c,c) array with theta_ij
    :param int N:
        Number of cells

    :return:
        (c) array containing the belief that a cell fired and (c,c) array that a pair of cells fired
    """
    b_i = numpy.empty([N, 2])
    # Compute unnormalized first order beliefs
    b_i[:, 1] = numpy.exp(theta1[:, 0])*numpy.prod(messages[:, :, 1], axis=0)
    b_i[:, 0] = numpy.prod(messages[:, :, 0], axis=0)
    # Normalize
    k_i = numpy.sum(b_i, axis=1)
    b_i /= k_i[:,numpy.newaxis]
    # Compute unnormalized pair beliefs for x_i = 1
    b_ij = numpy.empty([N, N, 2])
    b_ij[:,:,0] = numpy.exp(theta1)*numpy.prod(messages[:, :, 1], axis=0)[:,numpy.newaxis]/messages[:, :, 1].T\
                    *numpy.prod(messages[:, :, 0],axis=0)[numpy.newaxis,:]/messages[:, :, 0]
    b_ij[:,:,1] = numpy.exp(theta1 + theta1.T + theta2)*numpy.prod(messages[:, :, 1], axis=0)[:,numpy.newaxis]\
                    /messages[:, :, 1].T*numpy.prod(messages[:, :, 1], axis=0)[numpy.newaxis, :]/messages[:, :, 1]
    # Normalize
    k = numpy.sum(b_ij, axis=2)/b_i[:, 1, numpy.newaxis]
    b_ij /= k[:, :, numpy.newaxis]
    # Return
    return b_i[:, 1], b_ij[:, :, 1]