"""
Functions for computing maximum a-posterior probability estimates of natural
parameters given the observed data.

---

State-Space Analysis of Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012)
Copyright (C) 2014  Thomas Sharp (thomas.sharp@riken.jp)

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

import probability
import transforms



# Named function pointers to MAP estimators
# SEE BOTTOM OF FILE

# Parameters for gradient-ascent methods of MAP estimation
MAX_GA_ITERATIONS = 100
GA_CONVERGENCE = 1e-4


def newton_raphson(emd, t):
    """
    Computes the MAP estimate of the natural parameters at some timestep, given
    the observed spike patterns at that timestep and the one-step-prediction
    mean and covariance for the same timestep.

    TODO update comments to elaborate on how this method differs from the others

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.
    """
    # Extract observed patterns and one-step predictions for time t
    y_t = emd.y[t,:]
    theta_o = emd.theta_o[t,:]
    sigma_o = emd.sigma_o[t,:,:]
    # Compute the inverse of one-step covariance
    sigma_o_i = numpy.linalg.inv(sigma_o)
    R = emd.R
    max_dlpo = 1.
    # Initialise theta_max to the smooth theta value of the previous iteration
    theta_max = emd.theta_s[t,:]
    # Use non-normalised posterior prob. as loop guard
    iterations = 0
    # Iterate the gradient ascent algorithm until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Compute the eta of the current theta values
        p = transforms.compute_p(theta_max)
        eta = transforms.compute_eta(p)
        # Compute the first derivative of the posterior prob. w.r.t. theta_max
        dllk = R * (y_t - eta)
        dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
        dlpo = dllk + dlpr
        # Compute the second derivative of the posterior prob. w.r.t. theta_max
        ddlpo = -R * transforms.compute_fisher_info(p, eta) - sigma_o_i
        # Dot the results to climb the gradient, and accumulate the result
        ddlpo_i = numpy.linalg.inv(ddlpo)
        theta_max -= numpy.dot(ddlpo_i, dlpo)
        # Update previous and current posterior prob.
        # lpp = lpc
        # lpc = probability.log_likelihood(y_t, theta_max, R) +\
        #       probability.log_multivariate_normal(theta_max, theta_o, sigma_o)
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        # Check for check for overrun
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior gradient-ascent '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    return theta_max, -ddlpo_i


def conjugate_gradient(emd, t):
    """ Fits with `Nonlinear Conjugate Gradient Method
    <https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method>`_.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.
    """

    # Get observations, one-step prediction and constants
    y_t = emd.y[t, :]
    R = emd.R
    theta_o = emd.theta_o[t, :]
    sigma_o = emd.sigma_o[t, :, :]
    sigma_o_i = numpy.linalg.inv(sigma_o)
    # Initialize theta with previous smoothed theta
    theta_max = emd.theta_s[t, :]
    # Get p and eta values for current theta
    p = transforms.compute_p(theta_max)
    eta = transforms.compute_eta(p)
    # Compute derivative of posterior
    dllk = R*(y_t - eta)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0
    # Get theta gradient
    d_th = dlpo
    # Set initial search direction
    s = dlpo
    # Compute line search
    alpha = transforms.compute_alpha(R, p, s, d_th, sigma_o_i)
    # Update theta
    theta_max += alpha * d_th

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Compute p and eta at current theta
        p = transforms.compute_p(theta_max)
        eta = transforms.compute_eta(p)
        # Set current theta gradient to previous
        d_th_prev = d_th
        # Compute derivative of posterior
        dllk = R * (y_t - eta)
        dlpr = - numpy.dot(sigma_o_i, theta_max - theta_o)
        dlpo = dllk + dlpr
        # The new theta gradient
        d_th = dlpo
        # Calculate beta
        beta = transforms.compute_beta(d_th, d_th_prev)
        # New search direction
        s = d_th + beta * s
        # Line search
        alpha = transforms.compute_alpha(R, p, s, dlpo, sigma_o_i)
        # Update theta
        theta_max += alpha * s
        # Get maximal entry of log posterior gradient divided by number of trials
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior conjugate-gradient '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final covariance matrix
    p = transforms.compute_p(theta_max)
    eta = transforms.compute_eta(p)
    ddllk = - R*transforms.compute_fisher_info(p, eta)
    ddlpo = ddllk - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)

    return theta_max, -ddlpo_i


def bfgs(emd, t):
    """ Fits due to `Broyden-Fletcher-Goldfarb-Shanno algorithm
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_.

    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    :param int t:
        Timestep for which to compute the maximum posterior probability.

    :returns:
        Tuple containing the mean and covariance of the posterior probability
        density, each as a numpy.ndarray.
    """
    # Get observations, one-step prediction and constants
    y_t = emd.y[t, :]
    R = emd.R
    theta_o = emd.theta_o[t, :]
    sigma_o = emd.sigma_o[t, :, :]
    sigma_o_i = numpy.linalg.inv(sigma_o)
    # Initialize theta with previous smoothed theta
    theta_max = emd.theta_s[t, :]
    # Get p and eta values for current theta
    p = transforms.compute_p(theta_max)
    eta = transforms.compute_eta(p)
    # Initialize the estimate of the inverse fisher info
    ddlpo_i_e = numpy.identity(theta_max.shape[0])/R
    # Compute derivative of posterior
    dllk = R*(y_t - eta)
    dlpr = -numpy.dot(sigma_o_i, theta_max - theta_o)
    dlpo = dllk + dlpr
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0

    # Iterate until convergence or failure
    while max_dlpo > GA_CONVERGENCE:
        # Compute direction for line search
        s_dir = numpy.dot(dlpo, ddlpo_i_e)
        # Get alpha for optimal point i s-direction
        alpha = transforms.compute_alpha(R, p, s_dir, dlpo, sigma_o_i)
        # Compute theta change
        d_theta = alpha*s_dir
        # Update theta, eta and p
        theta_max += d_theta
        p = transforms.compute_p(theta_max)
        eta = transforms.compute_eta(p)
        # Set current log posterior gradient to previous
        dlpo_prev = dlpo
        # Compute new log posterior gradient
        dllk = R*(y_t - eta)
        dlpr = -numpy.dot(sigma_o_i, theta_max-theta_o)
        dlpo = dllk + dlpr
        # Difference in log posterior gradients
        dlpo_diff = dlpo_prev - dlpo
        # Project gradient change on theta change
        dlpo_diff_dth = numpy.inner(dlpo_diff, d_theta)
        # Compute the estimate of covariance matrix according to Sherman-Morrison Formula
        a = (dlpo_diff_dth + numpy.dot(dlpo_diff, numpy.dot(ddlpo_i_e, dlpo_diff.T)))*numpy.outer(d_theta, d_theta)
        b = numpy.inner(d_theta, dlpo_diff)**2
        c = numpy.dot(ddlpo_i_e, numpy.outer(dlpo_diff, d_theta)) + numpy.outer(d_theta, numpy.inner(dlpo_diff, ddlpo_i_e))
        d = dlpo_diff_dth
        ddlpo_i_e += (a/b - c/d)
        # Get maximal entry of log posterior gradient divided by number of trials
        max_dlpo = numpy.amax(numpy.absolute(dlpo)) / R
        # Count iterations
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception('The maximum-a-posterior BFGS '+\
                'algorithm did not converge before reaching the maximum '+\
                'number iterations.')

    # Compute final covariance matrix
    ddllk = -R*transforms.compute_fisher_info(p, eta)
    ddlpo = ddllk - sigma_o_i
    ddlpo_i = numpy.linalg.inv(ddlpo)

    return theta_max, -ddlpo_i


# Named function pointers to MAP estimators
functions = {'nr': newton_raphson,
             'cg': conjugate_gradient,
             'bf': bfgs}
