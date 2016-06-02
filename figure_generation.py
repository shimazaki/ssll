import numpy
import h5py
import multiprocessing
from functools import partial
import bethe_approximation, synthesis, transforms, __init__

def get_sampled_eta_psi(i, theta_sampled, N):
    print i
    psi = numpy.empty([100,3])
    eta = numpy.empty([int(N + N*(N-1)/2),100])
    for j in range(100):
        for k, beta in enumerate([.999,1.,1.001]):
            if k == 1:
                eta[:,j], psi[j,k] = bethe_approximation.compute_eta_hybrid(beta*theta_sampled[i,:,j], int(N), return_psi=True)
            else:
                psi[j,k] = bethe_approximation.compute_eta_hybrid(beta*theta_sampled[i,:,j], int(N), return_psi=True)[1]
    return eta, psi, i

def figure1(data_path = '../Data/'):
    N, O, R, T = 15, 2, 200, 500
    mu = numpy.zeros(T)
    x = numpy.arange(1, 401)
    mu[100:] = .8 * (3. / (2. * numpy.pi * (x/400.*3.) ** 3)) ** .5 * \
               numpy.exp(-3. * ((x/400.*3.) - 1.) ** 2 / (2. * (x/400.*3.)))
    theta1 = synthesis.generate_thetas(N, O, T)
    theta2 = synthesis.generate_thetas(N, O, T)
    theta1[:, :N] += mu[:, numpy.newaxis]
    theta2[:, :N] += mu[:, numpy.newaxis]
    D = transforms.compute_D(N * 2, O)
    theta_all = numpy.empty([T, D])
    theta_all[:, :N] = theta1[:, :N]
    theta_all[:, N:2 * N] = theta2[:, :N]
    triu_idx = numpy.triu_indices(N, k=1)
    triu_idx_all = numpy.triu_indices(2 * N, k=1)
    for t in range(T):
        theta_ij = numpy.zeros([2 * N, 2 * N])
        theta_ij[triu_idx] = theta1[t, N:]
        theta_ij[triu_idx[0] + N, triu_idx[1] + N] = theta2[t, N:]
        theta_all[t, 2 * N:] = theta_ij[triu_idx_all]

    psi1 = numpy.empty([T, 3])
    psi2 = numpy.empty([T, 3])
    eta1 = numpy.empty(theta1.shape)
    eta2 = numpy.empty(theta2.shape)
    alpha = [.999,1.,1.001]
    for i in range(T):
        for j, a in enumerate(alpha):
            psi1[i, j] = transforms.compute_psi(a * theta1[i])
        p = transforms.compute_p(theta1[i])
        eta1[i] = transforms.compute_eta(p)
        for j, a in enumerate(alpha):
            psi2[i, j] = transforms.compute_psi(a * theta2[i])
        p = transforms.compute_p(theta2[i])
        eta2[i] = transforms.compute_eta(p)

    psi_all = psi1 + psi2
    S1 = -numpy.sum(eta1 * theta1, axis=1) + psi1[:, 1]
    S1 /= numpy.log(2)
    S2 = -numpy.sum(eta2 * theta2, axis=1) + psi2[:, 1]
    S2 /= numpy.log(2)
    S_all = S1 + S2

    C1 = (psi1[:, 0] - 2. * psi1[:, 1] + psi1[:, 2]) / .001 ** 2
    C1 /= numpy.log(2)
    C2 = (psi2[:, 0] - 2. * psi2[:, 1] + psi2[:, 2]) / .001 ** 2
    C2 /= numpy.log(2)

    C_all = C1 + C2

    spikes = synthesis.generate_spikes_gibbs_parallel(theta_all, 2 * N, O, R)

    print 'Model and Data generated'

    emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo', param_est_eta='bethe_hybrid', lmbda1=10,
                   lmbda2=200)

    f = h5py.File(data_path + 'figure1data.h5', 'w')
    g_data = f.create_group('data')
    g_data.create_dataset('theta_all', data=theta_all)
    g_data.create_dataset('psi_all', data=psi_all)
    g_data.create_dataset('S_all', data=S_all)
    g_data.create_dataset('C_all', data=C_all)
    g_data.create_dataset('spikes', data=spikes)
    g_data.create_dataset('theta1', data=theta1)
    g_data.create_dataset('theta2', data=theta2)
    g_data.create_dataset('psi1', data=psi1)
    g_data.create_dataset('S1', data=S1)
    g_data.create_dataset('C1', data=C1)
    g_data.create_dataset('psi2', data=psi2)
    g_data.create_dataset('S2', data=S2)
    g_data.create_dataset('C2', data=C2)
    g_fit = f.create_group('fit')
    g_fit.create_dataset('theta_s', data=emd.theta_s)
    g_fit.create_dataset('sigma_s', data=emd.sigma_s)
    g_fit.create_dataset('Q', data=emd.Q)
    f.close()

    print 'Fit and saved'

    f = h5py.File(data_path + '/figure1data.h5', 'r+')
    g_fit = f['fit']
    theta = g_fit['theta_s'].value
    sigma = g_fit['sigma_s'].value

    X = numpy.random.randn(theta.shape[0], theta.shape[1], 100)
    theta_sampled = theta[:, :, numpy.newaxis] + X * numpy.sqrt(sigma)[:, :, numpy.newaxis]

    T = range(theta.shape[0])
    eta_sampled = numpy.empty([theta.shape[0], theta.shape[1], 100])
    psi_sampled = numpy.empty([theta.shape[0], 100, 3])

    func = partial(get_sampled_eta_psi, theta_sampled=theta_sampled, N=2*N)
    pool = multiprocessing.Pool(10)
    results = pool.map(func, T)

    for eta, psi, i in results:
        eta_sampled[i] = eta
        psi_sampled[i] = psi
    S_sampled = -numpy.sum(eta_sampled*theta_sampled, axis=1) - psi_sampled[:, :, 1]
    S_sampled /= numpy.log(2)
    C_sampled = -(psi_sampled[:, :, 0] - 2.*psi_sampled[:, :, 1] + psi_sampled[:, :, 2])/.001**2
    C_sampled /= numpy.log(2)
    g_sampled = f.create_group('sampled_results')
    g_sampled.create_dataset('theta_sampled', data=theta_sampled)
    g_sampled.create_dataset('eta_sampled', data=eta_sampled)
    g_sampled.create_dataset('psi_sampled', data=psi_sampled)
    g_sampled.create_dataset('S_sampled', data=S_sampled)
    g_sampled.create_dataset('C_sampled', data=C_sampled)
    f.close()

    print 'Done'
