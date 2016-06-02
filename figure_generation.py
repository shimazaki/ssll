import numpy
import h5py
import multiprocessing
from functools import partial
import bethe_approximation, synthesis, transforms, __init__, mean_field
from matplotlib import pyplot
import random
import itertools
from scipy.stats.mstats import mquantiles

def get_sampled_eta_psi(i, theta_sampled, N):
    print i
    psi = numpy.empty([100,3])
    eta = numpy.empty([int(N + N*(N-1)/2),100])
    alpha = [.999,1.,1.001]
    for j in range(100):
        for k, a in enumerate(alpha):
            if k == 1:
                eta[:,j], psi[j,k] = bethe_approximation.compute_eta_hybrid(a*theta_sampled[i,:,j], int(N), return_psi=True)
            else:
                psi[j,k] = bethe_approximation.compute_eta_hybrid(a*theta_sampled[i,:,j], int(N), return_psi=True)[1]
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
    transforms.initialise(N, O)
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

    spikes = synthesis.generate_spikes_gibbs_parallel(theta_all, 2 * N, O, R, sample_steps=100)

    print 'Model and Data generated'

    emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo', param_est_eta='bethe_hybrid', lmbda1=100,
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

def plot_figure1(data_path='../Data/', plot_path='../Plots/'):
    import networkx as nx
    N, O = 30, 2
    f = h5py.File(data_path+'figure1data.h5','r')
    # Figure A
    fig = pyplot.figure(figsize=(6,10))
    ax = fig.add_axes([0.1,0.5,.9,.3])
    ax.imshow(-f['data']['spikes'][:,0,:].T, cmap='gray', aspect=5, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes([.075,0.48,.9,.3])
    ax.imshow(-f['data']['spikes'][:,1,:].T, cmap='gray', aspect=5, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes([.05,0.46,.9,.3])
    ax.imshow(-f['data']['spikes'][:,2,:].T, cmap='gray', aspect=5, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Time [AU]', fontsize=16)
    ax.set_ylabel('Neuron ID', fontsize=16)
    ax = fig.add_axes([.05,0.15,.9,.3])
    ax.set_frame_on(False)
    ax.plot(numpy.mean(numpy.mean(f['data']['spikes'][:,:,:],axis=1),axis=1), linewidth=4, color='k')
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([.1,.2,.3])
    ax.set_xticks([50,150,300])
    ax.set_ylabel('Data spike prob.', fontsize=16)
    ax.set_xlabel('Time [AU]', fontsize=16)
    fig.savefig(plot_path + 'spikes.pdf')

    # Figure B
    N = 30
    fig = pyplot.figure(figsize = (20,6))
    theta = f['fit']['theta_s'].value
    sigma_s = f['fit']['sigma_s'].value
    bounds = numpy.empty([theta.shape[0],theta.shape[1] - N,2])
    bounds[:,:,0] = theta[:,N:] - 2.58*numpy.sqrt(sigma_s[:,N:])
    bounds[:,:,1] = theta[:,N:] + 2.58*numpy.sqrt(sigma_s[:,N:])


    graph_ax = [fig.add_subplot(131),
                fig.add_subplot(132),
                fig.add_subplot(133)]
    T = [50,150,300]
    for i, t in enumerate(T):
        idx = numpy.where(numpy.logical_or(bounds[t,:,0] > 0, bounds[t,:,1] < 0))[0]
        conn_idx_all = numpy.arange(0,N*(N-1)/2)
        conn_idx = conn_idx_all[idx]
        all_conns = itertools.combinations(range(N),2)
        conns = numpy.array(list(all_conns))[conn_idx]
        G1 = nx.Graph()
        G1.add_nodes_from(range(N))
        #conns = itertools.combinations(range(30),2)
        G1.add_edges_from(conns)
        pos1 = nx.circular_layout(G1)
        net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=graph_ax[i], node_color=theta[t,:N],
                                           cmap=pyplot.get_cmap('hot'), vmin=-3,vmax=-1.)
        e1 = nx.draw_networkx_edges(G1, pos1, ax=graph_ax[i], edge_color=theta[t,conn_idx].tolist(),
                                    edge_cmap=pyplot.get_cmap('seismic'),edge_vmin=-.7,edge_vmax=.7, width=2)
        graph_ax[i].axis('off')
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(net_nodes, cax=cbar_ax)
    cbar.set_ticks([-3,-2,-1])
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(e1, cax=cbar_ax)
    cbar.set_ticks([-.5,0.,.5])
    fig.savefig(plot_path+'network.pdf', transparent=True)

    # Figure C

    theta = f['data']['theta_all'][:,[34,71,163]]
    theta_fit = f['fit']['theta_s'][:,[34,71,163]]
    sigma_fit = f['fit']['sigma_s'][:,[34,71,163]]


    fig = pyplot.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(311)
    ax1.set_frame_on(False)
    ax1.fill_between(range(0,500),theta_fit[:,0] - 2.58*numpy.sqrt(sigma_fit[:,0]),
                     theta_fit[:,0] + 2.58*numpy.sqrt(sigma_fit[:,0]), color=[.4,.4,.4])
    ax1.plot(range(500), theta[:,0], linewidth=4, color='k')
    ax1.set_yticks([-1,0,1])
    ax1.set_ylim([-1.1,1.1])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax1.set_xticks([])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1 = fig.add_subplot(312)
    ax1.set_frame_on(False)
    ax1.fill_between(range(0,500),theta_fit[:,1] - 2.58*numpy.sqrt(sigma_fit[:,1]),
                     theta_fit[:,1] + 2.58*numpy.sqrt(sigma_fit[:,1]), color=[.5,.5,.5])
    ax1.plot(range(500), theta[:,1], linewidth=4, color='k')
    ax1.set_yticks([-1,0,1])
    ax1.set_ylim([-1.1,1.5])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax1.set_xticks([])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylabel('$\\theta_{ij}$', fontsize=18)
    ax1 = fig.add_subplot(313)
    ax1.set_frame_on(False)
    ax1.fill_between(range(0,500),theta_fit[:,2] - 2.58*numpy.sqrt(sigma_fit[:,2]),
                     theta_fit[:,2] + 2.58*numpy.sqrt(sigma_fit[:,2]), color=[.6,.6,.6])
    ax1.plot(range(500), theta[:,2], linewidth=4, color='k')
    ax1.set_ylim([-1.1,1.1])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax1.set_xticks([50,150,300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xlabel('Time [AU]', fontsize=16)
    ax1.set_yticks([-1,0,1])
    fig.savefig('../Plots/theta_ij.pdf', transparent=True)

    # Figure D

    psi_color = numpy.array([51, 153., 255]) / 256.
    eta_color = numpy.array([0, 204., 102]) / 256.
    S_color = numpy.array([255, 162, 0]) / 256.
    C_color = numpy.array([204, 60, 60]) / 256.
    psi_quantiles = mquantiles(f['sampled_results']['psi_sampled'][:, :, 1], prob=[.01, .99], axis=1)
    psi_true = f['data']['psi_all'].value
    eta_quantiles = mquantiles(numpy.mean(f['sampled_results']['eta_sampled'][:, :N, :], axis=1), prob=[.01, .99],
                               axis=1)
    C_quantiles = mquantiles(f['sampled_results']['C_sampled'][:, :], prob=[.01, .99], axis=1)
    C_true = f['data']['C_all']
    S_quantiles = mquantiles(f['sampled_results']['S_sampled'][:, :], prob=[.01, .99], axis=1)
    S_true = f['data']['S_all']
    eta1 = numpy.empty(f['data']['theta1'].shape)
    eta2 = numpy.empty(f['data']['theta2'].shape)
    T = eta1.shape[0]
    N1, N2 = 15, 15
    transforms.initialise(N1, O)
    for i in range(T):
        p = transforms.compute_p(f['data']['theta1'][i])
        eta1[i] = transforms.compute_eta(p)
        p = transforms.compute_p(f['data']['theta2'][i])
        eta2[i] = transforms.compute_eta(p)

    fig = pyplot.figure(figsize=(8, 9))
    ax1 = fig.add_subplot(411)
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), eta_quantiles[:, 0], eta_quantiles[:, 1], color=eta_color)
    eta_true = 1./ 2.*(numpy.mean(eta1[:, :N1], axis=1) + numpy.mean(eta2[:, :N2], axis=1))
    ax1.fill_between(range(0, 500), eta_quantiles[:, 0], eta_quantiles[:, 1], color=eta_color)
    ax1.plot(range(500), eta_true, linewidth=3, color=eta_color * .8)

    ax1.set_yticks([.1, .2, .3])
    ax1.set_ylim([.09, .35])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax1.set_xticks([50, 150, 300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_ylabel('Spike Prob.', fontsize=16)
    ax1 = fig.add_subplot(412)
    ax1.set_frame_on(False)
    ax1.fill_between(range(0, 500), numpy.exp(psi_quantiles[:, 0]), numpy.exp(psi_quantiles[:, 1]), color=psi_color)
    ax1.plot(range(500), numpy.exp(-psi_true), linewidth=3, color=psi_color * .8)
    # ax1.plot(numpy.exp(-psi_quantiles[:,0]), c=[.5,.5,.5])
    # ax1.plot(, c=[.5,.5,.5])
    # ax1.fill_between(range(0,500),eta_quantiles[:,0], eta_quantiles[:,2], color=eta_color)
    # ax1.plot(range(500),eta_quantiles[:,1], linewidth=3,color=eta_color*.8)
    # ax1.plot(eta_quantiles[:,0], c=[.5,.5,.5])
    # ax1.plot(eta_quantiles[:,1], c=[.5,.5,.5])
    # ax1.plot(numpy.exp(-f['psi_true'].value),'k', linewidth=3)
    # ax1.plot(numpy.exp(-f['psi_true'].value),c=psi_color,linewidth=2)
    # ax1.plot(numpy.mean(f['eta_true'][:,:15], axis=1),'k', linewidth=3)
    # ax1.plot(numpy.mean(f['eta_true'][:,:15], axis=1) ,c=eta_color,linewidth=2)
    ax1.set_yticks([.01, .02, 0.03])
    ax1.set_ylim([.0, .03])
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ax1.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax1.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax1.set_xticks([50, 150, 300])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    # ax1.legend(['Silence', 'Spike'], frameon=0)
    ax1.set_ylabel('Silence Prob.', fontsize=16)
    # Entropy
    ax2 = fig.add_subplot(413)
    ax2.set_frame_on(False)

    ax2.fill_between(range(0, 500), S_quantiles[:, 0], S_quantiles[:, 1], color=S_color)
    ax2.plot(range(500), S_true, linewidth=3, color=S_color * .8)
    ax2.set_xticks([50, 150, 300])
    ax2.set_yticks([15, 20, 25])
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    # ax2.set_yticks([10,15,20])
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('Entropy [bits]', fontsize=16)
    # Heat capacity
    ax2 = fig.add_subplot(414)
    ax2.set_frame_on(False)
    ax2.fill_between(range(0, 500), C_quantiles[:, 0], C_quantiles[:, 1], color=C_color)
    ax2.plot(range(500), C_true, linewidth=3, color=C_color * .8)
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ax2.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    ax2.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax2.set_xticks([50, 150, 300])
    ax2.set_yticks([10, 15, 20])
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlabel('Time [AU]', fontsize=16)
    ax2.set_ylabel('Heat capacity [bits]', fontsize=16)
    fig.savefig(plot_path+'energies.pdf', transparent=True)




def figure2and3(data_path = '../Data/'):
    R, T, N, O = 200, 500, 15, 2
    f = h5py.File(data_path + 'figure1data.h5', 'r')
    theta = f['data']['theta1'].value
    f.close()

    transforms.initialise(N, O)
    psi_true = numpy.empty(T)
    for i in range(T):
        psi_true[i] = transforms.compute_psi(theta[i])
    p = numpy.zeros((T, 2 ** N))
    for i in range(T):
        p[i, :] = transforms.compute_p(theta[i, :])
    # Generate spikes!
    fitting_methods = ['exact', 'bethe_hybrid', 'mf']

    f = h5py.File(data_path + 'figure2and3data.h5', 'w')
    f.create_dataset('psi_true', data=psi_true)
    f.create_dataset('theta_true', data=theta)
    for fit in fitting_methods:
        g = f.create_group(fit)
        g.create_dataset('MISE_theta', shape=[10])
        g.create_dataset('MISE_psi', shape=[10])
    f.close()


    for iteration in range(10):
        print 'Iteration %d' %iteration
        spikes = synthesis.generate_spikes(p, R, seed=1)


        for fit in fitting_methods:
            if fit == 'exact':
                emd = __init__.run(spikes, O, map_function='cg', param_est='exact', param_est_eta='exact')
            else:
                emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo', param_est_eta=fit)

            psi = numpy.empty(T)

            if fit == 'exact':
                for i in range(T):
                    psi[i] = transforms.compute_psi(emd.theta_s[i])
            elif fit == 'bethe_hybrid':
                for i in range(T):
                    psi[i] = bethe_approximation.compute_eta_hybrid(emd.theta_s[i], N, return_psi=1)[1]
            elif fit == 'mf':
                for i in range(T):
                    eta_mf = mean_field.forward_problem(emd.theta_s[i], N, 'TAP')
                    psi[i] = mean_field.compute_psi(emd.theta_s[i], eta_mf, N)

            mise_theta = numpy.mean((theta - emd.theta_s)**2)
            mise_psi = numpy.mean((psi_true - psi) ** 2)
            f = h5py.File(data_path + 'figure2and3data.h5', 'r+')
            g = f[fit]
            g['MISE_theta'][iteration] = mise_theta
            g['MISE_psi'][iteration] = mise_psi
            if iteration == 0:
                g.create_dataset('theta', data=emd.theta_s)
                g.create_dataset('sigma', data=emd.sigma_s)
                g.create_dataset('psi', data=psi)
            f.close()
            print 'Fitted with %s' % fit


def figure4(data_path='../Data/'):

    N, O, R, T = 10, 2, 200, 500
    mu = numpy.zeros(T)
    x = numpy.arange(1, 401)
    mu[100:] = .8 * (3. / (2. * numpy.pi * (x/400.*3.) ** 3)) ** .5 * \
               numpy.exp(-3. * ((x/400.*3.) - 1.) ** 2 / (2. * (x/400.*3.)))

    num_of_networks = 10
    D = transforms.compute_D(N, O)
    thetas = numpy.empty([num_of_networks, T, D])
    etas = numpy.empty([num_of_networks, T, D])
    psi = numpy.empty([num_of_networks, T])
    S = numpy.empty([num_of_networks, T])
    C = numpy.empty([num_of_networks, T])
    transforms.initialise(N,O)
    for i in range(num_of_networks):
        thetas[i] = synthesis.generate_thetas(N, O, T)
        thetas[i,:,:N] += mu[:,numpy.newaxis]
        for t in range(T):
            p = transforms.compute_p(thetas[i,t])
            etas[i,t] = transforms.compute_eta(p)
            psi[i,t] = transforms.compute_psi(thetas[i,t])
            psi1 = transforms.compute_psi(.999*thetas[i,t])
            psi2 = transforms.compute_psi(1.001 * thetas[i, t])
            C[i,t] = (psi1 - 2. * psi[i, t] + psi2) / .001 ** 2
            S[i,t] = -(numpy.sum(etas[i,t]*thetas[i,t]) - psi[i,t])
    C /= numpy.log(2)
    S /= numpy.log(2)

    f = h5py.File(data_path+'figure4data.h5', 'w')
    g1 = f.create_group('data')
    g1.create_dataset('thetas', data=thetas)
    g1.create_dataset('etas', data=etas)
    g1.create_dataset('psi', data=psi)
    g1.create_dataset('S', data=S)
    g1.create_dataset('C', data=C)

    g2 = f.create_group('error')
    g2.create_dataset('MISE_thetas', shape=[num_of_networks])
    g2.create_dataset('MISE_population_rate', shape=[num_of_networks])
    g2.create_dataset('MISE_psi', shape=[num_of_networks])
    g2.create_dataset('MISE_S', shape=[num_of_networks])
    g2.create_dataset('MISE_C', shape=[num_of_networks])
    f.close()

    for i in range(num_of_networks):
        D = transforms.compute_D((i + 1)*N, O)
        theta_all = numpy.empty([T, D])
        triu_idx = numpy.triu_indices(N, k=1)
        triu_idx_all = numpy.triu_indices((i+1)*N, k=1)

        for j in range(i+1):
            theta_all[:, N*j:(j+1)*N] = thetas[j, :, :N]

        for t in range(T):
            theta_ij = numpy.zeros([(i + 1) * N, (i + 1) * N])
            for j in range(i + 1):
                theta_ij[triu_idx[0] + j*N, triu_idx[1] + j*N] = thetas[j, t, N:]

            theta_all[t, (i+1)*N:] = theta_ij[triu_idx_all]

        spikes = synthesis.generate_spikes_gibbs_parallel(theta_all, (i+1) * N, O, R, sample_steps=10)
        emd = __init__.run(spikes, O, map_function='cg', param_est='pseudo', param_est_eta='bethe_hybrid', lmbda1=100,
                           lmbda2=200)

        eta_est = numpy.empty(emd.theta_s.shape)
        psi_est = numpy.empty(T)
        S_est = numpy.empty(T)
        C_est = numpy.empty(T)

        for t in range(T):
            eta_est[t], psi_est[t] = bethe_approximation.compute_eta_hybrid(emd.theta_s[t], (i+1)*N, return_psi=1)
            psi1 = bethe_approximation.compute_eta_hybrid(.999*emd.theta_s[t], (i + 1) * N, return_psi=1)[1]
            psi2 = bethe_approximation.compute_eta_hybrid(.001*emd.theta_s[t], (i + 1) * N, return_psi=1)[1]
            S_est[t] = -numpy.sum(eta_est[t]*emd.theta_s[t]) - psi_est[t]
            C_est[t] = -(psi1 - 2. * psi_est[t] + psi2) / .001 ** 2
        S_est /= numpy.log(2)
        C_est /= numpy.log(2)
        population_rate = numpy.mean(numpy.mean(etas[:i+1,:,:N], axis=0), axis=1)
        population_rate_est = numpy.mean(eta_est[:, :(i+1)*N], axis=1)
        psi_true = numpy.sum(psi[:(i+1),:], axis=0)
        S_true = numpy.sum(S[:(i + 1), :], axis=0)
        C_true = numpy.sum(C[:(i + 1), :], axis=0)
        f = h5py.File(data_path + 'figure4data.h5', 'r+')
        f['error']['MISE_thetas'][i] = numpy.mean((theta_all - emd.theta_s)**2)
        f['error']['MISE_population_rate'][i] = numpy.mean((population_rate - population_rate_est) ** 2)
        f['error']['MISE_psi'][i] = numpy.mean((psi_est - psi_true) ** 2)
        f['error']['MISE_S'][i] = numpy.mean((S_est - S_true) ** 2)
        f['error']['MISE_C'][i] = numpy.mean((C_est - C_true) ** 2)
        f.close()



