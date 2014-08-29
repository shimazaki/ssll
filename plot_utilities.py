__author__ = 'Christian Donner'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy
import networkx as nx
import itertools

axis_font = {'size': '13'}
title_font = {'size': '15'}

def rate_correlations_error(theta, theta_est, X, X_est):
    """ Creates Figure 1 of Report
    """

    T, N, R = X.shape
    eta = numpy.mean(X, axis=1)
    eta_est = numpy.mean(X_est, axis=1)
    corrs = numpy.zeros([N, N])
    corrs_est = numpy.zeros([N, N])

    for i in range(N):
        for j in range(i+1,N):
            corrs[i,j] = numpy.mean((X[0,:,i] - numpy.mean(X[0,:,i])) * (X[0,:,j] - numpy.mean(X[0,:,j])))
            corrs[j,i] = corrs[i,j]
            corrs_est[i,j] = numpy.mean((X_est[0,:,i] - numpy.mean(X_est[0,:,i])) * (X_est[0,:,j] - numpy.mean(X_est[0,:,j])))
            corrs_est[j,i] = corrs_est[i,j]

    theta_max = numpy.amax(theta_est)+0.2
    theta_min_est = numpy.amin(theta_est)-0.2
    fig = plt.figure(figsize=(12, 4),facecolor=[1,1,1])
    ax1 = plt.subplot2grid((1, 6), (0, 0), colspan=3)
    ax1.plot([theta_min_est, theta_max],[theta_min_est, theta_max],'k--',
               linewidth=1,alpha=0.5)
    ax1.plot(theta[0][:N], theta_est[:N],'r.')
    ax1.plot(theta[0][N:], theta_est[N:],'k.')
    ax1.set_xlabel('Underlying $\mathbf{\\theta}$',fontsize=20)
    ax1.set_ylabel('Estimated $\mathbf{\\theta}$',fontsize=20)
    ax1.set_xticks([-6,-3,0,3])
    ax1.set_yticks([-6,-3,0,3])
    ax1.set_xlim([theta_min_est, theta_max])
    ax1.set_ylim([theta_min_est, theta_max])
    ax1.set_aspect('equal')
    ax2 = plt.subplot2grid((1, 6), (0, 3), colspan=3)
    ax2.plot([0, numpy.amax(eta_est)+0.05],[0, numpy.amax(eta_est)+0.05],'k--',
             alpha=0.5, linewidth=1)
    ax2.plot(eta, eta_est[0,:N],'k.')
    ax2.set_xlabel('Data Rate $\mathbf{\\eta}$',fontsize=20)
    ax2.set_xticks([0,0.1,0.2,0.3])
    ax2.set_yticks([0.1,0.2,0.3])
    ax2.set_xlim([0, numpy.amax(eta)+0.05])
    ax2.set_ylim([0, numpy.amax(eta_est)+0.05])
    ax2.set_ylabel('Resampled Rate $\mathbf{\\eta}$',fontsize=20)
    ax2.set_aspect('equal')
    fig.tight_layout()
    fig = plt.figure(figsize=(12, 4),facecolor=[1,1,1])
    ax3 = plt.subplot2grid((1, 16), (0, 0), colspan=5)
    im = ax3.imshow(corrs, interpolation='nearest',vmin=-.2,vmax=.2, cmap=plt.get_cmap('seismic'))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('Neuron ID', axis_font)
    ax3.set_ylabel('Neuron ID', axis_font)
    ax3.set_title('Data Correlations',title_font)
    ax3.set_aspect('equal')
    ax4 = plt.subplot2grid((1, 16), (0, 5), colspan=5, sharey=ax3)
    ax4.imshow(corrs_est, interpolation='nearest',vmin=-.2,vmax=.2, cmap=plt.get_cmap('seismic'))
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_xlabel('Neuron ID', axis_font)
    ax4.set_title('Resampled Correlations',title_font)
    ax4.set_aspect('equal')
    ax5 = plt.subplot2grid((1, 16), (0, 10), colspan=6, sharey=ax3)
    ax5.imshow(corrs-corrs_est, interpolation='nearest',vmin=-.2,vmax=.2, cmap=plt.get_cmap('seismic'))
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_xlabel('Neuron ID', axis_font)
    ax5.set_title('Difference',title_font)
    ax5.set_aspect('equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_xticks([])
    cax.set_yticks([])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Corr')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_ticks([-.2,0.,.2])
    fig.tight_layout()
    plt.show()


def plot_graphs(theta, theta_est, X, X_est, time_points, max_conns=20):
    """ Produces network Snap Shots

    """
    N = X.shape[2]
    eta = numpy.mean(X, axis=1)
    eta_est = numpy.mean(X_est, axis=1)
    no_snap_shots = len(time_points)
    max_theta2 = numpy.amax(numpy.absolute(theta_est[:,N:]))
    conns = numpy.array(list(itertools.combinations(range(N),2)))

    for i,t in enumerate(time_points):
        weights = numpy.array(theta[t,N:])
        weight_idx = numpy.argsort(numpy.absolute(weights))[-max_conns:]
        g = nx.Graph()
        g.add_nodes_from(range(N))
        g.add_edges_from(conns[weight_idx,:])
        pos=nx.circular_layout(g)
        nodes = numpy.array(eta[t,:N])
        plt.subplot(2,no_snap_shots,i+1)
        nx.draw(g,pos, node_color=nodes, edge_color=weights[numpy.sort(weight_idx)]
                ,cmap=plt.get_cmap('hot'),edge_cmap=plt.get_cmap('seismic'),vmin=0,vmax=1,
                width=4, edge_vmin=-max_theta2, edge_vmax=max_theta2)
        plt.title('T = %d' %t, fontsize=20)
        weights_est = numpy.array(theta_est[t,N:])
        weight_idx_est = numpy.argsort(numpy.absolute(weights_est))[-max_conns:]
        g2 = nx.Graph()
        g2.add_nodes_from(range(N))
        g2.add_edges_from(conns[weight_idx_est,:])
        pos=nx.circular_layout(g2)
        nodes = numpy.array(eta_est[t,:N])
        plt.subplot(2,no_snap_shots,i+1+no_snap_shots)
        nx.draw(g2,pos,node_color=nodes, edge_color=weights_est[numpy.sort(weight_idx_est)]
                ,cmap=plt.get_cmap('hot'),edge_cmap=plt.get_cmap('seismic'), vmin=0,vmax=1,
                width=4, edge_vmin=-max_theta2, edge_vmax=max_theta2)

    ax1 = plt.subplot(2,no_snap_shots,i+1)
    ax2 = plt.subplot(2,no_snap_shots,i+no_snap_shots+1)
    ax3 = plt.subplot(2,no_snap_shots,1)
    ax4 = plt.subplot(2,no_snap_shots,no_snap_shots+1)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("left", size="5%", pad=0.05)
    cax.set_frame_on(False)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.set_ylabel('Underlying',axis_font)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("left", size="5%", pad=0.05)
    cax.set_frame_on(False)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.set_ylabel('Estimated',axis_font)
    a = numpy.array([[0,1]])
    plt.figure()
    img = plt.imshow(a, cmap="hot")
    plt.gca().set_visible(False)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_xticks([])
    cax.set_yticks([])
    cbar = plt.colorbar(img,cax=cax)
    cbar.set_label('Rate',fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks([0,1])
    a = numpy.array([[-max_theta2,max_theta2]])
    img = plt.imshow(a, cmap="seismic")
    plt.gca().set_visible(False)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_xticks([])
    cax.set_yticks([])
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('$\mathbf{\\theta}$',fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks([-2,0,2])
    plt.show()


def plot_interactions(emd, theta_orig='none', order=2, corder='median', 
    cmap='coolwarm', transparency=0.8):
    """ Pltos time-varying interactions with credible intervals.
        Color is assigned according to their median values. 
        Default 'coolwarm'. For random color, 'Set1'.
        http://matplotlib.org/examples/color/colormaps_reference.html

        This is a wrapper function for plot_theta.
    """
    theta_est = emd.theta_s
    sigma_est = emd.sigma_s

    plot_theta(theta_est, sigma_est, theta_orig, order, 
        corder, cmap, transparency)

def plot_theta(theta_est, sigma_est, theta_orig='none', order=2, 
    corder='median', cmap='coolwarm', transparency=0.8):
    """ Pltos time-varying interactions with credible intervals.
        Color is assigned according to their median values. 
        Default 'coolwarm'. For random color, 'Set1'.
        http://matplotlib.org/examples/color/colormaps_reference.html
    """
    if theta_orig == 'none':
        theta_orig = emd.theta_s

    T, D = theta_est.shape
    ti = range(T)

    #fig = plt.figure(figsize=(12, 4),facecolor=[1,1,1])

    # construct credible intervals
    confb = numpy.zeros([T,D])
    for t in range(T):
        confb[t,:] = 1.96*numpy.sqrt( numpy.diag(sigma_est[t,:,:]) )

    # draw lines from lines with larger to smaller median abs values
    sort_idx = numpy.argsort( numpy.median( numpy.abs(theta_est), 0) )

    if corder == 'max':
        theta_max = numpy.amax(theta_est)
        theta_min = numpy.amin(theta_est)
        cval = (numpy.max(theta_est, 0) - theta_min) / (theta_max - theta_min)
    else:
        theta_max = numpy.max(numpy.median(theta_est, 0))
        theta_min = numpy.amin(numpy.median(theta_est, 0))
        cval = (numpy.median(theta_est, 0) - theta_min) / (theta_max - theta_min)

    for i in sort_idx:
        cmx = plt.get_cmap(cmap) 
        colors = cmx( cval[i] )
        #colors = cm.coolwarm( cval[i] )

        plt.fill_between(ti, theta_est[:,i]-confb[:,i], 
                         theta_est[:,i]+confb[:,i],
                         interpolate=True, color=colors, alpha=transparency)

        plt.plot(ti, theta_est[:,i], linewidth=.7, color=colors)
        plt.plot(ti, theta_orig[:,i],'--', linewidth=.7, color=colors)

    # plot settings
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.tick_params(labelsize=12)

    # color bar
    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, 
            norm=plt.Normalize(vmin=theta_min, vmax=theta_max))
    sm._A = []
    plt.colorbar(sm)

    # 
    plt.ylabel('Interactions $\mathbf{\\theta_{ij}}$', axis_font)
    plt.xlabel('Time [A.U.]', axis_font)
    plt.show()
