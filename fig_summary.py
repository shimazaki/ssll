"""
Self-contained summary figure for the State-Space Ising Model.

Generates a publication-quality multi-panel figure using synthetic data
and only ssll library functions -- no ssll_jup dependencies required.

Panels:
  A top:    Raster plot of spikes (3 trials)
  A bottom: Population firing rate (data vs fit)
  B top:    Network graphs at 3 time snapshots
  B middle: Mean theta_i over time (original vs shuffled)
  B bottom: Mean theta_ij over time (original vs shuffled)
  C top:    Entropy with credible intervals
  C middle: Entropy ratio with credible intervals
  C bottom-top: p_silence with credible intervals
  C bottom: Heat capacity with credible intervals

Usage:
    cd ssll/
    python fig_summary.py
"""

import numpy
import decimal
import itertools
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib as mpl

import __init__ as ssll
import synthesis
import transforms
import energies
from util import compute_bounds, shuffle_spikes
from thermodynamics import compute_entropy_b, compute_c_b, compute_p_silence_b

# Color constants (same as ssll_jup/utils/display_settings.py)
DEFAULT_ORIGINAL_COLOR = '#FF0000'
DEFAULT_SHUFFLED_COLOR = '#808080'
DEFAULT_DATA_COLOR = '#0000FF'


def main():
    # --- Synthetic data ---
    N, O, T, R = 9, 2, 100, 400
    seed = 42

    theta_true = synthesis.generate_thetas(N, O, T, seed=seed)
    spikes = synthesis.generate_spikes_gibbs(theta_true, N, O, R,
                                             pre_n=100, sample_steps=1, seed=seed)
    shuffled_spikes = shuffle_spikes(spikes)

    # --- Fit models (pseudo likelihood + mean-field TAP) ---
    emd = ssll.run(spikes, O, max_iter=200,
                   param_est='pseudo', param_est_eta='mf', EM_Info=True)
    emd_shuffled = ssll.run(shuffled_spikes, O, max_iter=200,
                            param_est='pseudo', param_est_eta='mf', EM_Info=True)

    energies.get_energies(emd)
    energies.get_energies(emd_shuffled)

    # --- Figure setup ---
    def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    size_in_cm = 19. * numpy.array([1, 1. / 2.])
    size_in_inch = cm2inch(size_in_cm)[0]
    fontsize = 6

    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['xtick.major.width'] = .3
    mpl.rcParams['ytick.major.width'] = .3
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['ytick.major.size'] = 2

    fig = plt.figure(figsize=size_in_inch, dpi=200)

    dt = 1  # bin width (time-step indices)
    theta = emd.theta_s
    time_steps = numpy.arange(T)

    # ==================== Panel A ====================
    # Raster (3 trials stacked)
    for k, (x_off, y_off) in enumerate([(0.07, 0.5), (0.06, 0.47), (0.05, 0.44)]):
        ax = fig.add_axes([x_off, y_off, .25, .4])
        ax.imshow(spikes[:, k, :].transpose(), 'Greys', aspect=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 2:
            ax.set_ylabel('Unit ID', fontsize=fontsize)

    # Population firing rate
    ax = fig.add_axes([.05, 0.1, .25, .33])
    rate = numpy.mean(numpy.mean(spikes, axis=1), axis=1)
    ax.plot(time_steps, rate, linewidth=1, color=DEFAULT_DATA_COLOR)
    ax.plot(time_steps, numpy.mean(emd.eta[:, :N], axis=1), linewidth=1, color=DEFAULT_ORIGINAL_COLOR)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(['Data', 'Fit'], frameon=0, fontsize=int(.9 * fontsize))
    ax.set_ylabel('$p_{\\mathrm{spike}}$', fontsize=fontsize)
    ax.set_xlabel('Time step', fontsize=fontsize)
    ax.yaxis.labelpad = -.05

    # ==================== Panel B ====================
    # Network snapshots
    snapshots_t = [T // 5, T // 2, 4 * T // 5]
    graph_ax = [fig.add_axes([.315, 0.55, .13, .33]),
                fig.add_axes([.42, 0.55, .13, .33]),
                fig.add_axes([.525, 0.55, .13, .33])]

    eta = emd.eta
    eta_i_max = numpy.max(eta[:, :N])
    theta_ij_max = numpy.max(theta[:, N:])
    theta_ij_min = numpy.min(theta[:, N:])
    limit_ij = numpy.max((numpy.abs(theta_ij_max), numpy.abs(theta_ij_min)))
    bounds = compute_bounds(emd, 0)[:, N:, :]

    for i, t in enumerate(snapshots_t):
        ax = graph_ax[i]
        idx = numpy.where(numpy.logical_or(bounds[t, :, 0] > 0, bounds[t, :, 1] < 0))[0]
        all_conns = itertools.combinations(range(N), 2)
        conns = numpy.array(list(all_conns))[idx]
        G1 = nx.Graph()
        G1.add_nodes_from(range(N))
        for link, index in zip(conns, idx):
            G1.add_edge(*link, color=theta[t, N + index], weight=numpy.abs(theta[t, N + index]))
        pos1 = nx.circular_layout(G1, scale=0.035)
        edges = G1.edges()
        colors = [G1[u][v]['color'] for u, v in edges]
        weights = [G1[u][v]['weight'] for u, v in edges]
        net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=ax, node_color=eta[t, :N],
                                           cmap=plt.get_cmap('hot'), vmin=0, vmax=eta_i_max, node_size=25, linewidths=.5)
        e1 = nx.draw_networkx_edges(G1, pos1, ax=ax, edge_color=colors,
                                    edge_cmap=plt.get_cmap('seismic'), edge_vmin=-limit_ij, edge_vmax=limit_ij, width=2 * weights)
        ax.axis('off')
        ax.set_title('t= ' + str(t), fontsize=fontsize)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    cbar_ax = fig.add_axes([0.38, 0.57, 0.1, 0.01])
    cbar = fig.colorbar(net_nodes, cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_linewidth(.5)
    cbar.set_ticks([0, float(round(decimal.Decimal(eta_i_max - 0.05), 1))])
    cbar_ax.set_title('$\\eta_{i}$', fontsize=fontsize)
    cbar_ax = fig.add_axes([0.50, 0.57, 0.1, 0.01])
    cbar = fig.colorbar(e1, cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_linewidth(.5)
    cbar.set_ticks((-1, 1))
    cbar_ax.set_title('$\\theta_{ij}$', fontsize=fontsize)

    # theta_i
    theta_shuffled = emd_shuffled.theta_s
    ax = fig.add_axes([.35, 0.33, .28, .2], frameon=0)
    ax.set_xticks([])
    plt.plot(time_steps, numpy.mean(theta_shuffled[:, :N], axis=1), label='shuffled', color=DEFAULT_SHUFFLED_COLOR, linestyle='--', lw=1)
    plt.plot(time_steps, numpy.mean(theta[:, :N], axis=1), label='original', color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.yaxis.labelpad = -.3
    plt.ylabel('$\\langle \\theta_{i} \\rangle_{i}$', fontsize=fontsize)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
    for snap in snapshots_t:
        ax.add_artist(plt.Line2D((snap, snap), (ymin, ymax), color='black', linewidth=0.3, linestyle='-'))

    # theta_ij
    ax = fig.add_axes([.35, 0.1, .28, .2])
    ax.set_xticks(snapshots_t)
    plt.plot(time_steps, numpy.mean(theta_shuffled[:, N:], axis=1), label='shuffled', color=DEFAULT_SHUFFLED_COLOR, lw=1, linestyle='--')
    plt.plot(time_steps, numpy.mean(theta[:, N:], axis=1), label='original', color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.yaxis.labelpad = -.2
    plt.ylabel('$\\langle \\theta_{ij} \\rangle_{ij}$', fontsize=fontsize)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    for snap in snapshots_t:
        ax.add_artist(plt.Line2D((snap, snap), (ymin, ymax), color='black', linewidth=0.3, linestyle='-'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time step', fontsize=fontsize)

    # ==================== Panel C ====================
    samples = 1000
    threshold = 90
    S_pair, S_pair_bounds, S_ratio, S_ratio_bounds = compute_entropy_b(emd, samples, threshold)
    S_pair_shuffled, S_pair_shuffled_bounds, S_ratio_shuffled, S_ratio_shuffled_bounds = compute_entropy_b(emd_shuffled, samples, threshold)

    # Entropy
    ax = fig.add_axes([.69, 0.7, .28, .15], frameon=0)
    ax.set_xticks([])
    ax.fill_between(time_steps, S_pair_shuffled_bounds[:, 0], S_pair_shuffled_bounds[:, 1], color=DEFAULT_SHUFFLED_COLOR, edgecolor=None, alpha=0.2)
    ax.plot(time_steps, S_pair, color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.fill_between(time_steps, S_pair_bounds[:, 0], S_pair_bounds[:, 1], color=DEFAULT_ORIGINAL_COLOR, edgecolor=None, alpha=0.2)
    ax.set_ylabel('$S_{\\mathrm{pair}}$', fontsize=fontsize)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

    # Entropy ratio
    ax = fig.add_axes([.69, 0.5, .28, .15], frameon=0)
    ax.set_xticks([])
    ax.fill_between(time_steps, S_ratio_shuffled_bounds[:, 0], S_ratio_shuffled_bounds[:, 1], color=DEFAULT_SHUFFLED_COLOR, edgecolor=None, alpha=0.2)
    ax.plot(time_steps, S_ratio, color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.fill_between(time_steps, S_ratio_bounds[:, 0], S_ratio_bounds[:, 1], color=DEFAULT_ORIGINAL_COLOR, edgecolor=None, alpha=0.2)
    ax.set_ylabel('$\\frac{S_{\\mathrm{ind}}-S_{\\mathrm{pair}}}{S_{\\mathrm{0}}-S_{\\mathrm{pair}}}\\ [\\%]$', fontsize=fontsize)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.set_ylim(-0.1, ymax)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

    # p_silence
    ax = fig.add_axes([.69, 0.3, .28, .15], frameon=0)
    ax.set_xticks([])
    p_silence, p_silence_bounds = compute_p_silence_b(emd, samples, threshold)
    p_silence_shuffle, p_silence_bounds_shuffle = compute_p_silence_b(emd_shuffled, samples, threshold)
    ax.fill_between(time_steps, p_silence_bounds_shuffle[:, 0], p_silence_bounds_shuffle[:, 1], color=DEFAULT_SHUFFLED_COLOR, edgecolor=None, alpha=0.2)
    ax.plot(time_steps, p_silence, color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.fill_between(time_steps, p_silence_bounds[:, 0], p_silence_bounds[:, 1], color=DEFAULT_ORIGINAL_COLOR, edgecolor=None, alpha=0.2)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
    ax.set_ylabel('$p_{silence}$', fontsize=fontsize)

    # Heat capacity
    C, C_bounds = compute_c_b(emd, samples, threshold)
    C_shuffled, C_bounds_shuffled = compute_c_b(emd_shuffled, samples, threshold)
    ax = fig.add_axes([.69, 0.1, .28, .15])
    ax.fill_between(time_steps, C_bounds_shuffled[:, 0], C_bounds_shuffled[:, 1], color=DEFAULT_SHUFFLED_COLOR, edgecolor=None, alpha=0.2)
    ax.plot(time_steps, C, color=DEFAULT_ORIGINAL_COLOR, lw=1)
    ax.fill_between(time_steps, C_bounds[:, 0], C_bounds[:, 1], color=DEFAULT_ORIGINAL_COLOR, edgecolor=None, alpha=0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('$C$', fontsize=fontsize)
    ax.set_xlabel('Time step', fontsize=fontsize)

    # ==================== Panel labels ====================
    for label, x_pos, title in [('A', 0.03, f'N={N}, T={T}, R={R}'), ('B', 0.33, ''), ('C', 0.66, '')]:
        ax = fig.add_axes([x_pos, 0.88, .05, .05], frameon=0)
        ax.set_yticks([])
        ax.set_xticks([])
        txt = f'{label}  {title}' if title else label
        ax.text(.0, .0, txt, fontsize=fontsize, fontweight='bold')

    fig.savefig('fig_summary.png', bbox_inches='tight', dpi=200)
    print('Saved fig_summary.png')


if __name__ == '__main__':
    main()
