"""
Chord diagram for the equilibrium (state-space log-linear) Ising model.

Visualizes symmetric pairwise couplings J_{ij} as a chord diagram with
one undirected arc per neuron pair. Counterpart of ssll_kinetic.chord
but adapted for the symmetric (equilibrium) case: no flow direction,
no gradient — each arc has a single color from J_{ij}.

- Nodes on a circle, colored by bias (field) h_i = theta_s[t, i]
- Arcs colored by coupling J_{ij} (seismic cmap)
- Arc width proportional to |J_{ij}|
- Optional percentile threshold on |J_{ij}| to hide weak edges

Usage:
    from ssll.chord import show_chord, show_chord_snapshots

    show_chord(emd, dt=0.001, t=25)
    show_chord_snapshots(emd, dt=0.001, times=[0.05, 0.15, 0.25])
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _ensure_save_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_coupling_matrix(emd, t):
    """
    Extract the symmetric pairwise coupling matrix at time bin t.

    theta_s[t] is laid out as [h_1, ..., h_N, J_{1,2}, J_{1,3}, ...,
    J_{N-1,N}] (upper triangle, row-major). This routine repacks the
    pairwise block into an N x N symmetric matrix.

    :param emd: container.EMData
        Fitted model with theta_s of shape (T, D), D = N + N(N-1)/2.
    :param t: int
        Time bin index.
    :returns: (J, h)
        J: (N, N) symmetric coupling matrix with zero diagonal.
        h: (N,)  bias / field vector.
    """
    N = emd.N
    h = emd.theta_s[t, :N]
    J = np.zeros((N, N))
    triu = np.triu_indices(N, k=1)
    J[triu] = emd.theta_s[t, N:N + len(triu[0])]
    J = J + J.T
    return J, h


def _draw_arc(ax, p_i, p_j, val, edge_cm, edge_norm,
              linewidth, curvature=0.3, alpha=0.7, n_seg=50):
    """
    Draw a single-color quadratic-Bezier arc between p_i and p_j.

    :param p_i, p_j: array-like (2,) — endpoint positions.
    :param val: float — coupling value (for colormap lookup).
    :param curvature: float — 0 = straight, 1 = bow through origin.
    """
    p_i = np.asarray(p_i, dtype=float)
    p_j = np.asarray(p_j, dtype=float)

    midpoint = 0.5 * (p_i + p_j)
    origin = np.array([0.0, 0.0])
    ctrl = midpoint + curvature * (origin - midpoint)

    ts = np.linspace(0, 1, n_seg + 1)
    pts = (((1 - ts) ** 2)[:, None] * p_i
           + (2 * (1 - ts) * ts)[:, None] * ctrl
           + (ts ** 2)[:, None] * p_j)

    color = edge_cm(edge_norm(val))
    color = (color[0], color[1], color[2], alpha)
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments, colors=[color] * (n_seg),
                        linewidths=linewidth, zorder=10)
    ax.add_collection(lc)


def show_chord(emd, dt, t, ax=None,
               node_cmap='hot', edge_cmap='seismic',
               bias_lim=None, limit_ij=None, threshold=0,
               curvature=0.3,
               node_size=300, fp_chord=None, dpi=200, figsize=None):
    """
    Display a chord diagram of pairwise couplings at a single time bin.

    :param emd: container.EMData — fitted model.
    :param dt: float — bin size in seconds (used only for the title).
    :param t: int — time bin index.
    :param threshold: float — percentile (0-100) on |J_{ij}|. Edges below
        this percentile are hidden. Default 0 (show all).
    :param curvature: float — arc curvature. Default 0.3.
    :param ax: matplotlib Axes or None — if given, draw on this axes.
    :param node_cmap: str — colormap for nodes (bias).
    :param edge_cmap: str — colormap for edges (coupling).
    :param bias_lim: float or None — symmetric limit for node colormap.
    :param limit_ij: float or None — symmetric limit for edge colormap.
    :param node_size: int — marker size for nodes.
    :param fp_chord: str or None — if given, save figure to this path.
    """
    N = emd.N
    J, h = get_coupling_matrix(emd, t)

    triu_idx = np.triu_indices(N, k=1)
    abs_couplings = np.abs(J[triu_idx])
    if threshold > 0 and len(abs_couplings) > 0:
        cutoff = np.percentile(abs_couplings, threshold)
    else:
        cutoff = 0.0

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    fontsize = 8
    if N > 20:
        gap = 2 * np.sin(np.pi / N)
        gap_ref = 2 * np.sin(np.pi / 20)
        node_size = max(15, min(node_size, int(375 * np.sqrt(gap / gap_ref))))
        fontsize = max(4, min(8, 8 * np.sqrt(node_size / 300)))

    if bias_lim is None:
        bias_lim = np.max(np.abs(emd.theta_s[:, :N]))
    if limit_ij is None:
        limit_ij = np.max(np.abs(emd.theta_s[:, N:]))
    if limit_ij == 0:
        limit_ij = 1.0

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (7, 7))

    node_cm = plt.get_cmap(node_cmap)
    edge_cm = plt.get_cmap(edge_cmap)
    edge_norm = Normalize(vmin=-limit_ij, vmax=limit_ij)
    node_norm = Normalize(vmin=-bias_lim, vmax=bias_lim)

    for i in range(N):
        for j in range(i + 1, N):
            val = J[i, j]
            if threshold and abs(val) < cutoff:
                continue
            linewidth = float(np.clip(3.0 * abs(val) / limit_ij, 0.5, 3.0))
            _draw_arc(ax, pos[i], pos[j], val, edge_cm, edge_norm,
                      linewidth, curvature=curvature)

    node_colors = [node_cm(node_norm(h[k])) for k in range(N)]
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
               edgecolors='black', linewidths=0.5, zorder=5)
    for k in range(N):
        ax.text(pos[k, 0], pos[k, 1], str(k), ha='center', va='center',
                fontsize=fontsize, color='#D3D3D3', zorder=6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('t = %.3f s' % (t * dt), fontsize=14)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    if own_fig:
        fig = ax.get_figure()
        sm_node = plt.cm.ScalarMappable(cmap=node_cm, norm=node_norm)
        sm_node.set_array([])
        try:
            cbar_n = fig.colorbar(sm_node, ax=ax, fraction=0.03, pad=0.04,
                                  location='left')
        except TypeError:
            cbar_n = fig.colorbar(sm_node, ax=ax, fraction=0.03, pad=0.04)
        cbar_n.set_label('$h_i$', fontsize=12)

        sm_edge = plt.cm.ScalarMappable(cmap=edge_cm, norm=edge_norm)
        sm_edge.set_array([])
        cbar_e = fig.colorbar(sm_edge, ax=ax, fraction=0.03, pad=0.02)
        cbar_e.set_label('$J_{ij}$', fontsize=12)

        if fp_chord is not None:
            _ensure_save_dir(fp_chord)
            fig.savefig(fp_chord, dpi=dpi, bbox_inches='tight')
        plt.show()


def show_chord_snapshots(emd, dt, times, threshold=0, curvature=0.3,
                         fp_chord=None, dpi=200, figsize=None,
                         node_cmap='hot', edge_cmap='seismic'):
    """
    Display chord diagrams at multiple time points with shared color scales.

    :param emd: container.EMData — fitted model.
    :param dt: float — bin size in seconds.
    :param times: list of float — times in seconds.
    :param threshold: float — percentile on |J_{ij}|. Default 0.
    """
    n_panels = len(times)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (6 * ncols, 6 * nrows)

    N = emd.N
    bias_lim = np.max(np.abs(emd.theta_s[:, :N]))
    limit_ij = np.max(np.abs(emd.theta_s[:, N:]))
    if limit_ij == 0:
        limit_ij = 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, time_s in enumerate(times):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        t = int(round(time_s / dt))
        show_chord(emd, dt, t, ax=ax,
                   node_cmap=node_cmap, edge_cmap=edge_cmap,
                   bias_lim=bias_lim, limit_ij=limit_ij,
                   threshold=threshold, curvature=curvature)

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    node_cm = plt.get_cmap(node_cmap)
    edge_cm = plt.get_cmap(edge_cmap)
    node_norm = Normalize(vmin=-bias_lim, vmax=bias_lim)
    edge_norm = Normalize(vmin=-limit_ij, vmax=limit_ij)
    sm_node = plt.cm.ScalarMappable(cmap=node_cm, norm=node_norm)
    sm_node.set_array([])
    sm_edge = plt.cm.ScalarMappable(cmap=edge_cm, norm=edge_norm)
    sm_edge.set_array([])

    try:
        fig.tight_layout(rect=[0.11, 0.02, 0.89, 0.98])
    except Exception:
        fig.subplots_adjust(left=0.11, right=0.89, top=0.98, bottom=0.02)

    cax_bias = fig.add_axes([0.02, 0.18, 0.018, 0.64])
    cbar_n = fig.colorbar(sm_node, cax=cax_bias)
    cbar_n.set_label('$h_i$', fontsize=12)
    cax_edge = fig.add_axes([0.965, 0.18, 0.018, 0.64])
    cbar_e = fig.colorbar(sm_edge, cax=cax_edge)
    cbar_e.set_label('$J_{ij}$', fontsize=12)

    if fp_chord is not None:
        _ensure_save_dir(fp_chord)
        fig.savefig(fp_chord, dpi=dpi, bbox_inches='tight')
    plt.show()
