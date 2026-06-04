"""
This file contains all methods that are concerned with the pseudolikelood
approximation.

---

This code implements approximate inference methods for State-Space Analysis of
Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012). It is an extension of
the existing code from repository <https://github.com/tomxsharp/ssll> (For
Matlab Code refer to <http://github.com/shimazaki/dynamic_corr>). We
acknowledge Thomas Sharp for providing the code for exact inference.

In this library are additional methods provided to perform the State-Space
Analysis approximately. This includes pseudolikelihood, TAP, and Bethe
approximations. For details see: <http://arxiv.org/abs/1607.08840>

Copyright (C) 2016

Authors of the extensions: Christian Donner (christian.donner@bccn-berlin.de)
                           Hideaki Shimazaki (shimazaki@brain.riken.jp)

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
import max_posterior
from scipy import sparse
from scipy.special import expit
import transforms
import mean_field
import bethe_approximation

import multiprocessing
from functools import partial


MAX_GA_ITERATIONS = 5000
Fx_s = None
Fx_s_stacked = None  # Precomputed stacked sparse matrices for vectorized ops
Fx_s_stacked_T = None  # Precomputed transpose (CSC) of Fx_s_stacked for fast .T.dot
# Cached structural dims for compute_cond_eta when per-(s) Fx_s is not built
# (skipped for order=2 + cg/bf MAP path; see compute_Fx_s_parallel).
_N_cells = 0
_R_trials = 0
# Cached lex pair (i<j) indices (length n_pairs each) for dense symmetric
# reconstruction of the pair-block of a theta-like (D,) vector. Set in
# compute_Fx_s_parallel when running the order=2 cg/bf direct-CSR path.
_pair_i_idx = None
_pair_j_idx = None
time_bin = -1


def _fs_from_theta_dense(theta, X):
    """Dense replacement for ``Fx_s_stacked_T[t].dot(theta).reshape((R, N), 'F')``.

    For order=2, ``Fx_s_stacked.T @ theta`` reduces to

        fs[r, c] = theta_h[c] + sum_{k != c} Theta2[c, k] * X[r, k]

    where ``Theta2`` is the symmetric (N,N) matrix with off-diagonal
    entries given by the pair-block of ``theta``. Equivalent to a dense
    matmul ``X @ Theta2`` plus a row-broadcast of ``theta_h``; dense BLAS
    is ~1.4x faster than the CSC matvec for N=60, more for larger N.
    """
    N = _N_cells
    Theta2 = numpy.zeros((N, N))
    Theta2[_pair_i_idx, _pair_j_idx] = theta[N:]
    Theta2[_pair_j_idx, _pair_i_idx] = theta[N:]
    return theta[:N] + X.dot(Theta2)


def _build_stacked_csr_o2(spike_cols_flat, spike_cols_offset, N, R,
                          pair_i_idx, pair_j_idx):
    """Build the (D, N*R) stacked Fx_s CSR directly for order=2.

    Equivalent to
        _fast_hstack_csr([compute_Fx_s_t(s, ...) for s in range(N)], R)
    but skips the per-neuron sparse construction. For pair {i, j} with i<j,
    the row N+p has nnz_j entries (cols i*R + spike_cols[j]) followed by
    nnz_i entries (cols j*R + spike_cols[i]); columns are naturally
    ascending since i*R < j*R and each spike_cols block is sorted.

    :param spike_cols_flat: int32 concatenation of spike_cols[k] for k in
        0..N-1 (trials where Xt[:, k] fires).
    :param spike_cols_offset: length N+1 int32, cumulative nnz per neuron.
    :param N, R: number of neurons and trials.
    :param pair_i_idx, pair_j_idx: length n_pairs int32, lex pair enumeration
        (deterministic from N; cached by the caller).
    """
    n_pairs = N * (N - 1) // 2
    D = N + n_pairs

    nnz_arr = spike_cols_offset[1:] - spike_cols_offset[:-1]
    nnz_j_per_pair = nnz_arr[pair_j_idx]
    nnz_i_per_pair = nnz_arr[pair_i_idx]

    nnz_per_row = numpy.empty(D, dtype=numpy.int32)
    nnz_per_row[:N] = R
    nnz_per_row[N:] = nnz_j_per_pair + nnz_i_per_pair
    indptr = numpy.empty(D + 1, dtype=numpy.int32)
    indptr[0] = 0
    numpy.cumsum(nnz_per_row, out=indptr[1:])
    total_nnz = int(indptr[-1])

    indices = numpy.empty(total_nnz, dtype=numpy.int32)
    data = numpy.ones(total_nnz, dtype=numpy.float64)

    # Singleton block: row s covers [s*R, s*R+R).
    indices[:N * R] = numpy.arange(N * R, dtype=numpy.int32)

    pair_start = N * R
    pair_seg_start = indptr[N:-1] - pair_start

    # j-side: row N+p gets nnz_j entries (i*R + spike_cols[j]).
    total_j = int(nnz_j_per_pair.sum())
    if total_j:
        repeated_j_base = numpy.repeat(spike_cols_offset[pair_j_idx],
                                       nnz_j_per_pair)
        j_pair_starts = numpy.empty(n_pairs + 1, dtype=numpy.int32)
        j_pair_starts[0] = 0
        numpy.cumsum(nnz_j_per_pair, out=j_pair_starts[1:])
        within_j = (numpy.arange(total_j, dtype=numpy.int32)
                    - numpy.repeat(j_pair_starts[:-1], nnz_j_per_pair))
        j_vals = (spike_cols_flat[repeated_j_base + within_j]
                  + numpy.repeat((pair_i_idx * R).astype(numpy.int32),
                                 nnz_j_per_pair))
        j_write_pos = (pair_start
                       + numpy.repeat(pair_seg_start, nnz_j_per_pair)
                       + within_j)
        indices[j_write_pos] = j_vals

    # i-side: positioned after the j-side within each pair-row.
    total_i = int(nnz_i_per_pair.sum())
    if total_i:
        repeated_i_base = numpy.repeat(spike_cols_offset[pair_i_idx],
                                       nnz_i_per_pair)
        i_pair_starts = numpy.empty(n_pairs + 1, dtype=numpy.int32)
        i_pair_starts[0] = 0
        numpy.cumsum(nnz_i_per_pair, out=i_pair_starts[1:])
        within_i = (numpy.arange(total_i, dtype=numpy.int32)
                    - numpy.repeat(i_pair_starts[:-1], nnz_i_per_pair))
        i_vals = (spike_cols_flat[repeated_i_base + within_i]
                  + numpy.repeat((pair_j_idx * R).astype(numpy.int32),
                                 nnz_i_per_pair))
        i_write_pos = (pair_start
                       + numpy.repeat(pair_seg_start + nnz_j_per_pair,
                                      nnz_i_per_pair)
                       + within_i)
        indices[i_write_pos] = i_vals

    return sparse.csr_matrix((data, indices, indptr), shape=(D, N * R))


def _enumerate_pair_idx(N):
    """Lex pair (i<j) row indices, deterministic from N. int32."""
    n_pairs = N * (N - 1) // 2
    pair_i = numpy.empty(n_pairs, dtype=numpy.int32)
    pair_j = numpy.empty(n_pairs, dtype=numpy.int32)
    p = 0
    for i in range(N):
        for j in range(i + 1, N):
            pair_i[p] = i
            pair_j[p] = j
            p += 1
    return pair_i, pair_j


def _fast_hstack_csr(mats, n_cols_per_mat):
    """Horizontal-stack same-row CSR matrices without going through COO.

    All input mats are (D, n_cols_per_mat) CSR with identical row count
    and column count. scipy.sparse.hstack routes through bmat -> COO
    conversion (~5ms for 60 mats of (1830, 100)); the direct construction
    here is ~3x faster and produces a CSR with sorted indices.
    """
    n_mats = len(mats)
    D = mats[0].shape[0]
    R = n_cols_per_mat

    # nnz_pr[i, r] = mats[i].indptr[r+1] - mats[i].indptr[r]
    nnz_pr = numpy.empty((n_mats, D), dtype=numpy.int32)
    for i in range(n_mats):
        ip = mats[i].indptr
        nnz_pr[i] = ip[1:] - ip[:-1]

    nnz_per_row = nnz_pr.sum(axis=0)
    indptr_out = numpy.empty(D + 1, dtype=numpy.int32)
    indptr_out[0] = 0
    numpy.cumsum(nnz_per_row, out=indptr_out[1:])
    total_nnz = int(indptr_out[-1])

    indices_out = numpy.empty(total_nnz, dtype=numpy.int32)
    data_out = numpy.empty(total_nnz, dtype=numpy.float64)

    # cum_within[i, r] = sum(nnz_pr[0:i, r]) — start offset (in the row)
    # at which mat i's entries land.
    cum_within = numpy.zeros((n_mats + 1, D), dtype=numpy.int32)
    numpy.cumsum(nnz_pr, axis=0, out=cum_within[1:])

    for i in range(n_mats):
        m = mats[i]
        if m.nnz == 0:
            continue
        rows_for_entries = numpy.repeat(numpy.arange(D), nnz_pr[i])
        within_off = numpy.arange(m.nnz) - m.indptr[rows_for_entries]
        out_pos = (indptr_out[rows_for_entries]
                   + cum_within[i, rows_for_entries]
                   + within_off)
        indices_out[out_pos] = m.indices + i * R
        data_out[out_pos] = m.data

    return sparse.csr_matrix((data_out, indices_out, indptr_out),
                             shape=(D, n_mats * R))


def _build_subset_lookup(subsets, N):
    """Precompute per-neuron subset membership for fast Fx_s_t computation.

    Returns a list of length N. For each neuron s, the entry is a list of
    (subset_index, other_neurons) tuples where other_neurons is a tuple of
    the remaining neurons in subsets that contain s.
    """
    lookup = [[] for _ in range(N)]
    for i, sub in enumerate(subsets):
        for s in sub:
            others = tuple(c for c in sub if c != s)
            lookup[s].append((i, others))
    return lookup


def compute_Fx_s(X, O):
    """
    Constructs F(x_s=1, x_\s), feature vectors of interactions up to the
    'O'th order from observed patterns for conditional likelihood model.

    :param numpy.array X:
        Three dimensional (t, r, c) binary array, where the first dimension is time bin,
        the second is runs (trials) and the third is the number of cells.
    :param int O:
        Order of interactions.

    :returns Fx_s:
        A list composed of (r, D) sparse matrix, where D is the model dimension.
        The list size is the number of bins.
    """
    T, R, N = X.shape
    # Compute each n-choose-k subset of cell IDs up to the 'O'th order
    subsets = transforms.enumerate_subsets(N, O)
    subset_lookup = _build_subset_lookup(subsets, N)
    # Initialize Fx_s
    global Fx_s, Fx_s_stacked, Fx_s_stacked_T
    # List of lists (for each time bin) of sparse matrices (for each cell)
    Fx_s = []
    Fx_s_stacked = []
    Fx_s_stacked_T = []
    # For each time bin
    for i in range(T):
        # Initialize list
        Fx_s.append([])
        # For each cell
        for s in range(N):
            Fx_s[i].append(compute_Fx_s_t(s, X[i,:,:], subsets, subset_lookup))
        # Precompute stacked sparse matrix for vectorized gradient: (D, R*N)
        # Each Fx_s[i][s] is (D, R) CSR. Stack horizontally by neuron.
        M = _fast_hstack_csr(Fx_s[i], R)
        Fx_s_stacked.append(M)
        Fx_s_stacked_T.append(M.T.tocsc())


def compute_Fx_s_t(neuron, Xt, subsets, subset_lookup=None,
                   spike_cols=None):
    """
    Constructs the sparse matrix F(x_s=1, x_\\s) - F(x_s=0, x_\\s) at time t
    for neuron s.  Only subsets containing neuron s produce non-zero rows.

    : param numpy.array Xt:
        Two dimensional (r, c) binary array, where the first dimension is runs
        (trials) and the second is the number of cells.
    : param list spike_cols:
        Optional precomputed per-neuron nonzero-trial index arrays for time t,
        i.e. spike_cols[k] = numpy.nonzero(Xt[:, k])[0]. When provided, pair
        subsets {s, k} look up cols directly instead of recomputing.
    : returns (D, R) sparse CSR feature-difference matrix at time t.
    """
    s = neuron
    R = Xt.shape[0]
    D = len(subsets)
    if subset_lookup is not None:
        entries = subset_lookup[s]
    else:
        entries = []
        for i, sub in enumerate(subsets):
            if s in sub:
                entries.append((i, tuple(c for c in sub if c != s)))
    # Build CSR directly: each row idx gets a slice of column indices
    # corresponding to runs where the product of others' spikes equals 1.
    # Sort entries by row index so we can construct CSR indptr in order.
    entries_sorted = sorted(entries, key=lambda e: e[0])
    all_cols_arrays = []
    nnz_per_row = []
    row_indices = []
    for idx, others in entries_sorted:
        if len(others) == 0:
            cols = numpy.arange(R, dtype=numpy.int32)
        elif len(others) == 1:
            k = others[0]
            cols = (spike_cols[k] if spike_cols is not None
                    else numpy.nonzero(Xt[:, k])[0].astype(numpy.int32))
        else:
            if spike_cols is not None:
                mask = numpy.zeros(R, dtype=bool)
                mask[spike_cols[others[0]]] = True
                for c in others[1:]:
                    m = numpy.zeros(R, dtype=bool)
                    m[spike_cols[c]] = True
                    mask &= m
            else:
                mask = Xt[:, others[0]].astype(bool)
                for c in others[1:]:
                    mask &= Xt[:, c].astype(bool)
            cols = numpy.nonzero(mask)[0].astype(numpy.int32)
        if cols.size:
            row_indices.append(idx)
            nnz_per_row.append(cols.size)
            all_cols_arrays.append(cols)
    if not all_cols_arrays:
        return sparse.csr_matrix((D, R), dtype=numpy.float64)
    indices = numpy.concatenate(all_cols_arrays)
    data = numpy.ones(indices.size, dtype=numpy.float64)
    # Build indptr: zeros except at populated rows, then cumsum.
    indptr = numpy.zeros(D + 1, dtype=numpy.int32)
    for r, n in zip(row_indices, nnz_per_row):
        indptr[r + 1] = n
    numpy.cumsum(indptr, out=indptr)
    return sparse.csr_matrix((data, indices, indptr), shape=(D, R))


def compute_Fx(X, subsets):
    """
    Construct feature vectors of interactions up to the 'O'th order from
    pattern data.

    :param numpy.array X:
        (r, c) binary array, where the first dimension are runs (trials)
        and second cells.
    :param int O:
        Order of interactions

    :returns Fx:
        (D, r) matrix of feature vectors, where D is the model
        dimension.
    """
    # Get spike-matrix metadata
    R, N = X.shape
    # Set up the output array
    Fx = numpy.zeros((len(subsets), R))
    # Vectorize by subset size
    for i in range(len(subsets)):
        sub = subsets[i]
        if len(sub) == 1:
            Fx[i, :] = X[:, sub[0]]
        elif len(sub) == 2:
            Fx[i, :] = X[:, sub[0]] * X[:, sub[1]]
        else:
            sp = X[:, sub]
            Fx[i, :] = sp.sum(axis=1) == len(sub)

    return Fx


def compute_Fx_s_parallel(X, O, map_function='cg'):
    """
    Constructs F(x_s=1, x_\s), feature vectors of interactions up to the
    'O'th order from observed patterns for conditional likelihood model.

    This is a parallelization version of compute_Fx_s.

    :param numpy.array X:
        Three dimensional (t, r, c) binary array, where the first dimension is time bin,
        the second is runs (trials) and the third is the number of cells.
    :param int O:
        Order of interactions.
    :param str map_function:
        Selected MAP function ('cg', 'bf', 'nr'). For 'cg' and 'bf' the
        per-neuron Fx_s[t][s] matrices are unused, so we skip them and
        build Fx_s_stacked directly via _build_stacked_csr_o2 (order=2 only).

    :returns Fx_s:
        A list composed of (r, D) sparse matrix, where D is the model dimension.
        The list size is the number of bins.
    """
    T, R, N = X.shape
    # Compute each n-choose-k subset of cell IDs up to the 'O'th order
    subsets = transforms.enumerate_subsets(N, O)
    # Initialize Fx_s
    global Fx_s, Fx_s_stacked, Fx_s_stacked_T, _N_cells, _R_trials
    global _pair_i_idx, _pair_j_idx
    _N_cells = N
    _R_trials = R
    # List of lists (for each time bin) of sparse matrices (for each cell)
    Fx_s = []
    Fx_s_stacked = []
    Fx_s_stacked_T = []

    # Direct stacked-CSR build for order=2 (cg/bf): skips per-s sparse
    # construction + _fast_hstack_csr entirely. ~4.5x faster than the
    # per-s + hstack path on the build itself.
    skip_per_s = (O == 2 and map_function in ('cg', 'bf'))
    if skip_per_s:
        pair_i, pair_j = _enumerate_pair_idx(N)
        _pair_i_idx = pair_i
        _pair_j_idx = pair_j
        for i in range(T):
            Xt = X[i, :, :]
            spike_cols = [numpy.nonzero(Xt[:, k])[0].astype(numpy.int32)
                          for k in range(N)]
            nnz_arr = numpy.fromiter((sc.size for sc in spike_cols),
                                     dtype=numpy.int32, count=N)
            offsets = numpy.empty(N + 1, dtype=numpy.int32)
            offsets[0] = 0
            numpy.cumsum(nnz_arr, out=offsets[1:])
            if nnz_arr.sum():
                flat = numpy.concatenate(spike_cols).astype(numpy.int32,
                                                            copy=False)
            else:
                flat = numpy.empty(0, dtype=numpy.int32)
            M = _build_stacked_csr_o2(flat, offsets, N, R, pair_i, pair_j)
            Fx_s.append(None)  # per-s not used by cg/bf
            Fx_s_stacked.append(M)
            Fx_s_stacked_T.append(M.T.tocsc())
        return

    subset_lookup = _build_subset_lookup(subsets, N)
    # With direct-CSR build, the per-task cost is small enough that
    # multiprocessing IPC/fork overhead dominates. Run serially.
    for i in range(T):
        Xt = X[i, :, :]
        # Precompute per-neuron nonzero-trial indices so pair subsets
        # in compute_Fx_s_t don't recompute nonzero() N times each.
        spike_cols = [numpy.nonzero(Xt[:, k])[0].astype(numpy.int32)
                      for k in range(N)]
        Fx_s.append([compute_Fx_s_t(s, Xt, subsets, subset_lookup,
                                    spike_cols=spike_cols)
                     for s in range(N)])
        M = _fast_hstack_csr(Fx_s[i], R)
        Fx_s_stacked.append(M)
        Fx_s_stacked_T.append(M.T.tocsc())


def pseudo_newton(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i,
                  param_est_eta='bethe_hybrid'):
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
    max_dlpo = numpy.inf
    # Intialize array for sum of active thetas (r,c)
    fs = numpy.empty([R, N])

    # Iterate until convergence or failure
    while max_dlpo > max_posterior.GA_CONVERGENCE:

        # Initialize gradient and Hessian arrays
        dllk = numpy.zeros(D)
        ddllk = numpy.zeros([D,D])

        # Compute all fs at once via precomputed CSC transpose
        fs_flat = Fx_s_stacked_T[time_bin].dot(theta_max)  # (R*N,)
        fs = fs_flat.reshape((R, N), order='F')

        # Iterate over all cells
        for s_i in range(N):
            # Calculate conditional rate
            etas = expit(fs[:, s_i])
            # Calculate derivative of conditional rate
            deta = - etas * (1-etas)
            # Calculate derivative for neuron
            dllk += Fx_s[time_bin][s_i].dot(X_t[:, s_i] - etas)
            # Compute Hessian for neuron: Fx_s[s_i] @ diag(deta) @ Fx_s[s_i].T
            Fx_si = Fx_s[time_bin][s_i]
            ddllk += Fx_si.multiply(deta).dot(Fx_si.T).toarray()
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
    eta = compute_eta[param_est_eta](theta_max, N)
    ddllk = -R*bethe_approximation.construct_fisher_diag(eta, N)
    #ddllk = pseudo_ddllk(etas,D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ### ddlpo_i = 1./ddlpo#numpy.linalg.inv(ddlpo)
    ddlpo_i = 1./ddlpo
    return theta_max, -ddlpo_i


def pseudo_cg(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i,
              param_est_eta='bethe_hybrid'):
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
    # Calculate fs = sum(theta_I*F_I(x_s = 1, x_/s)) via dense reconstruction
    fs = _fs_from_theta_dense(theta_max, X_t)

    # Initialize stopping criterion variables
    max_dlpo = numpy.inf
    iterations = 0
    # Get likelihood gradient
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    # Get prior
    dlpr = -sigma_o_i*(theta_max - theta_o)
    # Get posterior
    dlpo = dllk + dlpr
    # Initialize theta gradient
    d_th = dlpo
    # Set initial search direction
    s = dlpo
    # Perform first line search
    theta_max, fs = pseudo_line_search2(theta_max, X_t, s, fs, dlpo, sigma_o_i,
                                       etas, theta_o, dllk)
    # Calculate new likelihood gradient
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    # and new prior
    dlpr = -sigma_o_i*(theta_max - theta_o)
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
        theta_max, fs = pseudo_line_search2(theta_max, X_t, s, fs, dlpo,
                                            sigma_o_i, etas, theta_o, dllk)
        # Calculate the new gradient and conditional rates
        dllk, etas = pseudo_dllk(theta_max, X_t, fs)

        # Calculate prior
        dlpr = -sigma_o_i*(theta_max - theta_o)
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
    eta = compute_eta[param_est_eta](theta_max, N)
    ddllk = -R*bethe_approximation.construct_fisher_diag(eta, N)
    #ddllk = pseudo_ddllk(etas,D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ### ddlpo_i = 1./ddlpo#numpy.linalg.inv(ddlpo)
    ddlpo_i = 1./ddlpo
    # Return fitted theta and Fisher Info matrix
    return theta_max, -ddlpo_i


def pseudo_bfgs(y_t, X_t, R, theta_0, theta_o, sigma_o, sigma_o_i,
                param_est_eta='bethe_hybrid'):
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
    # Calculate fs = sum(theta_I*F_I(x_s = 1, x_/s)) via dense reconstruction
    fs = _fs_from_theta_dense(theta_max, X_t)

    # Initialize the estimate of the inverse fisher info
    ddlpo_i_e = numpy.identity(theta_max.shape[0])
    # Initialize stopping criterion variables
    max_dlpo = 1.
    iterations = 0
    # Compute derivative of posterior
    dllk, etas = pseudo_dllk(theta_max, X_t, fs)
    dlpr = -sigma_o_i*(theta_max - theta_o)
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
                                           sigma_o_i, etas, theta_o, dllk)
        # Get the difference between old and new theta
        d_theta = theta_max - theta_prev
        # Compute derivative of posterior
        dllk, etas = pseudo_dllk(theta_max, X_t, fs)
        dlpr = -sigma_o_i*(theta_max - theta_o)
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
    eta = compute_eta[param_est_eta](theta_max, N)
    ddllk = -R*bethe_approximation.construct_fisher_diag(eta, N)
    #ddllk = pseudo_ddllk(etas,D)
    ddlpo = ddllk - sigma_o_i
    # Calculate Inverse
    ### ddlpo_i = 1./ddlpo#numpy.linalg.inv(ddlpo)
    ddlpo_i = 1./ddlpo
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
    # Project all Fx_s on search direction at once via precomputed CSC transpose
    Fx_s_s_flat = Fx_s_stacked_T[time_bin].dot(s)  # (R*N,)
    Fx_s_s = Fx_s_s_flat.reshape((R, N), order='F')  # (R, N)
    # Project posterior on search direction
    dlpo_s = numpy.dot(dlpo.T, s)
    # Project conditional rate on search direction
    detas = etas*(1-etas)
    # Project one-step covariance matrix on search direction
    sigma_o_i_s = numpy.dot(s, numpy.dot(sigma_o_i, s))
    # Compute projection of pseudo-log-likelihood Hessian on search direction
    ddlpo_s = numpy.sum(detas * Fx_s_s * Fx_s_s) + sigma_o_i_s
    # Compute how much the step should be along search direction
    alpha = dlpo_s/ddlpo_s
    # Update sum of active thetas
    fs_new = fs + alpha*Fx_s_s
    # Update theta
    theta_new = theta + alpha*s
    # Return
    return theta_new, fs_new


def pseudo_line_search2(theta, X, s, fs, dlpo, sigma_o_i_tmp, etas, theta_o,
                        dllk):
    """ Performs the line search for pseudo-log-likelihood as objective
    function by quadratic approximation at current theta, but does more than one
    step.

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
    :param numpy.ndarray dllk:
        (d,) log-likelihood gradient at (theta, fs); reused inside the
        quadratic line search rather than recomputed by ``pseudo_dllk``,
        saving one csr_matvec per call.

    :returns:
        (d,) new theta according to quadratic approximation
        (r, c) new sums of active thetas
    """
    # Extract number of runs and cells
    R, N = X.shape
    # sigma_o_i_tmp is 1D diagonal — avoid constructing full (D,D) matrix
    # Precompute sigma_o_i projected on search direction: sum(diag * s^2)
    sigma_o_i_s = numpy.dot(sigma_o_i_tmp, s * s)
    # Project all Fx_s on search direction via dense reconstruction
    # (~1.4x faster than the equivalent CSC matvec for N=60).
    Fx_s_s = _fs_from_theta_dense(s, X)  # (R, N)
    # Precompute Fx_s_s squared for Hessian projection
    Fx_s_s_sq = Fx_s_s * Fx_s_s  # (R, N)
    # Project posterior on search direction
    dlpo_s = numpy.dot(dlpo.T, s)
    detas = etas * (1 - etas)
    ddlpo_s = numpy.sum(detas * Fx_s_s_sq) + sigma_o_i_s
    # Inside the original inner loop, ``theta`` does not accumulate -- each
    # iter rebuilds theta_new = theta + 0.5*alpha*s from the SAME input
    # theta with a different alpha. Because dlpr is affine in alpha:
    #     dlpr_k       = dlpr_input - 0.5 * alpha_k * sigma_o_i * s
    #     dlpo_k       = dlpo_input - 0.5 * alpha_k * sigma_o_i * s
    #     dlpo_s_k     = dlpo_s_input - r * dlpo_s_{k-1}      (r = 0.5*sigma_o_i_s/ddlpo_s)
    # so the entire inner loop reduces to a scalar affine recurrence. The
    # per-iter D-vector ops (theta_new, dlpr, dlpo, dot(dlpo,s)) in the
    # original loop are dead weight: only the final alpha matters, so we
    # iterate in scalars and reconstruct theta_new and fs_new once at the
    # end.
    dlpo_s_input = dlpo_s
    r = 0.5 * sigma_o_i_s / ddlpo_s
    last_alpha = 0.0
    num_iter = 0
    conv = numpy.inf
    while conv > 1e-2 and num_iter < 10:
        dlpo_s_old = abs(dlpo_s)
        last_alpha = dlpo_s / ddlpo_s
        dlpo_s = dlpo_s_input - r * dlpo_s
        conv = abs(dlpo_s_old - dlpo_s)
        num_iter += 1
    half_alpha = 0.5 * last_alpha
    theta_new = theta + half_alpha * s
    fs_new = fs + half_alpha * Fx_s_s
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
    # Use cached dims so this works whether or not per-(s) Fx_s was built.
    N = _N_cells
    R = _R_trials
    fs_flat = Fx_s_stacked_T[t].dot(theta)
    fs = fs_flat.reshape((R, N), order='F')
    etas = expit(fs)
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
    # Calculate conditional rate using scipy expit (handles overflow, vectorized C)
    etas = expit(fs)
    # Dense formulation of ``Fx_s_stacked @ (X - etas).ravel('F')``:
    #   singleton block: dllk[i]      = sum_r res[r, i]                 = res.sum(0)[i]
    #   pair (i<j):      dllk[N+p]    = sum_r X[r,j]*res[r,i] + X[r,i]*res[r,j]
    #                                = (M + M.T)[i, j] where M = X.T @ res.
    # Avoids the order='F' ravel forced-copy that the sparse path requires,
    # which dominates at N=60 when fs is now produced C-contiguous.
    res = X - etas
    M = X.T.dot(res)
    dllk = numpy.empty(_N_cells + _pair_i_idx.shape[0])
    dllk[:_N_cells] = res.sum(axis=0)
    dllk[_N_cells:] = (M[_pair_i_idx, _pair_j_idx]
                       + M[_pair_j_idx, _pair_i_idx])
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
        # Compute Hessian for each cell: Fx_s[s_i] @ diag(deta) @ Fx_s[s_i].T
        Fx_si = Fx_s[time_bin][s_i]
        ddllk += Fx_si.multiply(deta).dot(Fx_si.T).toarray()
    # Return
    return ddllk


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
    # Compute all fs at once via precomputed CSC transpose
    fs_flat = Fx_s_stacked_T[t].dot(theta)
    fs = fs_flat.reshape((R, N), order='F')
    # Vectorized pseudo-log-likelihood: sum over all cells and trials
    pseudo_llk = numpy.sum(X_t * fs - numpy.log(1 + numpy.exp(fs)))
    return pseudo_llk

functions = {'nr': pseudo_newton,
             'cg': pseudo_cg,
             'bf': pseudo_bfgs}

compute_eta = {'mf': mean_field.forward_problem_hessian,
               'bethe_BP': bethe_approximation.compute_eta_BP,
               'bethe_CCCP': bethe_approximation.compute_eta_CCCP,
               'bethe_hybrid': bethe_approximation.compute_eta_hybrid}