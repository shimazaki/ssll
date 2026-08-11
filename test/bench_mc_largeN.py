"""
Large-N benchmark of the MC (Boltzmann-learning) inference path.

Fits the same synthetic ground-truth data with the pseudo-likelihood + TAP
path and with param_est='mc', and reports wall time per EM iteration and
recovery accuracy against the known theta / eta. Ground-truth eta at every
time bin is estimated by long Gibbs sampling from the true theta (the
exact eta is not computable at these N).

Intended to run on a Slurm compute node (CPU only):

    sbatch submit_bench_mc.sh <N>

Usage: python test/bench_mc_largeN.py [N] [T] [R] [EM_ITERS]
Defaults: N=40, T=100, R=200, EM_ITERS=10.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy

import __init__ as ssll
import boltzmann_learning
import synthesis

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
T = int(sys.argv[2]) if len(sys.argv) > 2 else 100
R = int(sys.argv[3]) if len(sys.argv) > 3 else 200
EM_ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
O = 2

print('=== bench_mc_largeN: N=%d T=%d R=%d EM_ITERS=%d ===' % (N, T, R, EM_ITERS))
print('numpy %s' % numpy.__version__)

# ----- Ground truth and data -----
t0 = time.time()
theta_true = synthesis.generate_thetas(N, O, T, seed=42)
spikes = synthesis.generate_spikes_gibbs_parallel(theta_true, N, O, R, seed=1)
print('data generated: mean rate=%.4f  (%.0fs)'
      % (spikes.mean(), time.time() - t0))

# Ground-truth eta by long Gibbs runs from theta_true (persistent chains
# along t, so per-bin burn-in can be short).
t0 = time.time()
boltzmann_learning.initialise(N, O)
rng = numpy.random.default_rng(7)
chains = (rng.random((200, N)) < 0.2).astype(numpy.float64)
eta_true = numpy.empty_like(theta_true)
for t in range(T):
    burn = 200 if t == 0 else 20
    boltzmann_learning.gibbs_sample_eta(theta_true[t], N, chains, rng, burn,
                                        accumulate=False)
    eta_true[t] = boltzmann_learning.gibbs_sample_eta(theta_true[t], N,
                                                      chains, rng, 100)
print('ground-truth eta sampled (%.0fs)' % (time.time() - t0))


def report(name, emd, dt):
    th_err = numpy.abs(emd.theta_s - theta_true)
    eta_err = emd.eta_s - eta_true
    c_th1 = numpy.corrcoef(emd.theta_s[:, :N].ravel(),
                           theta_true[:, :N].ravel())[0, 1]
    c_th2 = numpy.corrcoef(emd.theta_s[:, N:].ravel(),
                           theta_true[:, N:].ravel())[0, 1]
    c_eta2 = numpy.corrcoef(emd.eta_s[:, N:].ravel(),
                            eta_true[:, N:].ravel())[0, 1]
    print('--- %s ---' % name)
    print('wall time: %.1fs total, %.1fs per EM iteration (%d iterations)'
          % (dt, dt / max(emd.iterations, 1), emd.iterations))
    print('mll trace: %s' % ', '.join('%.1f' % m for m in emd.mll_list))
    print('theta err  mean|1st|=%.4f  mean|2nd|=%.4f  corr1=%.4f  corr2=%.4f'
          % (th_err[:, :N].mean(), th_err[:, N:].mean(), c_th1, c_th2))
    print('eta err    rmse1=%.4f  rmse2=%.4f  corr2=%.4f'
          % (numpy.sqrt((eta_err[:, :N] ** 2).mean()),
             numpy.sqrt((eta_err[:, N:] ** 2).mean()), c_eta2))
    sys.stdout.flush()


# ----- Baseline: pseudo-likelihood + TAP -----
t0 = time.time()
emd_p = ssll.run(spikes, O, param_est='pseudo', param_est_eta='mf',
                 max_iter=EM_ITERS, EM_Info=False)
report('pseudo+mf', emd_p, time.time() - t0)

# ----- MC (Boltzmann learning) -----
t0 = time.time()
emd_m = ssll.run(spikes, O, param_est='mc', param_est_eta='mc',
                 max_iter=EM_ITERS, EM_Info=False)
report('mc', emd_m, time.time() - t0)

print('=== done ===')
