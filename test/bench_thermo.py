"""Baseline benchmark for analysis-time thermodynamics on a fitted emd.

Measures:
- get_heat_capacity_beta sweep (num betas x T bins)
- compute_heat_capacity_b (samples x T bins, credible bounds)

These are independent of the EM speed and target separate optimization
opportunities (batching across samples / betas, vectorizing psi).

Run from ssll/:
    python test/bench_thermo.py [--N 6 12] [--T 50] [--R 100]
"""

import argparse
import os
import platform
import sys
import time

import numpy

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..'))
sys.path.insert(0, _ROOT)

import __init__ as ssll  # noqa: E402
import synthesis  # noqa: E402
import transforms  # noqa: E402
import thermodynamics  # noqa: E402


def _fit(N, T, R, seed):
    transforms.initialise(N, 2)
    theta = synthesis.generate_thetas(N, 2, T, seed=seed)
    p = numpy.array([transforms.compute_p(theta[t]) for t in range(T)])
    spikes = synthesis.generate_spikes(p, R, seed=seed)
    return ssll.run(spikes, order=2, window=1,
                    param_est='exact', param_est_eta='exact', max_iter=20)


def _time(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, nargs='+', default=[6, 12])
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--R', type=int, default=100)
    p.add_argument('--num_beta', type=int, default=50)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--method', default='auto',
                   choices=['auto', 'exact', 'approx', 'sampling'])
    p.add_argument('--n_samples', type=int, default=2000,
                   help='Gibbs samples per bin (sampling method only)')
    p.add_argument('--pre_n', type=int, default=200,
                   help='Gibbs burn-in (sampling method only)')
    p.add_argument('--out', default=os.path.join(_THIS, 'bench_thermo.txt'))
    args = p.parse_args()

    sampling_kw = {}
    if args.method == 'sampling':
        sampling_kw = dict(n_samples=args.n_samples, pre_n=args.pre_n, seed=0)

    header = [
        f"# bench_thermo @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# host={platform.node()} python={platform.python_version()} "
        f"numpy={numpy.__version__}",
        f"# T={args.T} R={args.R} num_beta={args.num_beta} "
        f"samples={args.samples} repeats={args.repeats} method={args.method}"
        + (f" n_samples={args.n_samples} pre_n={args.pre_n}"
           if args.method == 'sampling' else ''),
        f"# columns: N  fn  median_s  min_s  max_s",
    ]
    print('\n'.join(header))

    rows = []
    for N in args.N:
        emd = _fit(N, args.T, args.R, args.seed)

        ts = _time(lambda: thermodynamics.get_heat_capacity_beta(
            emd, num=args.num_beta, span=[0.25, 2.0], method=args.method,
            **sampling_kw),
            repeats=args.repeats)
        med = sorted(ts)[len(ts) // 2]
        row = f"{N:>3d}  get_heat_capacity_beta  {med:7.3f}  {min(ts):7.3f}  {max(ts):7.3f}"
        rows.append(row)
        print(row, flush=True)

        ts = _time(lambda: thermodynamics.compute_heat_capacity_b(
            emd, samples=args.samples, threshold=95, beta=1.0,
            method=args.method, **sampling_kw),
            repeats=args.repeats)
        med = sorted(ts)[len(ts) // 2]
        row = f"{N:>3d}  compute_heat_capacity_b  {med:7.3f}  {min(ts):7.3f}  {max(ts):7.3f}"
        rows.append(row)
        print(row, flush=True)

    with open(args.out, 'a') as f:
        f.write('\n'.join(header) + '\n')
        f.write('\n'.join(rows) + '\n\n')
    print(f"# wrote {args.out}")


if __name__ == '__main__':
    main()
