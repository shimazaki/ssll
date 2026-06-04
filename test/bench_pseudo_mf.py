"""Baseline benchmark for the pseudo+mf EM path on N=20/40/60.

This is the fixture against which any JAX/GPU port of the approximate
fitting path should be measured. The CLAUDE.md workflow rule requires
recording baselines BEFORE replacing code.

Run from the ssll/ directory:
    python test/bench_pseudo_mf.py [--n 20 40 60] [--T 50] [--R 100]

Writes results to test/bench_pseudo_mf.txt (append mode, with timestamp,
host, and the relevant numpy/JAX versions). The timing is wall-clock for
the full ssll.run call; we report median of three runs to suppress
noise from one-off GC / page-fault costs.
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


def _make_spikes(N, T, R, seed):
    rng = numpy.random.default_rng(seed)
    # crude correlated-bernoulli surrogate: shared latent + per-cell noise
    z = rng.normal(size=(T, R, 1))
    eps = rng.normal(size=(T, R, N))
    h = 0.4 * z + 0.6 * eps - 1.0
    p = 1.0 / (1.0 + numpy.exp(-h))
    return (rng.random((T, R, N)) < p).astype(numpy.int32)


def _time_run(spikes, repeats=3):
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ssll.run(
            spikes, order=2, window=1,
            param_est='pseudo', param_est_eta='mf',
            max_iter=20,
        )
        timings.append(time.perf_counter() - t0)
    return timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, nargs='+', default=[20, 40, 60])
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--R', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out', default=os.path.join(_THIS, 'bench_pseudo_mf.txt'))
    args = p.parse_args()

    header_lines = [
        f"# bench_pseudo_mf @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# host={platform.node()} python={platform.python_version()} "
        f"numpy={numpy.__version__}",
        f"# T={args.T} R={args.R} repeats={args.repeats} "
        f"order=2 param_est=pseudo param_est_eta=mf max_iter=20",
        f"# columns: N  median_s  min_s  max_s",
    ]
    print('\n'.join(header_lines))

    rows = []
    for N in args.n:
        spikes = _make_spikes(N, args.T, args.R, args.seed)
        ts = _time_run(spikes, repeats=args.repeats)
        ts_sorted = sorted(ts)
        med = ts_sorted[len(ts_sorted) // 2]
        row = f"{N:>3d}  {med:7.3f}  {min(ts):7.3f}  {max(ts):7.3f}"
        rows.append(row)
        print(row, flush=True)

    with open(args.out, 'a') as f:
        f.write('\n'.join(header_lines) + '\n')
        f.write('\n'.join(rows) + '\n\n')
    print(f"# wrote {args.out}")


if __name__ == '__main__':
    main()
