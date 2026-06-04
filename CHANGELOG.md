# Changelog

## Speed optimizations — pseudo + mf path (cg/bf), 2026-06-05

Cumulative paired benchmark on carnot (N=60, T=50, R=100, order=2,
`param_est=pseudo`, `param_est_eta=mf`, `max_iter=20`):
end-to-end pipeline went from **~10.5s** (pre-optimization baseline) to
**~1.13s** (~89% reduction); ~58% reduction over the paired-opt era
starting at 2.66s.

Per-opt paired master→opt medians live in `test/bench_pseudo_mf.txt`.
Full commit history with the same deltas is in `git log --oneline master`.

| Opt branch | Mechanism | Δ median (paired, N=60) |
|---|---|---|
| `opt/precompute-fx-stacked-T` | precompute CSC transpose of stacked Fx_s | −13.5% |
| `opt/fast-hstack-csr` | custom CSR hstack, avoid scipy overhead | ~−7% |
| `opt/tap-linearity` | reuse `θ²ᵀ@η_var` across TAP NR iterations | −5% |
| `opt/hoist-dllk-from-ls2` | pass `dllk` into line search, skip recompute | −14% |
| `opt/direct-stacked-o2` | build stacked CSR directly; skip per-s + hstack | −17.6% |
| `opt/ls2-scalar-recurrence` | collapse line-search inner loop to scalar recurrence | −7% |
| `opt/dense-dllk` | dense BLAS gemm replaces sparse Fx_s_stacked matvecs | −12.7% |
| `opt/skip-fx-stacked-cg` | skip dead Fx_s_stacked build for cg/bf order=2 | −5.7% |
| `opt/flat-pair-index` | 1D flat indexing + persistent Theta2 buffer | −6.4% |
| `opt/spikes-float` | `binalize_spikes` returns float64 → BLAS gemm for `X_t @ Theta2` | −4.6% |

### Invariants preserved

- All 12 tests in `testing.py` continue to pass at their existing MLL
  tolerances (most at 1e-6 absolute on hard-coded expected values).
- Source-line **equations are literally unchanged**. Only computation
  order, data layout (dtype, dense-vs-sparse storage), dead-code
  elimination, and buffer reuse changed.

  A counter-example branch (`opt/tap-logit`, commit `7079e4d`) that
  rewrote `log(η) - log(1-η)` as `logit(η)` and `1/η + 1/(1-η)` as
  `1/η_var` passed tests but was **dropped** because the source line no
  longer matched the TAP derivation in the paper. The standing rule
  going forward: speed work must be invisible to anyone reading the
  math.

### Drop reasons (attempted but not landed)

- `opt/tap-logit` — equation rewrite (see above).
- Dense matmul rewrite of `transforms.compute_y` order=2 path —
  bit-identical and ~10x standalone, but the pipeline impact was within
  bench noise (compute_y is only ~3% of total run; the `astype(float)`
  copy and the (T,N,N) temp absorbed the win). Not committed.

### Benchmarking notes

- Bench host (carnot) is multi-tenant; paired 15-rep runs occasionally
  contain load-contaminated tails. Compare clean clusters rather than
  full ranges. Standalone microbenchmarks are useful for sizing a
  change, but the only number that matters for shipping is paired
  pipeline median back-to-back on the same host state.
- See `test/bench_pseudo_mf.py` for the paired bench harness and
  `test/bench_pseudo_mf.txt` for the historical log.
