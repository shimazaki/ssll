# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSLL (State-Space analysis of spike correlations) implements EM-based approximate inference for estimating time-varying natural parameters of spike-train interactions. Based on Shimazaki et al., PLoS Computational Biology 2012, extended with approximation methods (arXiv:1607.08840).

## Workflow Rules

- **Always run tests before pushing.** Run `python -m unittest testing -v` and confirm all tests pass before any `git push`.

## Running Tests

```bash
# All tests (from project root)
python -m unittest testing -v

# Single test
python -m unittest testing.TestEstimator.test_0_spike_generation
```

Tests use `unittest` (not pytest). The 10 tests are numbered 0–9 and validate against expected KL divergence thresholds and log marginal likelihood values.

## Running Examples

```bash
python example_exact.py    # Exact inference (3 neurons, 2nd-order)
python example_approx.py   # Approximate inference (20 neurons, pseudo-likelihood + TAP/Bethe)
```

## Dependencies

numpy, scipy, matplotlib, tqdm. Conda env: `ssll`. Install: `conda env create -f environment.yml` or `pip install -r requirements.txt`.

## Architecture

This is a flat research library (no `setup.py`), imported directly:
```python
import __init__ as ssll
emd = ssll.run(spikes, order=2, window=1, param_est='exact', param_est_eta='exact')
```

### Data Flow

The `run()` function in `__init__.py` is the main entry point. It initializes an `EMData` container and iterates E-steps and M-steps until convergence:

1. **`transforms.py`** — Converts spike matrices to binary patterns and sufficient statistics. Pre-computes global sparse matrices (`p_map`, `eta_map`) for theta↔probability↔eta transformations.
2. **`container.py`** (`EMData`) — Holds all EM state: natural parameters (theta), covariances (sigma), expectation parameters (eta), hyperparameters (Q, F). Functions modify EMData in-place.
3. **`exp_max.py`** — E-step (forward filter + backward smoother) and M-step (hyperparameter optimization). E-step calls MAP estimation to find posterior mode at each timestep.
4. **MAP estimation** — Three algorithms in `max_posterior.py` (exact) and `pseudo_likelihood.py` (approximate): Newton-Raphson (`nr`), conjugate gradient (`cg`, default), BFGS (`bf`).
5. **Approximation methods** — For large N where exact 2^N computation is infeasible:
   - `mean_field.py` — TAP approximation (`param_est_eta='mf'`)
   - `bethe_approximation.py` — Belief propagation, CCCP, or hybrid (`'bethe_BP'`, `'bethe_CCCP'`, `'bethe_hybrid'`)
6. **`synthesis.py`** — Generates synthetic theta (via GP) and spikes (direct sampling or Gibbs MCMC). Gibbs sampling supports parallel execution.
7. **`probability.py`** — Log-likelihood and log-marginal-likelihood computations.

### Key Parameters

- `order`: Interaction order (1=rates, 2=pairwise, 3=triplet). Complexity scales as 2^N with exact methods.
- `param_est`: `'exact'` (full likelihood) or `'pseudo'` (pseudo-likelihood, needed for large N).
- `param_est_eta`: `'exact'`, `'mf'`, `'bethe_BP'`, `'bethe_CCCP'`, `'bethe_hybrid'`.
- `state_cov`: Noise covariance — scalar (isotropic), vector (diagonal), or matrix (full).
- `state_ar`: Autoregressive parameter matrix for state dynamics.
- `stationary`: If `True`, pools all T×R observations into a single time step (Q=0) for time-independent analysis.
- Spike data format: binary numpy array shaped `(time_bins, trials, neurons)`.

### Performance Notes

The approximate inference path (`param_est='pseudo'`, `param_est_eta='mf'`) has been optimized for large N:
- `pseudo_likelihood.py`: CSR sparse format for Fx_s matrices, precomputed stacked sparse matrices (`Fx_s_stacked`) for vectorized gradient/fs computation, single-pool parallelism for init, direct Fx_s_t diff computation (skips subsets not containing neuron s), precomputed subset membership lookup.
- `mean_field.py`: Precomputed `theta2_sq` reused throughout TAP solver.
- `exp_max.py`: Identity matrix allocated once outside M-step loop.
