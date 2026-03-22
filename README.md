# SSLL: State-Space Analysis of Spike Correlations

A Python library for estimating time-varying natural parameters of spike-train interactions using EM-based inference in a state-space log-linear model.

## Installation

```bash
# Using conda
conda env create -f environment.yml
conda activate ssll

# Or using pip
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, matplotlib, tqdm

## Quick Start

```python
import numpy
import __init__ as ssll
import synthesis
import transforms

# Generate synthetic data (3 neurons, pairwise interactions)
N, O, T, R = 3, 2, 50, 200
transforms.initialise(N, O)
theta = synthesis.generate_thetas(N, O, T, seed=42)
p = numpy.array([transforms.compute_p(theta[t]) for t in range(T)])
spikes = synthesis.generate_spikes(p, R, seed=42)

# Run EM inference
emd = ssll.run(spikes, order=2, window=1, param_est='exact', param_est_eta='exact')
```

## Mathematical Background

### Log-Linear Model

Spike patterns are modelled by an exponential-family distribution over binary vectors **x** = (x_1, ..., x_N):

```
log p(x | theta) = theta^T F(x) - psi(theta)
```

where **theta** are the natural parameters, **F(x)** are sufficient statistics (individual spikes and their coincidences up to order O), and psi(theta) is the log partition function. The expectation parameters **eta** = E[F(x)] form the dual coordinates.

### State-Space Formulation

The natural parameters evolve over time as a linear dynamical system:

```
theta_t = F * theta_{t-1} + xi_t,    xi_t ~ N(0, Q)
```

The EM algorithm alternates between:

- **E-step:** Recursive Bayesian filter (forward) and smoother (backward) with Laplace approximation at each timestep. The MAP estimate is found via Newton-Raphson (`nr`), conjugate gradient (`cg`, default), or BFGS (`bf`).
- **M-step:** Optimize the noise covariance Q and (optionally) the autoregressive parameter F.

### Approximation Methods

For large N where exact 2^N computation is infeasible:

- **Pseudolikelihood** (`param_est='pseudo'`): Replaces the full likelihood with a product of conditional likelihoods. Scales to N ~ 60 neurons.
- **TAP mean-field** (`param_est_eta='mf'`): Second-order mean-field (Thouless-Anderson-Palmer) approximation for expectation parameters and the log partition function.
- **Bethe approximation**: Based on the Bethe free energy, solved via:
  - Belief propagation (`'bethe_BP'`): Iterative message passing.
  - CCCP (`'bethe_CCCP'`): Concave-convex procedure, guaranteed convergence.
  - Hybrid (`'bethe_hybrid'`): Tries BP first, falls back to CCCP.

**When to use which:** Use exact methods for N <= 12. For larger networks, use `pseudo` + `mf` for speed, or `pseudo` + `bethe_hybrid` for better accuracy.

## API Reference

### `ssll.run(spikes, **kwargs)`

Main entry point. Returns an `EMData` container with smoothed posterior estimates.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `spikes` | ndarray (T, R, N) | required | Binary spike data |
| `order` | int | 2 | Interaction order (1=rates, 2=pairwise, 3=triplet) |
| `window` | int | 1 | Bin width in ms |
| `map_function` | str | `'cg'` | MAP optimizer: `'nr'`, `'cg'`, `'bf'` |
| `param_est` | str | `'exact'` | `'exact'` or `'pseudo'` |
| `param_est_eta` | str | `'exact'` | `'exact'`, `'mf'`, `'bethe_BP'`, `'bethe_CCCP'`, `'bethe_hybrid'` |
| `state_cov` | float/array | 0.01 | Noise covariance Q (scalar, vector, or matrix) |
| `state_ar` | ndarray | None | Autoregressive parameter F (DxD); None = identity |
| `max_iter` | int | 100 | Maximum EM iterations |
| `theta_o` | float/array | 0 | Prior mean |
| `sigma_o` | float | 0.1 | Prior covariance scaling |
| `mstep` | bool | True | Whether to run M-step |

**Returns:** `container.EMData` object with fields `theta_s` (smoothed means), `sigma_s` (smoothed covariances), `eta_s` (expectation parameters), `mllk` (log marginal likelihood), etc.

## Examples

```bash
python example_exact.py    # Exact inference (3 neurons, 2nd-order)
python example_approx.py   # Approximate inference (20 neurons, pseudo-likelihood + TAP/Bethe)
```

## References

1. Shimazaki H, Amari S, Brown EN, Gruen S (2012). State-space analysis of time-varying higher-order spike correlation for multiple neural spike train data. *PLoS Computational Biology*, 8(3): e1002385.

2. Donner C, Obermayer K, Shimazaki H (2017). Approximate inference for time-varying interactions and macroscopic dynamics of neural populations. *PLoS Computational Biology*, 13(1): e1005309.

## License

GPL-3.0. See [LICENSE](http://www.gnu.org/licenses/gpl-3.0.html).

## Authors

- Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)
- Christian Donner (christian.donner@bccn-berlin.de)
- Thomas Sharp (original exact inference code)
- Jimmy Gaudreault (prior specification extensions)
- Magalie Tatischeff (autoregressive parameter extensions)
