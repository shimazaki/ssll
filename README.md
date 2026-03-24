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

Spike patterns are modelled by an exponential-family distribution over binary vectors $\mathbf{x} = (x_1, \ldots, x_N)$. The log-linear (maximum entropy) model decomposes the log-probability into interaction terms up to order O:

$$\log p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_i \theta_i x_i + \sum_{i \lt j} \theta_{ij} x_i x_j + \sum_{i \lt j \lt k} \theta_{ijk} x_i x_j x_k + \cdots - \psi(\boldsymbol{\theta})$$

where $\psi(\boldsymbol{\theta})$ is the log partition function ensuring normalisation, and the sums extend up to the maximum interaction order O set by the `order` parameter.

**Example (N=3, O=2):** The model has D = 6 natural parameters — three first-order ($\theta_1, \theta_2, \theta_3$) controlling individual firing rates and three pairwise ($\theta_{12}, \theta_{13}, \theta_{23}$) controlling spike correlations:

$$\log p(\mathbf{x} \mid \boldsymbol{\theta}) = \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_{12} x_1 x_2 + \theta_{13} x_1 x_3 + \theta_{23} x_2 x_3 - \psi(\boldsymbol{\theta})$$

- **O=1:** Independent model — each neuron fires independently (no interactions).
- **O=2:** Pairwise Ising model — captures pairwise correlations (the most common choice).
- **O=3:** Adds triplet interactions for higher-order correlations beyond pairwise.

The sufficient statistics $\mathbf{F}(\mathbf{x})$ collect all monomials up to order O, and the expectation parameters $\boldsymbol{\eta} = \mathrm{E}[\mathbf{F}(\mathbf{x})]$ (firing rates, spike coincidence probabilities) form the dual coordinates.

### State-Space Formulation

The natural parameters evolve over time as a state-space model with a linear state equation and a log-linear observation model.

**State equation:**

$$\boldsymbol{\theta}_t = F \boldsymbol{\theta}_{t-1} + \boldsymbol{\xi}_t, \qquad \boldsymbol{\xi}_t \sim \mathcal{N}(\mathbf{0}, Q)$$

**Observation equation:**

$$\log p(\mathbf{y}_t \mid \boldsymbol{\theta}_t) = R\left[\mathbf{y}_t^\top \boldsymbol{\theta}_t - \psi(\boldsymbol{\theta}_t)\right]$$

where $\mathbf{y}_t$ is the vector of empirical spike-pattern frequencies at time $t$ and $R$ is the number of trials. This is the log-linear model from the previous section applied as the observation likelihood: the sufficient statistics of the observed spike patterns are compared against the model's expectation under $\boldsymbol{\theta}_t$.

**Autoregressive parameter F** is a D×D matrix controlling how the parameters at time $t{-}1$ predict the parameters at time $t$:

- **Default ($F = I$):** The identity matrix gives a random-walk model: $\boldsymbol{\theta}\_t = \boldsymbol{\theta}\_{t-1} + \boldsymbol{\xi}\_t$. Each parameter drifts freely from its previous value.
- **General F:** An autoregressive matrix that can capture mean-reverting dynamics, coupling between parameters, or other structured temporal dependencies.
- Set the initial value of F via the `state_ar` parameter. When `state_ar` is provided, F is optimised during the M-step. When `state_ar=None` (default), F stays fixed at identity.

**State noise covariance Q** controls the expected magnitude of parameter changes between timesteps. The `state_cov` parameter sets the initial value of Q in one of four forms:

- **Scalar** (e.g. `0.01`): Q = 0.01 × I — isotropic noise, all D parameters share one variance. Updated as a single scalar in the M-step. Simplest and usually sufficient.
- **Vector** (1D array, length D): Q = diag(state_cov) — each parameter has its own variance. Updated element-wise in the M-step.
- **Matrix** (D×D array): Q = state_cov — full covariance, captures correlations between parameter changes. Updated as a full matrix in the M-step. Expensive (D² parameters).
- **List of 2 values** (e.g. `[0.01, 0.001]`): Q = diag(λ1,...,λ1, λ2,...,λ2) — separate variances for first-order parameters (rates) and higher-order parameters (interactions). λ1 and λ2 are updated separately. Use this when rates and interactions evolve at different timescales.

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

### Macroscopic Network Properties

After fitting, the model provides time-resolved thermodynamic quantities that characterise the collective state of the population (Donner et al. 2017). These are computed by `energies.py` and stored in the `EMData` container:

- **Log partition function $\psi(\boldsymbol{\theta})$:** Normalisation constant of the log-linear model. Computed exactly for small N, via the Ogata-Tanemura estimator for large N, or via TAP/Bethe approximations. Stored in `emd.psi` (shape: T×1).

- **Entropy:** Measures the variability of population spike patterns.

$$S = -\sum_{\mathbf{x}} p(\mathbf{x}) \log p(\mathbf{x}) = \psi(\boldsymbol{\theta}) - \boldsymbol{\theta} \cdot \boldsymbol{\eta}$$

  `emd.S1` — entropy of the independent (O=1) model. `emd.S2` — entropy of the fitted model. `emd.S_ratio = (S1 - S2) / S1` — fractional entropy reduction due to interactions.

- **Internal energy:** Expected value of the energy function $E(\mathbf{x}) = -\boldsymbol{\theta}^\top \mathbf{F}(\mathbf{x})$.

$$U = \sum_{\mathbf{x}} p(\mathbf{x}) E(\mathbf{x}) = -\boldsymbol{\theta} \cdot \boldsymbol{\eta}$$

  `emd.U1` — internal energy of the independent model. `emd.U2` — internal energy of the fitted model.

- **Population spike rate:** The first-order expectation parameters `emd.eta_s[:, :N]` give the marginal firing probability of each neuron at each timestep.

- **Silence probability:** The probability that no neuron fires: $p(\mathbf{x}=\mathbf{0}) = \exp(-\psi(\boldsymbol{\theta}))$, computable from `emd.psi`.

**Note:** Heat capacity (Donner 2017, Eq. 33) is not yet implemented — it requires an augmented partition function with a temperature parameter beta.

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
| `state_cov` | float/array/list | 0.01 | Initial noise covariance Q: scalar (isotropic), 1D array (diagonal), D×D array (full), or list of 2 values [λ1, λ2] (separate rates/interactions) |
| `state_ar` | ndarray | None | Autoregressive parameter F (DxD); None = identity |
| `max_iter` | int | 100 | Maximum EM iterations |
| `theta_o` | float/array | 0 | Prior mean |
| `sigma_o` | float | 0.1 | Prior covariance scaling |
| `mstep` | bool | True | Whether to run M-step |

**Returns:** `container.EMData` object with fields:
- `theta_s` — smoothed natural parameters (T×D)
- `sigma_s` — smoothed covariances (T×D×D)
- `eta_s` — expectation parameters (T×D)
- `mllk` — log marginal likelihood history (per EM iteration)
- `psi` — log partition function (T×1)
- `S1`, `S2` — entropy of independent and fitted models (T×1)
- `S_ratio` — fractional entropy reduction (S1−S2)/S1 (T×1)
- `U1`, `U2` — internal energy of independent and fitted models (T×1)

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
