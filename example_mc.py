"""
Minimal working example of the MC (Boltzmann-learning) inference path,
`param_est='mc'`. The exact likelihood is fitted at every time step with
Gibbs-sampled expectation parameters (persistent contrastive divergence),
so the same code scales to population sizes where the 2**N enumeration of
the exact path is infeasible, without the bias of pseudo-likelihood/TAP.

Here a small population (N=8) is used so the MC fit can be overlaid on the
exact fit for comparison; for large N simply drop the exact run. Sampler
settings can be adjusted through boltzmann_learning.MC_PARAMS before
calling ssll.run. Runtime is a few minutes (the MC path pays a sampling
cost per gradient step).

For the exact and pseudo-likelihood paths see 'example_exact.py' and
'example_approx.py'.
---

This code implements approximate inference methods for State-Space Analysis of
Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012). It is an extension of
the existing code from repository <https://github.com/tomxsharp/ssll> (For
Matlab Code refer to <http://github.com/shimazaki/dynamic_corr>). We
acknowledge Thomas Sharp for providing the code for exact inference.

Copyright (C) 2026

Author: Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)

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


# Set time bins, number of trials, and number of cells
T, R, N = 50, 100, 8
# Set the interaction order
O = 2


# ----- SPIKE SYNTHESIS -----
# Global module
import numpy
# Local modules
import synthesis
import transforms

# Create underlying time-varying theta parameters as Gaussian processes
theta = synthesis.generate_thetas(N, O, T, seed=42)

# Initialise the transforms library in preparation for computing P
transforms.initialise(N, O)
# Compute P for each time step
p = numpy.zeros((T, 2 ** N))
for i in range(T):
    p[i, :] = transforms.compute_p(theta[i, :])
# Generate spikes according to those probabilities
spikes = synthesis.generate_spikes(p, R, seed=1)


# ----- FITTING -----
# Local module
import __init__  # From outside this folder, this would be 'import ssll'

# MC (Boltzmann-learning) fit: exact likelihood, Gibbs-sampled eta
emd_mc = __init__.run(spikes, O, param_est='mc', param_est_eta='mc',
                      max_iter=10)
# Exact fit on the same data, for comparison (feasible because N is small)
emd_exact = __init__.run(spikes, O, param_est='exact', param_est_eta='exact',
                         max_iter=10)

print('Log marginal likelihood: mc = %.4f, exact = %.4f'
      % (emd_mc.mll, emd_exact.mll))


# ----- PLOTTING -----
# Global modules
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='ticks', context='notebook')

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
# First-order thetas of the first two cells; one pairwise theta
for k, c in zip([0, 1], ['C0', 'C1']):
    ax[0].plot(theta[:, k], c=c, linestyle='--', label='true' if k == 0 else None)
    ax[0].plot(emd_exact.theta_s[:, k], c=c, alpha=0.4,
               label='exact' if k == 0 else None)
    ax[0].plot(emd_mc.theta_s[:, k], c=c, linestyle=':',
               label='mc' if k == 0 else None)
ax[1].plot(theta[:, N], c='C2', linestyle='--')
ax[1].plot(emd_exact.theta_s[:, N], c='C2', alpha=0.4)
ax[1].plot(emd_mc.theta_s[:, N], c='C2', linestyle=':')

ax[0].set_title('Boltzmann-learning (MC) fit vs exact fit')
ax[0].set_ylabel('First-order theta')
ax[0].legend(frameon=False)
ax[1].set_xlabel('Time bin')
ax[1].set_ylabel('Second-order theta')
sns.despine(fig)
plt.show()
