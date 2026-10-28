# Release validation

The release was checked on Windows with Python 3.12.14, CPU PyTorch 2.3.1,
PyTorch Geometric 2.5.3, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, and
statsmodels 0.14.2. The manuscript environment remains specified separately as
Python 3.10.12 and CUDA 12.1 in `environment.yml`.

The full test suite passed **152 tests with zero skips**. This included actual
SUMO 1.22.0 stepping on a small road network, repeated mid-travel replanning,
physical energy/port/safe-return checks, recovery callbacks, graph input
distinguishability, physical coalition labels, training and checkpoint reload,
verified search, matched episode logging, and scalar/correlation inference.
Both `sumo` and `netconvert` reported version 1.22.0. The dependency check
reported no broken requirements.

The CLI also completed a constructed four-plan labeling/training run with the
128-dimensional, two-layer, order-two model, then searched a physically checked
example from its saved checkpoint. Separate wheel installation was tested from
outside the repository, including packaged statistical configuration access.
These are implementation checks, rather than estimates of benchmark performance.

Independent review found and closed issues in per-path connectivity, immutable
arrival features, plan power/timing marks, shared recovery query accounting,
metric aliases, seed/checkpoint matching, and installed configuration lookup.
Counterexamples are covered by executable tests. The six SciPy warnings arise
from deliberately tiny statistical fixtures; the manuscript protocol uses 100
scenario units after averaging the five training repetitions.

On Windows, use the documented standalone SUMO setup. The official 1.22 Windows
runtime wheel has inconsistent internal platform metadata; the setup helper
preserves its verified application bytes without registering that distribution.
Linux uses the pinned native optional dependencies.
