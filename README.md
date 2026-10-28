# RC-ShapleyNet

This anonymous package is a **reference implementation of the manuscript specification** for recovery-grounded route attribution and verified mobile-charging search. It provides the physical checker, complete-route game, recovery probe, graph model, independent supervision, bounded search, episode logging, and paired inference.

The supplied examples and tests are constructed checks of these definitions. Benchmark weights and original experiment logs were not supplied with this release. The manuscript's reported results remain the authors' empirical results; running these examples does not reproduce those measurements. Benchmark execution requires actual scenario inputs, fitted checkpoints, calibration, and implemented comparison methods.

## Installation

Use Python 3.10–3.12 in an isolated environment:

```text
python -m pip install -e ".[learning,test]"
python -m pytest -q
```

The base installation (`python -m pip install -e .`) supports physics, exact small games, recovery, search, and statistics. The `learning` extra supplies graph/training dependencies, `sumo` supplies SUMO/libsumo interfaces, and `milp` supplies the Gurobi binding. Solver execution requires a usable solver installation and license. [environment.yml](environment.yml) specifies the manuscript's Python 3.10.12/CUDA 12.1 profile; retained settings are in [configs/paper.yaml](configs/paper.yaml).

The command interface uses `validate`, `audit`, `train`, `search`, `run`, `statistics`, and `correlation`. Inspect installed command arguments with `rc-shapleynet --help` and `rc-shapleynet <command> --help`. The Python APIs below expose the same scientific objects directly.

An executable small training manifest is supplied in
[examples/constructed_training.json](examples/constructed_training.json).
It runs two epochs on constructed, disjoint inputs:

```text
rc-shapleynet validate examples/figure2.json
rc-shapleynet train examples/constructed_training.json --checkpoint outputs/check.pt
rc-shapleynet search examples/figure2.json --checkpoint outputs/check.pt --max-trials 2
```

The manifest contains `training_pools` and `validation_pools`. Each pool supplies
one `scenario`, a list of complete `plans`, and optional explicitly cached
`trajectories`. `num_locations` declares the frozen vocabulary size. The `training`
mapping uses `TrainingConfig` names; paper preset aliases such as `max_epochs`
and `rank_margin` are also accepted. Larger experiments supply their actual
state pools and eight cached trajectories rather than copying a single trace.

The [paper artifacts](paper_artifacts/README.md) include the editable TikZ sources,
controlled theorem checks, five theory surfaces, public UrbanEV prior audit,
and the supplied case-record plotting inputs. Install the `analysis` extra to
run the scientific plotting scripts.

### Windows SUMO 1.22 runtime

On Windows, the `sumo` extra installs `sumolib` and `traci` 1.22.0. SUMO and its matching native binding use a standalone runtime. The official 1.22 Windows runtime wheel has an internal CPython 3.8 tag despite its generic filename; installing it as a distribution leaves incompatible metadata on Python 3.10–3.12. The helper below verifies official PyPI lengths and SHA256 values, then extracts unchanged application and ABI-matched binding payloads without registering either problematic distribution:

```text
python -m pip install -e ".[sumo]"
python scripts/setup_windows_sumo.py --runtime-dir <external-runtime-directory>
```

Apply the printed `SUMO_HOME`, `PYTHONPATH` and executable-path settings in the active shell. Check that `sumo --version` and `netconvert --version` both report 1.22.0. The helper preserves upstream bytes and records source hashes; it does not alter package metadata. A matching official standalone Windows release can also supply the runtime. Non-Windows installations retain the four pinned 1.22.0 SUMO packages. The runtime and downloaded wheels are external dependencies, not repository contents.

## Constructed mechanism check

[examples/figure2.json](examples/figure2.json) represents the two-route example: A has seven available units, B has four, and one EV receives four from each in separate intervals. The JSON supplies the physical slot duration and uses kWh/kW for these schematic amounts.

```python
import json
from pathlib import Path
from rc_shapleynet.schema import Task, plan_from_dict, scenario_from_dict
from rc_shapleynet.physics import evaluate
from rc_shapleynet.game import FixedRouteGame, mobius_dividends
from rc_shapleynet.recovery import probe

payload = json.loads(Path("examples/figure2.json").read_text(encoding="utf-8"))
q = scenario_from_dict(payload["scenario"])
plan = plan_from_dict(payload["plan"])
t = payload["task"]
task = Task(t["id"], tuple(t["route_ids"]), tuple(t["recipient_ids"]),
            t.get("kind", "service"))
checked = evaluate(q, plan)
assert checked.feasible, checked.violations
dividends = mobius_dividends(FixedRouteGame(q, plan).enumerate())
for player in task.route_ids:
    result = probe(q, plan, task, player)
    print(player, result.committed_total, result.restored_total, result.difficulty)
```

Removing A leaves no spare service: difficulty is 1. Removing B permits A to restore three of four units: difficulty is 0.25. [examples/wrsn_small.json](examples/wrsn_small.json) is a separate sensor microfixture with assumed loads and transfer calibration, distinct from the manuscript's reported 20-node case.

## Units and metrics

`Scenario` freezes the planning reference `q`; `Plan` supplies complete routes. Routes contain ordered travel/wait arcs, received-energy services, and transmitter/depot energy entries.

| Quantity | Unit or convention |
| --- | --- |
| Energy, including `Service.energy` and `Power` fields | kWh per event or physical slot |
| Charger, auxiliary, and depot power | kW; multiply by `step_seconds / 3600` for slot energy |
| Coordinates and road distance | Metres |
| Arc, service, availability, and load indices | Integer physical slots; intervals are `[start, end)` |
| Physical observation time | `horizon * step_seconds`, distinct from search wall-clock seconds |
| WRSN outage | Accumulated node-hours |
| Completion and availability | Fractions; 0.01 is one percentage point |

Convert external traces explicitly: joules to kWh by division by 3,600,000; watts to kW by division by 1,000. `Power.output_energy` is battery-side transmitted energy; `Service.energy` is receiver-side energy. Broadcast transmitter consumption is charged once, with the sum of received energies checked. A recipient can receive successive sources, with at most one effective source per slot.

The WRSN objective uses `C_alive`, the fraction of powered node-time. `C_cov` evaluates spatial sensing coverage using supplied footprints and quadrature points. `C_conn` averages availability over specified communication paths. A relay can affect path availability beyond its sensing footprint. A zero-energy node with zero load remains unpowered.

## Fixed-reference attribution

`FixedRouteGame(q, plan)` evaluates `J(P|S;q) - J(Pempty;q)`. Restriction preserves retained schedules, original requests, and unavoidable travel committed in `q`; it does not reoptimize retained routes. Full, empty, and queried restricted plans must be feasible. This deletion-closure requirement also applies after rolling-reference updates: a feasible full plan alone does not establish a valid empty baseline.

`probe` measures service recoverable before original deadlines under a fixed local action set. It uses retained dwell locations and spare resources, preserves other commitments, and caps restoration separately for each recipient. Its difficulty `kappa` measures service recovery, distinct from recovery of nonlinear completion or connectivity utility.

`RCShapleyNet.forward_plan(graph)` returns an `EncodedPlan`. Its signed coefficients, recovery predictions, urgencies, and shares stay fixed while `values(masks)` queries that plan. `allocation()` preserves the learned sparse game's total. Physical-game accuracy depends on coefficient, recovery-score, and omitted-support errors.

Training uses separate physical coalition and recovery targets. Fit the location vocabulary and normalization on training inputs, reuse them on held-out graphs, and keep scenario references disjoint across splits. The labeling pipeline evaluates actual supplied cached trajectories; it does not duplicate one trace to emulate an ensemble.

For a same-architecture standard-Shapley comparison, use the same checkpoint and change `lambda_` to zero:

```python
from rc_shapleynet.search import LearnedGuide
from rc_shapleynet.training import load_checkpoint

model, normalizer, metadata = load_checkpoint("fitted_checkpoint.pt")
rc_guide = LearnedGuide(model, normalizer=normalizer, lambda_=0.5,
                        tau=0.5, eta=0.5)
standard_guide = LearnedGuide(model, normalizer=normalizer, lambda_=0.0,
                              tau=0.5, eta=0.5)
```

Keep initialization, support, neighborhood construction, reconstruction, acceptance, physical checks, exploration, and budgets common. Changing the neighborhood measures a different comparison; the manuscript separates these factors. Standard-Shapley fidelity uses its own target, while RC fidelity uses the recovery-conditioned target.

## Search and benchmark inputs

`search` maintains separate current and best verified plans. Reconstruction proposes routes; the original physical checker and objective govern incumbent replacement. The online budget includes initialization, graph construction, guidance, reconstruction checks, objective evaluation, and final attribution. A physical guide must pass every physical call through the shared budget callback. Offline recovery labels belong to training cost; online recovery probes belong to the online budget.

Execution commits the first interval and advances the reference before replanning. Physical time, simulation seeds, model-training repetitions, and search RNG seeds retain distinct roles and identifiers.

`experiments.load_adapter("module:factory", configuration)` loads an actual planner factory returning a callable from `Scenario` to `Plan`. Supply real adapters for external methods and preserve their physical assumptions. Method names do not select placeholder scores. Matched episode logging uses common frozen scenarios and simulation seeds; query-controlled and wall-clock profiles retain separate labels and inference families.

Retain actual input manifests, calibration, split IDs, checkpoints, configuration, runtime versions, and episode rows with a benchmark execution. Use those rows for inference. See [docs/EQUATION_MAP.md](docs/EQUATION_MAP.md), [docs/IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md), and [docs/STATISTICS.md](docs/STATISTICS.md).
