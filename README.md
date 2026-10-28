# RC-ShapleyNet — Official Code

Recovery-grounded Shapley attribution and verified search for mobile charging.

## Installation

Use Python 3.10–3.12. The paper environment is listed in [environment.yml](environment.yml).

```bash
python -m pip install -e ".[learning,sumo]"
```

For Windows SUMO 1.22.0, run the helper and apply its printed environment settings:

```bash
python scripts/setup_windows_sumo.py --runtime-dir runtime/sumo-1.22.0
```

## Execution

Model and experiment settings are in [configs/paper.yaml](configs/paper.yaml). Comparison families are in [configs/statistics.yaml](configs/statistics.yaml). Prepare training, planning and experiment inputs using the types in [schema.py](src/rc_shapleynet/schema.py) and the readers in [data.py](src/rc_shapleynet/data.py).

```bash
# Train
rc-shapleynet train data/training.json --checkpoint models/rc.pt

# Plan
rc-shapleynet validate data/planning.json
rc-shapleynet search data/planning.json --checkpoint models/rc.pt --out outputs/plan.json

# Evaluate
rc-shapleynet run data/experiment.json --out outputs/episodes.csv
rc-shapleynet statistics outputs/episodes.csv --family E01_std --out outputs/statistics.json
```

Add `--standard` to the search command for standard Shapley sharing. Run `rc-shapleynet <command> --help` for input and execution options.
