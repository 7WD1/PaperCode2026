"""Execution-contract checks using actual episode physics and explicit adapters."""
import csv
import importlib
import json
import math
from pathlib import Path

import pytest
import torch

from rc_shapleynet.experiments import canonical_metrics, learned_planner, load_adapter, run_manifest
from rc_shapleynet.graph import build_plan_graph
from rc_shapleynet.learning import RCShapleyNet
from rc_shapleynet.physics import evaluate
from rc_shapleynet.schema import Charger, Location, Plan, Recipient, Scenario, to_dict


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(13)
    torch.set_num_threads(1)


def explicit_idle_adapter(tmp_path, monkeypatch):
    """A named idle control runs through real physics, not a baseline surrogate."""
    name = "explicit_idle_control_test"
    source = tmp_path/f"{name}.py"
    source.write_text('''from rc_shapleynet.schema import Plan
CONFIGURATIONS = []
CALLS = []
def factory(configuration):
    CONFIGURATIONS.append((configuration['training_seed'],
                           configuration['simulation_seed'],
                           configuration.get('checkpoint'),
                           configuration['nested'].pop('constant')))
    calls = 0
    def planner(scenario):
        nonlocal calls
        calls += 1
        CALLS.append(scenario.fingerprint())
        if configuration.get('mutate_second') and calls == 2:
            object.__setattr__(scenario, 'clock_offset_seconds', 333.)
        if configuration.get('wrong_type'):
            return 'not a Plan'
        return Plan(())
    return planner
''', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys
    sys.modules.pop(name, None)
    return name, importlib.import_module(name)


def write_manifest(tmp_path, adapter_name, **overrides):
    folder = tmp_path/"campaign"
    (folder/"scenarios").mkdir(parents=True, exist_ok=True)
    scenario = Scenario("WRSN", (Location("L", 0, 0),),
        (Charger("C", "L", 3, 3, 1, 1),),
        (Recipient("R", "L", 0, 3, 0, 1, 1, load=(.1,)*3),), 3,
        coverage_points=((0, 0, 1),), simulation_seed=10001, scenario_id="scenario-1")
    source = folder/"scenarios"/"one.json"
    source.write_text(json.dumps({"scenario": to_dict(scenario)}), encoding="utf-8")
    manifest = {"experiment_id": "constructed-contract-check", "training_seeds": [1, 2],
        "replan_seconds": 1,
        "scenarios": [{"input": "scenarios/one.json", "condition_id": "fixed"}],
        "methods": [{"name": "Explicit idle control", "adapter": adapter_name+":factory",
            "configuration": {"nested": {"constant": 7}, "checkpoint": "models/frozen.pt"}}]}
    manifest.update(overrides)
    path = folder/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, source


def test_manifest_canonicalizes_episode_aliases_once_and_preserves_inputs(tmp_path, monkeypatch):
    name, module = explicit_idle_adapter(tmp_path, monkeypatch)
    path, _, source = write_manifest(tmp_path, name)
    original = source.read_bytes()
    output = tmp_path/"observations.csv"
    summary = run_manifest(path, output)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["condition_id"], r["scenario_id"], r["simulation_seed"],
             r["training_seed"], r["method"], r["metric"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert len(rows) == summary["rows"] == 22
    metrics = {r["metric"] for r in rows}
    assert {"coverage", "connectivity", "powered_node_time", "objective"} <= metrics
    assert not {"C_cov", "C_conn", "C_alive", "spatial_coverage"} & metrics
    assert source.read_bytes() == original
    assert len(module.CALLS) == 6
    for seed in ("1", "2"):
        endpoints={r["metric"]:float(r["value"]) for r in rows if r["training_seed"]==seed}
        assert endpoints["decision_count"] == 3
        assert all(math.isfinite(endpoints[k]) for k in
                   ("decision_latency_p50_seconds", "decision_latency_p95_seconds"))
        assert endpoints["decision_latency_p95_seconds"] >= endpoints["decision_latency_p50_seconds"] >= 0
    expected = str((path.parent/"models"/"frozen.pt").resolve())
    assert module.CONFIGURATIONS == [(1, 10001, expected, 7), (2, 10001, expected, 7)]


def test_existing_output_rejects_before_any_episode_is_rerun(tmp_path, monkeypatch):
    name, module = explicit_idle_adapter(tmp_path, monkeypatch)
    path, _, _ = write_manifest(tmp_path, name)
    output = tmp_path/"locked.csv"
    output.write_text("locked records", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_manifest(path, output)
    assert not module.CONFIGURATIONS and not module.CALLS
    assert output.read_text() == "locked records"


@pytest.mark.parametrize("seeds", ([], [1, 1], [True], [-1]))
def test_invalid_or_duplicate_seed_repeats_are_rejected_before_execution(tmp_path, monkeypatch, seeds):
    name, module = explicit_idle_adapter(tmp_path, monkeypatch)
    path, _, _ = write_manifest(tmp_path, name, training_seeds=seeds)
    with pytest.raises(ValueError, match="training seeds"):
        run_manifest(path, tmp_path/"out.csv")
    assert not module.CALLS


def test_duplicate_method_names_cannot_duplicate_matched_observations(tmp_path, monkeypatch):
    name, module = explicit_idle_adapter(tmp_path, monkeypatch)
    path, manifest, _ = write_manifest(tmp_path, name)
    manifest["methods"] *= 2
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Method names"):
        run_manifest(path, tmp_path/"out.csv")
    assert not module.CALLS


@pytest.mark.parametrize("seconds", (.5, 1.5))
def test_nonintegral_replanning_intervals_are_not_silently_rounded(tmp_path, monkeypatch, seconds):
    name, module = explicit_idle_adapter(tmp_path, monkeypatch)
    path, _, _ = write_manifest(tmp_path, name, replan_seconds=seconds)
    with pytest.raises(ValueError, match="integer number"):
        run_manifest(path, tmp_path/"out.csv")
    assert not module.CONFIGURATIONS


def test_mutation_of_a_later_rolling_reference_is_detected(tmp_path, monkeypatch):
    name, _ = explicit_idle_adapter(tmp_path, monkeypatch)
    path, manifest, source = write_manifest(tmp_path, name)
    manifest["methods"][0]["configuration"]["mutate_second"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="shared planning reference"):
        run_manifest(path, tmp_path/"out.csv")
    assert source.read_bytes() == original
    assert not (tmp_path/"out.csv").exists()


def test_nonplan_adapter_outputs_and_conflicting_metric_aliases_are_rejected(tmp_path, monkeypatch):
    name, _ = explicit_idle_adapter(tmp_path, monkeypatch)
    path, manifest, _ = write_manifest(tmp_path, name)
    manifest["methods"][0]["configuration"]["wrong_type"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TypeError, match="complete Plan"):
        run_manifest(path, tmp_path/"out.csv")
    with pytest.raises(ValueError, match="coverage"):
        canonical_metrics({"C_cov": .8, "spatial_coverage": .7})
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_metrics({"coverage": float("nan")})
    with pytest.raises(ValueError, match="module:factory"):
        load_adapter("inferred-baseline-name")


def real_checkpoint(tmp_path):
    scenario = Scenario("EV", (Location("D", 0, 0, True), Location("X", 1, 0)),
        (Charger("C", "X", 5, 5, 1, 1),), (), 3)
    graph = build_plan_graph(scenario, Plan(()))
    model = RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 1, hidden_dim=8)
    path = tmp_path/"model.pt"
    # Real model tensors exercise the runtime contract; no service-score claim.
    torch.save({"model_configuration": model.configuration, "state_dict": model.state_dict(),
                "training_configuration": {"seed": 1}, "normalizer": None}, path)
    return scenario, path


def test_learned_adapter_uses_safe_return_fallback_and_maps_search_alias(tmp_path):
    scenario, checkpoint = real_checkpoint(tmp_path)
    assert not evaluate(scenario, Plan(())).feasible
    planner = learned_planner({"checkpoint": str(checkpoint), "training_seed": 1,
        "search": {"max_trials": 0, "max_evaluations": 5, "time_limit": 5, "exploration": .2}})
    plan = planner(scenario)
    assert evaluate(scenario, plan).feasible
    assert plan.routes[0].arcs[0].destination == "D"


def test_learned_adapter_rejects_false_training_seed_and_conflicting_search_alias(tmp_path):
    _, checkpoint = real_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="fitted checkpoint seed"):
        learned_planner({"checkpoint": str(checkpoint), "training_seed": 2})
    with pytest.raises(ValueError, match="Conflicting exploration"):
        learned_planner({"checkpoint": str(checkpoint),
                         "search": {"exploration": .2, "exploration_probability": .3}})


def test_learned_adapter_rejects_ids_outside_its_frozen_vocabulary(tmp_path):
    from dataclasses import replace
    scenario, checkpoint = real_checkpoint(tmp_path)
    invalid = replace(scenario, recipients=(Recipient("R", "X", 0, 3, 1, 0, 1, location_index=2),))
    planner = learned_planner({"checkpoint": str(checkpoint), "search": {"max_trials": 0}})
    with pytest.raises(ValueError, match="checkpoint vocabulary"):
        planner(invalid)


def test_packaged_statistics_preset_matches_source_and_works_without_repository(tmp_path, monkeypatch, capsys):
    from rc_shapleynet import statistics
    repository=Path(__file__).resolve().parents[1]
    resource=repository/"src"/"rc_shapleynet"/"resources"/"statistics.yaml"
    assert resource.read_bytes() == (repository/"configs"/"statistics.yaml").read_bytes()
    monkeypatch.setattr(statistics,"ROOT",tmp_path/"absent-repository")
    monkeypatch.chdir(tmp_path)
    assert statistics.default_config_path() == resource
    families=statistics.load_config(resource)["families"]
    statistics.main(["--list-families"])
    output=capsys.readouterr().out
    assert families and all(f["id"] in output for f in families)
