"""Real physical labels, real training, and verified-search integration."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rc_shapleynet.game import FixedRouteGame, interaction_task
from rc_shapleynet.graph import FeatureNormalizer, build_plan_graph
from rc_shapleynet.learning import RCShapleyNet
from rc_shapleynet.physics import evaluate
from rc_shapleynet.pipeline import (
    LabelGenerationError, coalition_masks, label_plan, label_plans,
    label_reference_id, mask_stream_seed,
)
from rc_shapleynet.recovery import probe
from rc_shapleynet.schema import Arc, Charger, Location, Plan, Power, Recipient, Route, Scenario, Service
from rc_shapleynet.search import LearnedGuide, SearchConfig, search
from rc_shapleynet.training import TrainingConfig, fit, load_checkpoint


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(11)
    torch.set_num_threads(1)


def ev_pool(seed=10001):
    context = Scenario("EV", (Location("L", 0, 0),),
        (Charger("A", "L", 7, 7, 1, 1, receiver_efficiency=1),
         Charger("B", "L", 4, 4, 1, 1, receiver_efficiency=1)),
        (Recipient("R", "L", 0, 8, 8, 0, 8),), 8,
        step_seconds=3600, simulation_seed=seed, clock_offset_seconds=9*3600)
    def route(name, slots):
        return Route(name, tuple(Arc("L", "L", t, t+1) for t in range(8)),
            tuple(Service("R", t, 1) for t in slots), tuple(Power(t, 1) for t in slots))
    plans = tuple(Plan((route("A", range(a)), route("B", range(4, 4+b))))
                  for a, b in ((4, 4), (3, 4), (2, 4), (4, 3)))
    return context, plans


def sensor_example():
    context = Scenario("WRSN", (Location("L", 0, 0),),
        tuple(Charger(c, "L", 5, 5, 1, 1) for c in ("A", "B")),
        (Recipient("R", "L", 0, 4, 0, .5, 2, load=(.25,)*4),), 4,
        step_seconds=3600, clock_offset_seconds=7*3600,
        coverage_points=((0, 0, 1),))
    def route(name, slot):
        return Route(name, tuple(Arc("L", "L", t, t+1) for t in range(4)),
                     (Service("R", slot, .4),), (Power(slot, .5),))
    plan = Plan((route("A", 0), route("B", 2)))
    traces = tuple(replace(context, simulation_seed=20001+i,
        recipients=(replace(context.recipients[0], load=(amount,)*4),))
        for i, amount in enumerate((.1, .4)))
    return context, plan, traces


def test_shared_mask_streams_include_empty_full_and_do_not_depend_on_plan():
    assert mask_stream_seed(10001, 7) == 10001007
    first = coalition_masks(4, 16, 10001)
    assert torch.equal(first, coalition_masks(4, 16, 10001))
    assert not torch.equal(first, coalition_masks(4, 16, 10002))
    assert first[0].sum() == 0 and first[1].sum() == 4
    rng = np.random.Generator(np.random.PCG64(10001002))
    count = int(rng.integers(1, 4))
    expected = torch.zeros(4)
    expected[rng.choice(4, count, replace=False)] = 1
    assert torch.equal(first[2], expected)
    with pytest.raises(ValueError):
        coalition_masks(4, 1, 1)


def test_offline_labels_are_the_real_fixed_complete_route_game():
    context, plans = ev_pool()
    item = label_plan(context, plans[0], mask_count=8)
    game = FixedRouteGame(context, plans[0])
    for mask, target in zip(item.masks, item.coalition_targets):
        members = [name for name, present in zip(item.graph.route_ids, mask) if present]
        assert float(target) == pytest.approx(game(members), abs=1e-6)
    assert item.coalition_targets[0] == 0
    assert item.physical_full_value == pytest.approx(evaluate(context, plans[0]).objective)
    assert item.recovery_targets[("A", "B"), "A"] == pytest.approx(1, abs=1e-6)
    assert item.recovery_targets[("A", "B"), "B"] == pytest.approx(.25, abs=1e-6)
    pool = label_plans(context, plans, mask_count=8)
    assert len({example.reference_id for example in pool}) == 1
    assert all(torch.equal(example.masks, item.masks) for example in pool)
    assert label_reference_id(context) == label_reference_id(context, (context,))


def test_explicit_actual_ensemble_is_averaged_without_feature_leakage():
    context, plan, traces = sensor_example()
    averaged = label_plan(context, plan, mask_count=8, trajectories=traces)
    singles = [label_plan(context, plan, mask_count=8, trajectories=(trace,)) for trace in traces]
    assert torch.allclose(averaged.coalition_targets,
                          torch.stack([v.coalition_targets for v in singles]).mean(0))
    assert averaged.physical_full_value == pytest.approx(
        sum(evaluate(trace, plan).objective for trace in traces)/2)
    original_graph = build_plan_graph(context, plan)
    for kind in original_graph.node_features:
        assert torch.equal(averaged.graph.node_features[kind], original_graph.node_features[kind])
    load_relation = ("recipient", "forecast_load", "time")
    assert torch.equal(averaged.graph.edge_features[load_relation],
                       original_graph.edge_features[load_relation])
    assert not torch.equal(averaged.graph.node_features["recipient"],
                           build_plan_graph(traces[0], plan).node_features["recipient"])
    task = interaction_task(context, plan, ("A", "B"))
    results = [probe(trace, plan, task, "A") for trace in traces]
    committed = results[0].committed_total
    expected = 1-sum(r.restored_total for r in results)/len(results)/committed
    assert averaged.recovery_targets[("A", "B"), "A"] == pytest.approx(expected)
    assert label_reference_id(context, traces) == label_reference_id(context, traces[::-1])
    assert averaged.reference_id != singles[0].reference_id


def test_eight_cached_actual_traces_are_eight_explicit_physical_inputs():
    context, plan, _ = sensor_example()
    traces = tuple(replace(context, simulation_seed=30000+i,
        recipients=(replace(context.recipients[0], load=(.1+.04*i,)*4),))
        for i in range(8))
    item = label_plan(context, plan, mask_count=4, trajectories=traces)
    expected = []
    for trace in traces:
        game = FixedRouteGame(trace, plan)
        expected.append([game([cid for cid, present in zip(item.graph.route_ids, mask) if present])
                         for mask in item.masks])
    np.testing.assert_allclose(item.coalition_targets.numpy(), np.mean(expected, axis=0), atol=1e-7)
    with pytest.raises(LabelGenerationError, match="Duplicate cached"):
        label_plan(context, plan, trajectories=(traces[0],)*8)
    with pytest.raises(LabelGenerationError, match="cannot be empty"):
        label_plan(context, plan, trajectories=())


@pytest.mark.parametrize("field", ("recipients", "chargers", "road_slots", "clock_offset_seconds"))
def test_actual_trajectories_cannot_change_initial_requests_resources_or_travel(field):
    context, plan, traces = sensor_example()
    changes = {
        "recipients": (replace(context.recipients[0], initial_energy=.6),),
        "chargers": (replace(context.chargers[0], initial_energy=4), context.chargers[1]),
        "road_slots": {"A|L|L": 1},
        "clock_offset_seconds": context.clock_offset_seconds+1,
    }
    bad = replace(traces[0], **{field: changes[field]})
    with pytest.raises(LabelGenerationError, match=f"fixed context fields: {field}"):
        label_plan(context, plan, trajectories=(bad,))


def test_infeasible_cached_trace_reports_the_original_checker_reason():
    context = Scenario("WRSN", (Location("L", 0, 0),),
        (Charger("A", "L", 5, 5, 1, 1),),
        (Recipient("R", "L", 0, 2, 0, .9, 1, load=(.5, .5)),),
        2, step_seconds=3600)
    route = Route("A", tuple(Arc("L", "L", t, t+1) for t in range(2)),
                  (Service("R", 1, .5),), (Power(1, .625),))
    plan = Plan((route,))
    assert evaluate(context, plan).feasible
    actual = replace(context, simulation_seed=20001,
                     recipients=(replace(context.recipients[0], load=(0., 0.)),))
    with pytest.raises(LabelGenerationError, match="Trajectory 0.*receiver acceptance"):
        label_plan(context, plan, trajectories=(actual,))


def test_real_label_train_checkpoint_learned_guide_and_physical_search(tmp_path):
    train_context, plans = ev_pool(10001)
    validation_context, validation_plans = ev_pool(10002)
    train = list(label_plans(train_context, plans, mask_count=8))
    validation = list(label_plans(validation_context, validation_plans, mask_count=8))
    normalizer = FeatureNormalizer.fit([item.graph for item in train])
    train = [replace(item, graph=normalizer.transform(item.graph)) for item in train]
    validation = [replace(item, graph=normalizer.transform(item.graph)) for item in validation]
    graph = train[0].graph
    model = RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 1, hidden_dim=8)
    checkpoint = tmp_path/"checkpoint.pt"
    result = fit(model, train, validation, config=TrainingConfig(epochs=2, patience=2),
                 checkpoint=checkpoint, normalizer=normalizer)
    assert len(result.history) == 2 and result.best_epoch in (1, 2)
    loaded, frozen_normalizer, _ = load_checkpoint(checkpoint)
    guide = LearnedGuide(loaded, normalizer=frozen_normalizer)
    initial = Plan(())
    outcome = search(validation_context, initial, guide,
        config=SearchConfig(max_evaluations=120, time_limit=5, max_trials=1,
                            min_removed=2, max_removed=2))
    assert outcome.best_evaluation.feasible
    assert outcome.best_evaluation.objective >= evaluate(validation_context, initial).objective
    assert outcome.final_guidance is not None
    assert set(outcome.final_guidance.allocation) == {"A", "B"}
    assert guide.encoding_calls >= 2
