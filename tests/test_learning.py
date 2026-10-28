"""CPU checks of the manuscript's learning equations on constructed graphs."""
from dataclasses import replace
from itertools import product
import math

import pytest
import torch

from rc_shapleynet.graph import (
    FeatureNormalizer, PlanGraph, RecipientFeatureEncoder, RelationalMessageLayer,
    build_plan_graph,
)
from rc_shapleynet.learning import EncodedPlan, RCShapleyNet, interaction_support
from rc_shapleynet.schema import Arc, Charger, Location, Plan, Power, Recipient, Route, Scenario, Service, Task
from rc_shapleynet.physics import evaluate
from rc_shapleynet.training import (
    PlanTrainingExample, TrainingConfig, fit, load_checkpoint, loss_for_examples,
    make_ranking_pairs,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def make_graph(seed=10001, energy=.01):
    scenario = Scenario(
        "EV", (Location("L0", 0, 0), Location("L1", 10, 0),
               Location("L2", 10000, 0)),
        tuple(Charger(f"C{i}", f"L{i}", 10, 10, 20, 2) for i in range(3)),
        (Recipient("R0", "L0", 0, 4, 1, 0, 1, location_index=0),
         Recipient("R2", "L2", 0, 5, 1, 0, 1, location_index=2)),
        horizon=6, step_seconds=60, simulation_seed=seed,
        tasks=(Task("remote-path", ("C0", "C2"), ("R0", "R2")),),
    )
    plan = Plan((Route("C0", services=(Service("R0", 1, energy),)),
                 Route("C2", services=(Service("R2", 1, energy),))))
    return build_plan_graph(scenario, plan)


def small_model(graph):
    return RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 3,
                        hidden_dim=8, layers=2)


def example(graph, full_value):
    masks = torch.tensor(list(product((0., 1.), repeat=3)))
    targets = masks @ torch.tensor([.2, .1, .3]) + .4*masks[:, 0]*masks[:, 2]
    return PlanTrainingExample(graph, masks, targets,
        torch.full((len(interaction_support(graph)), 3), .5),
        graph.reference_id, full_value)


def test_thirteen_raw_fields_become_twenty_eight_dense_dimensions_with_gradient():
    graph = make_graph()
    assert graph.node_features["recipient"].shape == (2, 12)
    assert graph.recipient_location_ids.shape == (2,)
    assert graph.recipient_location_ids.dtype == torch.int64
    lookup = RecipientFeatureEncoder(3)
    dense = lookup(graph.node_features["recipient"], graph.recipient_location_ids)
    assert dense.shape == (2, 28)
    dense.square().sum().backward()
    assert lookup.embedding.weight.grad is not None
    assert lookup.embedding.weight.grad[[0, 2]].abs().sum() > 0
    with pytest.raises(ValueError):
        lookup(torch.ones(2, 13), graph.recipient_location_ids)


def test_relation_means_are_summed_instead_of_one_global_neighbor_mean():
    relations = (("a", "first", "b"), ("a", "second", "b"))
    graph = PlanGraph(
        {"a": torch.zeros(2, 1), "b": torch.zeros(1, 1)},
        {relations[0]: torch.tensor([[0, 1], [0, 0]]),
         relations[1]: torch.tensor([[1], [0]])},
        {relations[0]: torch.zeros(2, 1), relations[1]: torch.zeros(1, 1)},
        (), (), (), torch.empty(0, dtype=torch.long), "q",
    )
    layer = RelationalMessageLayer(("a", "b"), relations, hidden_dim=1, edge_dim=1)
    with torch.no_grad():
        for value in layer.parameters():
            value.zero_()
        for message in layer.messages:
            message[0].weight[0, 0] = 1
            message[2].weight[0, 0] = 1
    output = layer({"a": torch.tensor([[2.], [4.]]), "b": torch.zeros(1, 1)}, graph)
    assert float(output["b"][0, 0]) == pytest.approx((2+4)/2+4)


def test_wrsn_recipient_input_contains_forecast_load_and_current_sensor_energy():
    scenario = Scenario("WRSN", (Location("L", 0, 0),),
        (Charger("C", "L", 10, 10, 20, 2),),
        (Recipient("S", "L", 0, 2, 0, 1, 2, load=(.2, .3), initial_soc_proxy=.9),),
        horizon=2)
    graph = build_plan_graph(scenario, Plan(()))
    demand_fields = graph.node_features["recipient"][0, 4:6]
    assert torch.allclose(demand_fields, torch.tensor([.5, .5]))
    assert graph.node_features["recipient"].shape[1] == 12


def test_hour_phase_uses_nonmidnight_reference_and_survives_rolling_rebase():
    recipient = Recipient("R", "L", 3, 6, 1, 0, 1)
    first = Scenario("EV", (Location("L", 0, 0),),
        (Charger("C", "L", 5, 5, 1, 1),), (recipient,), horizon=8,
        step_seconds=900, clock_offset_seconds=8*3600)
    later = replace(first, recipients=(replace(recipient, arrival=1, departure=4),),
                    clock_offset_seconds=8*3600+2*900)
    original = build_plan_graph(first, Plan(()))
    rebased = build_plan_graph(later, Plan(()))
    phase = 2*math.pi*(8*3600+3*900)/86400
    expected = torch.tensor([math.sin(phase), math.cos(phase)])
    assert torch.allclose(original.node_features["recipient"][0, 2:4], expected)
    assert torch.equal(original.node_features["recipient"][0, 2:4],
                       rebased.node_features["recipient"][0, 2:4])
    phase_at_q = torch.tensor([math.sin(2*math.pi/3), math.cos(2*math.pi/3)])
    assert torch.allclose(original.node_features["time"][0, 1:3], phase_at_q)


def test_resource_only_interaction_keeps_value_support_but_uniform_recovery_shares():
    scenario = Scenario("EV", (Location("L", 0, 0),),
        tuple(Charger(name, "L", 5, 5, 1, 1) for name in ("A", "B")), (), 2,
        tasks=(Task("shared-resource", ("A", "B"), (), kind="resource"),))
    graph = build_plan_graph(scenario, Plan(()))
    model = RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 1, hidden_dim=8)
    encoded = model.forward_plan(graph)
    assert encoded.support == ((0,), (1,), (0, 1))
    assert not encoded.task_associated.any()
    assert torch.equal(encoded.shares()[-1], torch.tensor([.5, .5]))


def test_original_arrival_and_declared_stay_metadata_survive_clipped_rolling_windows():
    recipient = Recipient("R", "L", 3, 7, 1, 0, 1,
        raw_arrival_fraction=3/8, declared_stay_seconds=3600,
        arrival_clock_seconds=8*3600+3*900)
    context = Scenario("EV", (Location("L", 0, 0),),
        (Charger("C", "L", 5, 5, 1, 1),), (recipient,), 8,
        step_seconds=900, clock_offset_seconds=8*3600)
    rolled = replace(context, horizon=3, clock_offset_seconds=8*3600+5*900,
                     recipients=(replace(recipient, arrival=0, departure=2),))
    original = build_plan_graph(context, Plan(())).node_features["recipient"][0]
    after = build_plan_graph(rolled, Plan(())).node_features["recipient"][0]
    assert torch.equal(original[:4], after[:4])
    assert after[0] == 3/8 and after[1] == 60
    assert after[-1] == 30 and original[-1] == 60  # Current slack still advances.


def test_distinct_feasible_transmitter_powers_are_distinct_plan_graphs():
    scenario = Scenario("EV", (Location("L", 0, 0),),
        (Charger("C", "L", 3, 3, 2, 1, receiver_efficiency=1),),
        (Recipient("R", "L", 0, 2, .5, 0, .5),), 2, step_seconds=3600)
    arcs = tuple(Arc("L", "L", t, t+1) for t in range(2))
    first = Plan((Route("C", arcs, (Service("R", 0, .5),), (Power(0, .5),)),))
    second = Plan((replace(first.routes[0], powers=(Power(0, 1),)),))
    a, b = evaluate(scenario, first), evaluate(scenario, second)
    assert a.feasible and b.feasible
    assert a.objective != b.objective
    ga, gb = build_plan_graph(scenario, first), build_plan_graph(scenario, second)
    assert all(torch.equal(ga.node_features[k], gb.node_features[k]) for k in ga.node_features)
    relation = ("charger", "power", "time")
    assert not torch.equal(ga.edge_features[relation], gb.edge_features[relation])


def test_power_only_slot_is_an_explicit_route_event_and_pool_member():
    scenario = Scenario("EV", (Location("L", 0, 0),),
        (Charger("C", "L", 3, 3, 2, 1),), (), 10, step_seconds=3600)
    route = Route("C", tuple(Arc("L", "L", t, t+1) for t in range(10)),
                  powers=(Power(5, .1),))
    graph = build_plan_graph(scenario, Plan((route,)))
    times = graph.node_features["time"][:, 0]
    assert times.tolist() == pytest.approx([0, .5, .6, 1])
    assert set(graph.route_membership[0]["time"].tolist()) == {0, 1, 2, 3}


def test_route_owned_direction_and_timing_survive_identical_global_arc_sets():
    scenario = Scenario("EV", (Location("D", 0, 0), Location("X", 1, 0), Location("Y", 0, 1)),
        tuple(Charger(c, "D", 5, 5, 1, 1) for c in ("A", "B")), (), 4)
    clockwise = (Arc("D", "X", 0, 1), Arc("X", "Y", 1, 3), Arc("Y", "D", 3, 4))
    counter = (Arc("D", "Y", 0, 1), Arc("Y", "X", 1, 3), Arc("X", "D", 3, 4))
    first = Plan((Route("A", clockwise), Route("B", counter)))
    second = Plan((Route("A", counter), Route("B", clockwise)))
    assert evaluate(scenario, first).feasible and evaluate(scenario, second).feasible
    ga, gb = build_plan_graph(scenario, first), build_plan_graph(scenario, second)
    global_relation = ("location", "travel", "location")
    assert torch.equal(ga.edge_features[global_relation], gb.edge_features[global_relation])
    arrivals = ("charger", "arrives", "location")
    departures = ("charger", "departs", "location")
    assert not torch.equal(ga.edge_index[arrivals], gb.edge_index[arrivals])
    assert not torch.equal(ga.edge_index[departures], gb.edge_index[departures])
    assert torch.equal(ga.edge_features[arrivals], gb.edge_features[arrivals])


def test_wrsn_forecast_load_timing_is_not_collapsed_to_its_total():
    recipient = Recipient("S", "L", 0, 4, 0, 2, 2, load=(.1, .3, .1, .3))
    scenario = Scenario("WRSN", (Location("L", 0, 0),),
        (Charger("C", "L", 3, 3, 2, 1),), (recipient,), 4)
    altered = replace(scenario, recipients=(replace(recipient, load=(.3, .1, .3, .1)),))
    ga = build_plan_graph(scenario, Plan(()))
    gb = build_plan_graph(altered, Plan(()))
    assert torch.equal(ga.node_features["recipient"], gb.node_features["recipient"])
    relation = ("recipient", "forecast_load", "time")
    assert not torch.equal(ga.edge_features[relation], gb.edge_features[relation])


def test_task_support_keeps_distant_pair_and_every_singleton():
    graph = make_graph()
    assert interaction_support(graph, 1) == ((0,), (1,), (2,))
    assert interaction_support(graph, 2) == ((0,), (1,), (2,), (0, 2))
    assert (0, 2) in interaction_support(graph, 3)
    edge = graph.edge_index[("charger", "shares_task", "charger")]
    assert (edge == torch.tensor([[0], [2]])).all(dim=0).any()
    # Fixed-q urgency, not the later service timestamp: 1 - departure/H.
    assert graph.tasks[0].urgencies == pytest.approx((1-4/6, 0, 1-5/6))


def test_absent_future_route_keeps_unavoidable_travel_and_coalesces_idle_waits():
    scenario = Scenario(
        "EV", (Location("A", 0, 0), Location("B", 10, 0)),
        (Charger("C", "A", 10, 10, 20, 2),), (), horizon=100,
        committed_prefixes=(Route("C", arcs=(Arc("A", "B", 0, 2),)),),
    )
    graph = build_plan_graph(scenario, Plan(()))
    assert graph.route_membership[0]["location"].tolist() == [0, 1]
    assert graph.node_features["time"].shape[0] == 3  # 0, end of travel, end of horizon
    travel = graph.edge_features[("location", "travel", "location")]
    assert travel[:, 3].sum() == 1


def test_many_masks_reuse_one_encoding_and_leave_recovery_shares_fixed():
    graph = make_graph()
    model = small_model(graph)
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    encoded = model.forward_plan(graph)
    before = encoded.coefficients.detach().clone()
    shares = encoded.shares().detach().clone()
    masks = torch.tensor(list(product((0., 1.), repeat=3)))
    values = encoded.values(masks)
    for mask, value in zip(masks, values):
        assert torch.allclose(encoded.values(mask), value)
    assert calls == [1]
    assert torch.equal(before, encoded.coefficients)
    assert torch.equal(shares, encoded.shares())
    assert values[0] == 0
    assert torch.allclose(values[-1], encoded.coefficients.sum())
    hook.remove()


def test_signed_polynomial_and_recovery_allocation_match_exact_toy_game():
    support = ((0,), (1,), (2,), (0, 1), (0, 2))
    membership = torch.tensor([[i in members for i in range(3)] for members in support])
    kappa = torch.zeros(5, 3)
    high, low = .5+.5*math.log(7/3), .5
    kappa[3, 0], kappa[3, 1] = high, low
    kappa[4, 0], kappa[4, 2] = low, high
    encoded = EncodedPlan(torch.tensor([4., 3., 2., 4., -2.]), support, kappa,
        torch.zeros(5, 3), membership, torch.ones(5, dtype=torch.bool),
        ("A", "B", "C"), "q")
    assert encoded.values(torch.ones(3)) == 11
    assert torch.allclose(encoded.allocation(lambda_=0), torch.tensor([5., 5., 1.]))
    assert torch.allclose(encoded.allocation(eta=0), torch.tensor([5.6, 4.6, .8]), atol=1e-6)
    assert torch.allclose(encoded.shares().sum(1), torch.ones(5))
    assert encoded.allocation().sum() == pytest.approx(11)


def test_signed_coefficient_head_and_independent_recovery_loss_receive_gradients():
    graph = make_graph()
    model = small_model(graph)
    output = model.forward_plan(graph)
    targets = torch.full_like(output.recovery_predictions, .99)
    item = replace(example(graph, 1), recovery_targets=targets)
    terms = loss_for_examples(model, [item])
    assert torch.allclose(terms.total, terms.coalition + .5*terms.resource
                          + .5*terms.ranking + .001*terms.sparse)
    terms.total.backward()
    for component in (model.encoder, model.coefficient_head, model.recovery_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    missing = replace(item, recovery_targets={})
    with pytest.raises(ValueError, match="independent physical recovery"):
        loss_for_examples(model, [missing])


def test_coefficient_context_is_symmetric_in_interaction_members():
    graph = make_graph()
    model = small_model(graph)
    # Reversing input member order changes neither pooled route nor task context.
    states, nodes = model.encoder(graph)
    task = nodes["recipient"][list(graph.tasks[0].recipient_indices)].mean(0)
    contexts = [torch.cat((states[list(members)].mean(0), task, states.new_tensor([1.])))
                for members in ((0, 2), (2, 0))]
    assert torch.allclose(model.coefficient_head(contexts[0]),
                          model.coefficient_head(contexts[1]))


def test_supervision_ablation_switches_remove_the_named_loss_only():
    graph = make_graph()
    model = small_model(graph)
    item = example(graph, 1)
    changed = replace(item, coalition_targets=item.coalition_targets+10)
    config = TrainingConfig(coalition_weight=0)
    first = loss_for_examples(model, [item], config=config)
    second = loss_for_examples(model, [changed], config=config)
    assert not torch.allclose(first.coalition, second.coalition)
    assert torch.allclose(first.total, second.total)
    no_recovery = loss_for_examples(model, [replace(item, recovery_targets={})],
                                    config=TrainingConfig(resource_weight=0))
    assert no_recovery.resource == 0
    with pytest.raises(ValueError):
        TrainingConfig(coalition_weight=-1)


def test_same_q_top_bottom_ranking_never_crosses_references():
    first, second = make_graph(), make_graph(seed=10002)
    items = [example(first, float(i)) for i in range(5)]
    items += [example(second, float(i)) for i in range(10, 15)]
    pairs = make_ranking_pairs(items, seed=2)
    assert set(pairs) == {(4, 0), (9, 5)}
    assert all(items[g].reference_id == items[b].reference_id for g, b in pairs)
    with pytest.raises(ValueError, match="same frozen reference"):
        loss_for_examples(small_model(first), items, ranking_pairs=[(9, 0)])


def test_training_only_normalizer_freezes_values_and_embedding_ids():
    first, second = make_graph(), make_graph(seed=10002, energy=.02)
    scaler = FeatureNormalizer.fit([first])
    frozen = {k: v.clone() for k, v in scaler.means.items()}
    normalized = scaler.transform(second)
    assert normalized.input_dims == first.input_dims
    assert torch.equal(normalized.recipient_location_ids, second.recipient_location_ids)
    assert all(torch.equal(v, scaler.means[k]) for k, v in frozen.items())
    restored = FeatureNormalizer.from_state_dict(scaler.state_dict())
    assert torch.equal(restored.transform(second).node_features["recipient"],
                       normalized.node_features["recipient"])


def test_cpu_training_restores_best_checkpoint_and_predicts_after_reload(tmp_path):
    train_graphs = [make_graph(10001, energy=x) for x in (.01, .02)]
    validation_graphs = [make_graph(10002, energy=x) for x in (.01, .02)]
    normalizer = FeatureNormalizer.fit(train_graphs)
    training = [example(normalizer.transform(g), v) for g, v in zip(train_graphs, (1, 2))]
    validation = [example(normalizer.transform(g), v) for g, v in zip(validation_graphs, (1, 2))]
    model = small_model(training[0].graph)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path/"weights.pt"
    config = TrainingConfig(epochs=6, patience=2, batch_size=4, min_delta=1e9)
    result = fit(model, training, validation, config=config, checkpoint=path, normalizer=normalizer)
    assert result.best_epoch == 1 and result.stopped_early
    assert len(result.history) == 3
    assert math.isfinite(result.best_validation_mae)
    assert any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    loaded, loaded_scaler, payload = load_checkpoint(path)
    assert payload["training_configuration"]["ranking_margin"] == .05
    masks = training[0].masks
    assert torch.equal(model(training[0].graph, masks), loaded(training[0].graph, masks))
    assert loaded_scaler is not None
    with pytest.raises(ValueError, match="must be disjoint"):
        fit(model, training, training, config=config)


def test_paper_defaults_are_retained():
    graph = make_graph()
    model = RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 3)
    assert model.configuration["hidden_dim"] == 128
    assert model.configuration["layers"] == 2
    assert model.order == 2
    config = TrainingConfig()
    assert config.coalition_weight == 1
    assert (config.batch_size, config.epochs, config.patience) == (32, 100, 10)
    assert (config.learning_rate, config.weight_decay, config.ranking_margin) == (1e-3, 1e-4, .05)
