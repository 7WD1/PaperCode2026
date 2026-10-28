from dataclasses import replace
import math

import numpy as np
import pytest

from rc_shapleynet.schema import Arc, Charger, Location, Plan, Power, Recipient, Route, Scenario, Service
from rc_shapleynet.physics import evaluate


def ev_fixture():
    scenario = Scenario("EV", (Location("r", 0, 0),),
                        (Charger("A", "r", 7, 7, 1, 1, receiver_efficiency=1),
                         Charger("B", "r", 4, 4, 1, 1, receiver_efficiency=1)),
                        (Recipient("r", "r", 0, 8, 8, 0, 8),), 8, step_seconds=3600)
    def route(cid, slots):
        return Route(cid, tuple(Arc("r", "r", t, t + 1) for t in range(8)),
                     tuple(Service("r", t, 1) for t in slots), tuple(Power(t, 1) for t in slots))
    return scenario, Plan((route("A", range(4)), route("B", range(4, 8))))


def test_constructed_two_route_plan_and_full_energy_ledger():
    scenario, plan = ev_fixture()
    result = evaluate(scenario, plan)
    assert result.feasible, result.violations
    assert result.metrics["completion"] == 1
    assert result.metrics["delivered_energy"] == 8
    assert result.objective == pytest.approx(.65 - .1 * 8 / 1000)
    np.testing.assert_allclose(result.battery["A"], [7, 6, 5, 4, 3, 3, 3, 3, 3])
    np.testing.assert_allclose(result.recipient_energy["r"], np.arange(9))
    assert len(result.ledgers["chargers"]["A"]["dwell"]) == 8


def test_single_effective_recipient_input_rejects_parallel_chargers():
    scenario, plan = ev_fixture()
    b = replace(plan.route("B"), services=(Service("r", 0, 1),), powers=(Power(0, 1),))
    result = evaluate(scenario, Plan((plan.route("A"), b)))
    assert not result.feasible
    assert any("multiple effective recipient inputs" in v for v in result.violations)
    assert result.objective == -math.inf


def test_complete_route_requires_explicit_waiting_until_horizon():
    scenario, plan = ev_fixture()
    short = replace(plan.route("A"), arcs=plan.route("A").arcs[:-1])
    result = evaluate(scenario, Plan((short, plan.route("B"))))
    assert any("through horizon" in v for v in result.violations)
    assert evaluate(scenario, Plan(())).feasible  # Missing routes are idle.


def test_original_deadline_and_energy_acceptance_are_enforced():
    scenario, plan = ev_fixture()
    early = replace(scenario, recipients=(replace(scenario.recipients[0], departure=7),))
    assert any("outside original availability" in v for v in evaluate(early, plan).violations)
    full = replace(scenario, recipients=(replace(scenario.recipients[0], initial_energy=1),))
    assert any("acceptance capacity" in v for v in evaluate(full, plan).violations)


def test_travel_flow_frozen_duration_and_safe_return_deadline():
    scenario = Scenario("EV", (Location("d", 0, 0, True), Location("x", 2, 0)),
                        (Charger("C", "d", 10, 10, 1, 1, energy_per_metre=1),), (),
                        6, step_seconds=1,
                        road_distance={"d|x": 2, "x|d": 2},
                        road_slots={"C|d|x": 2, "C|x|d": 2})
    route = Route("C", (Arc("d", "x", 0, 2), Arc("x", "x", 2, 3),
                         Arc("x", "d", 3, 5), Arc("d", "d", 5, 6)))
    good = evaluate(scenario, Plan((route,)))
    assert good.feasible, good.violations
    np.testing.assert_allclose(good.battery["C"], [10, 9, 8, 8, 7, 6, 6])
    bad = replace(route, arcs=(Arc("d", "x", 0, 1),) + route.arcs[1:])
    assert not evaluate(scenario, Plan((bad,))).feasible
    terminal_x = Route("C", (Arc("d", "x", 0, 2),) + tuple(Arc("x", "x", t, t+1) for t in range(2, 6)))
    assert any("safe return misses deadline" in v for v in evaluate(scenario, Plan((terminal_x,))).violations)
    assert evaluate(replace(scenario, return_deadline=8), Plan((terminal_x,))).feasible


def test_broadcast_conservation_and_receiver_curve_are_separate_bounds():
    scenario = Scenario("WRSN", (Location("d", 0, 0, True),),
                        (Charger("C", "d", 2, 2, 1, 1),),
                        (Recipient("a", "d", 0, 1, 0, 0, 2), Recipient("b", "d", 0, 1, 0, 0, 2)),
                        1, step_seconds=3600, transfer_curve=((0, .8), (10, 0)))
    def check(a, b):
        return evaluate(scenario, Plan((Route("C", (Arc("d", "d", 0, 1),),
                                                    (Service("a", 0, a), Service("b", 0, b)),
                                                    (Power(0, 1),)),)))
    assert check(.5, .5).feasible
    assert any("broadcast energy conservation" in v for v in check(.6, .6).violations)
    assert any("transfer bound" in v for v in check(.9, .0).violations)


def test_fractional_outage_path_and_spatial_coverage_have_different_denominators():
    scenario = Scenario("WRSN", (Location("a", 0, 0), Location("b", 100, 0)), (),
                        (Recipient("a", "a", 0, 1, 0, .25, 1, load=(1,), sensing_radius=1),
                         Recipient("b", "b", 0, 1, 0, .75, 1, load=(1,), sensing_radius=1)),
                        1, step_seconds=3600, communication_edges=(("a", "b"),),
                        required_paths=(("a", "b"),), coverage_points=((0, 0, 1),))
    result = evaluate(scenario, Plan(()))
    assert result.feasible
    assert result.metrics["C_alive"] == .5
    assert result.metrics["C_conn"] == .25
    assert result.metrics["C_cov"] == .25
    assert result.metrics["outage_node_hours"] == 1
    assert result.objective == pytest.approx(.175)


def test_receiver_acceptance_is_before_load_and_load_cannot_be_suppressed():
    scenario = Scenario("WRSN", (Location("d", 0, 0, True),),
                        (Charger("C", "d", 2, 2, 1, 1),),
                        (Recipient("r", "d", 0, 1, 0, 1, 1, load=(1,)),),
                        1, step_seconds=3600)
    route = Route("C", (Arc("d", "d", 0, 1),), (Service("r", 0, .1),), (Power(0, .2),))
    assert any("acceptance capacity" in v for v in evaluate(scenario, Plan((route,))).violations)
    idle = evaluate(scenario, Plan(()))
    assert idle.recipient_energy["r"][-1] == 0
    assert idle.ledgers["recipients"]["r"]["executed_load"][0] == 1


def test_depot_input_requires_depot_and_cannot_hide_simultaneous_service():
    scenario, plan = ev_fixture()
    route = replace(plan.route("A"), powers=(Power(0, 1, 1),) + plan.route("A").powers[1:])
    result = evaluate(scenario, Plan((route, plan.route("B"))))
    assert any("depot input away" in v for v in result.violations)
    assert any("simultaneous depot" in v for v in result.violations)


def test_fixed_restriction_preserves_retained_schedule():
    scenario, plan = ev_fixture()
    restricted = plan.restrict({"B"})
    assert restricted.route("B") == plan.route("B")
    evaluation = evaluate(scenario, restricted)
    assert evaluation.feasible
    np.testing.assert_allclose(evaluation.ledgers["recipients"]["r"]["received"], [0, 0, 0, 0, 1, 1, 1, 1])


def test_safe_destination_must_jointly_satisfy_energy_and_time():
    scenario = Scenario("EV", (Location("x", 0, 0), Location("slow", 1, 0, True),
                                Location("fast", 2, 0, True)),
                        (Charger("C", "x", 2, 2, 1, 1, energy_per_metre=1),), (),
                        1, step_seconds=1, return_deadline=2,
                        road_distance={"x|slow": 1, "x|fast": 2},
                        road_slots={"C|x|slow": 5, "C|x|fast": 1})
    result = evaluate(scenario, Plan(()))
    assert result.feasible, result.violations
    assert result.ledgers["chargers"]["C"]["return_destinations"] == ("fast", "fast")


def test_execute_first_interval_preserves_unavoidable_travel_and_absolute_return_time():
    from rc_shapleynet.simulation import execute_interval, shift_plan
    scenario = Scenario("EV", (Location("d", 0, 0), Location("x", 2, 0)),
                        (Charger("C", "d", 10, 10, 1, 1, energy_per_metre=1),), (), 6,
                        road_distance={"d|x": 2, "x|d": 2},
                        road_slots={"C|d|x": 2, "C|x|d": 2})
    route = Route("C", (Arc("d", "x", 0, 2), Arc("x", "x", 2, 3),
                         Arc("x", "d", 3, 5), Arc("d", "d", 5, 6)))
    plan = Plan((route,))
    step = execute_interval(scenario, plan, 1)
    updated = step.next_scenario
    assert updated is not None
    assert updated.horizon == updated.return_deadline == 5
    assert updated.clock_offset_seconds == scenario.clock_offset_seconds+scenario.step_seconds
    assert updated.safe_locations["C"] == "d"
    assert updated.chargers[0].initial_energy == 9
    assert updated.chargers[0].initial_location.startswith("@transit:")
    assert updated.committed_prefixes[0].arcs[0].end == 1
    continuation = shift_plan(scenario, plan, 1)
    assert evaluate(updated, continuation).feasible
    # Even an absent future route must execute the same common travel prefix.
    empty = evaluate(updated, Plan(()))
    assert empty.ledgers["chargers"]["C"]["state_location"][1] == "x"
    changed = replace(continuation.route("C"), arcs=(Arc(updated.chargers[0].initial_location,
                                                        updated.chargers[0].initial_location, 0, 1),)
                      + continuation.route("C").arcs[1:])
    assert any("unavoidable in-progress" in v for v in evaluate(updated, Plan((changed,))).violations)


def test_episode_aggregation_matches_uninterrupted_execution():
    from rc_shapleynet.simulation import run_episode, shift_plan
    scenario, plan = ev_fixture()
    previous = []
    def planner(reference):
        candidate = plan if not previous else shift_plan(previous[-1][0], previous[-1][1], 2)
        previous.append((reference, candidate))
        return candidate
    episode = run_episode(scenario, planner, 2)
    uninterrupted = evaluate(scenario, plan)
    assert len(episode.steps) == 4
    assert episode.objective == pytest.approx(uninterrupted.objective)
    np.testing.assert_allclose(episode.battery["A"], uninterrupted.battery["A"])
    np.testing.assert_allclose(episode.recipient_energy["r"], uninterrupted.recipient_energy["r"])


def test_cached_exogenous_profiles_cannot_change_between_coalitions(tmp_path):
    from rc_shapleynet.sumo_adapter import FrozenRoadProfile, FrozenTrace
    profile = FrozenRoadProfile({"a|b": 10}, {"C|a|b": 2}, {"a|b": ("e",)}, "hash", "basis")
    with pytest.raises(TypeError):
        profile.road_distance["a|b"] = 99
    path = tmp_path / "profile.json"
    profile.save(path)
    assert FrozenRoadProfile.load(path).road_distance["a|b"] == 10
    trace = FrozenTrace(1, "test", ({"time": 1, "vehicles": {"v": {"position": [1, 2]}}},))
    with pytest.raises(TypeError):
        trace.snapshots[0]["vehicles"]["v"]["position"] = (9, 9)
    target = tmp_path / "trace.json"
    trace.save(target)
    assert FrozenTrace.load(target).snapshots[0]["vehicles"]["v"]["position"] == (1, 2)


def test_real_sumo_network_step_capture_and_frozen_profiles(tmp_path):
    pytest.importorskip("libsumo")
    pytest.importorskip("sumolib")
    from rc_shapleynet.sumo_adapter import RoadPosition, SUMOAdapter, freeze_network
    network = tmp_path / "network.net.xml"
    network.write_text('''<net version="1.20" junctionCornerDetail="0" limitTurnSpeed="5.50">
      <location netOffset="0,0" convBoundary="0,0,100,0" origBoundary="0,0,100,0" projParameter="!"/>
      <edge id="a" from="n0" to="n1" priority="1"><lane id="a_0" index="0" speed="10" length="50" shape="0,0 50,0"/></edge>
      <edge id="b" from="n1" to="n2" priority="1"><lane id="b_0" index="0" speed="10" length="50" shape="50,0 100,0"/></edge>
      <junction id="n0" type="dead_end" x="0" y="0" incLanes="" intLanes="" shape="0,-1 0,1"/>
      <junction id="n1" type="priority" x="50" y="0" incLanes="a_0" intLanes="" shape="49,-1 51,-1 51,1 49,1"><request index="0" response="0" foes="0" cont="0"/></junction>
      <junction id="n2" type="dead_end" x="100" y="0" incLanes="b_0" intLanes="" shape="100,-1 100,1"/>
      <connection from="a" to="b" fromLane="0" toLane="0" dir="s" state="M"/>
    </net>''', encoding="utf-8")
    config = tmp_path / "case.sumocfg"
    config.write_text('<configuration><input><net-file value="network.net.xml"/></input><time><begin value="0"/><end value="30"/></time></configuration>', encoding="utf-8")
    profile = freeze_network(network, {"s": RoadPosition("a", 0), "t": RoadPosition("b", 50)},
                             (Charger("C", "s", 1, 1, 1, 5),), 1)
    assert profile.road_distance["s|t"] == pytest.approx(100)
    assert profile.road_slots["C|s|t"] == 20
    assert "t|s" not in profile.road_distance
    with SUMOAdapter(config, seed=10001, extra_args=("--no-step-log", "true", "--no-warnings", "true")) as adapter:
        adapter.add_vehicle("charger", ("a", "b"), max_speed=5)
        trace = adapter.capture(5)
        assert len(trace.snapshots) == 5
        assert trace.snapshots[-1]["time"] == 5
        positions = [s["vehicles"]["charger"]["position"][0] for s in trace.snapshots if "charger" in s["vehicles"]]
        assert len(positions) >= 3
        assert positions[-1] > positions[0]
        assert "SUMO" in trace.version


def test_repeated_midtravel_replanning_has_no_energy_or_position_teleport():
    from rc_shapleynet.simulation import run_episode, shift_plan
    scenario = Scenario("EV", (Location("d", 0, 0, True), Location("x", 3, 0)),
                        (Charger("C", "d", 10, 10, 1, 1, energy_per_metre=1),), (), 7,
                        road_distance={"d|x": 3, "x|d": 3},
                        road_slots={"C|d|x": 3, "C|x|d": 3})
    initial_plan = Plan((Route("C", (Arc("d", "x", 0, 3), Arc("x", "x", 3, 4),
                                     Arc("x", "d", 4, 7))),))
    history = []
    def planner(reference):
        candidate = initial_plan if not history else shift_plan(history[-1][0], history[-1][1], 1)
        history.append((reference, candidate))
        return candidate
    result = run_episode(scenario, planner, 1)
    expected = evaluate(scenario, initial_plan)
    np.testing.assert_allclose(result.battery["C"], expected.battery["C"])
    assert result.objective == pytest.approx(expected.objective)
    assert result.battery["C"][-1] == 4


def test_wrsn_replanning_integrates_partial_last_slot_without_double_counting():
    from rc_shapleynet.simulation import run_episode
    scenario = Scenario("WRSN", (Location("r", 0, 0),), (),
                        (Recipient("r", "r", 0, 4, 0, 2.5, 3, load=(1, 1, 1, 1)),),
                        4, step_seconds=3600, required_paths=(("r",),), coverage_points=((0, 0, 1),))
    episode = run_episode(scenario, lambda _: Plan(()), 1)
    whole = evaluate(scenario, Plan(()))
    assert episode.metrics["C_alive"] == .625
    assert episode.metrics["outage_node_hours"] == 1.5
    assert episode.objective == pytest.approx(whole.objective)
    np.testing.assert_allclose(episode.recipient_energy["r"], whole.recipient_energy["r"])


@pytest.mark.parametrize("second_energy,missing_edge", [(0, False), (1, True)])
def test_connectivity_averages_required_paths_instead_of_requiring_them_all(second_energy, missing_edge):
    scenario = Scenario("WRSN", tuple(Location(rid, i*10, 0) for i, rid in enumerate("abcd")), (),
                        tuple(Recipient(rid, rid, 0, 1, 0, energy, 1, load=(1,))
                              for rid, energy in zip("abcd", [1, 1, second_energy, 1])),
                        1, step_seconds=3600,
                        communication_edges=(("a", "b"),) if missing_edge else (("a", "b"), ("c", "d")),
                        required_paths=(("a", "b"), ("c", "d")))
    result = evaluate(scenario, Plan(()))
    assert result.feasible, result.violations
    assert result.metrics["C_conn"] == .5
    assert result.ledgers["recipients"]["_network"]["connectivity_fraction"][0] == .5


def test_multiple_path_connectivity_matches_rolling_execution():
    from rc_shapleynet.simulation import run_episode
    scenario = Scenario("WRSN", tuple(Location(rid, i*10, 0) for i, rid in enumerate("abcd")), (),
                        tuple(Recipient(rid, rid, 0, 2, 0, energy, 2, load=(1, 1))
                              for rid, energy in zip("abcd", [2, 2, .5, 2])),
                        2, step_seconds=3600, communication_edges=(("a", "b"), ("c", "d")),
                        required_paths=(("a", "b"), ("c", "d")))
    full = evaluate(scenario, Plan(()))
    episode = run_episode(scenario, lambda _: Plan(()), 1)
    assert full.metrics["C_conn"] == .625
    assert episode.metrics["C_conn"] == .625
    assert episode.objective == pytest.approx(full.objective)
