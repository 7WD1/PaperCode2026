"""Constructed input checks; no author dataset or benchmark is reconstructed."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

from rc_shapleynet.data import (
    ACNParserConfig, EnergestCalibration, FrozenDaySelection, LocationVocabulary,
    fit_training_normalizer, freeze_test_days, group_session_days, load_acn_export,
    parse_timestamp, read_energest_csv,
)
from rc_shapleynet.scenarios import ev_scenario_from_day, synthetic_partitions, wrsn_scenario
from rc_shapleynet.schema import Charger, Location, Plan, plan_from_dict, scenario_from_dict, to_dict
from rc_shapleynet.physics import evaluate, service_fraction


def record(identifier="s", arrival="2021-03-01T18:00:00Z", disconnect="2021-03-01T20:00:00Z", **extra):
    return {"sessionID": identifier, "siteID": "0002", "spaceID": "space", "connectionTime": arrival,
            "disconnectTime": disconnect, "kWhDelivered": 4, **extra}


def test_utc_rfc3339_and_acn_rfc1123_share_one_instant():
    expected = parse_timestamp("2021-03-01T18:00:00Z")
    assert parse_timestamp("Mon, 01 Mar 2021 18:00:00 GMT") == expected
    assert parse_timestamp("2021-03-01T10:00:00-08:00") == expected
    with pytest.raises(ValueError, match="Naive"):
        parse_timestamp("2021-03-01T18:00:00")


def test_latest_valid_explicit_energy_and_declared_departure_take_priority():
    raw = record(userInputs=[
        {"modifiedAt": "2021-03-01T19:00:00Z", "kWhRequested": -2},
        {"modifiedAt": "2021-03-01T18:30:00Z", "kWhRequested": 6,
         "requestedDeparture": "2021-03-01T22:00:00Z"},
        {"modifiedAt": "2021-03-01T18:00:00Z", "kWhRequested": 3,
         "requestedDeparture": "2021-03-01T19:00:00Z"},
    ])
    session = load_acn_export({"_items": [raw]}).sessions[0]
    assert session.demand_kwh == 6
    assert not session.source_proxy
    assert session.departure == parse_timestamp("2021-03-01T22:00:00Z")
    assert session.deadline_source == "declared"
    capped = load_acn_export([raw], ACNParserConfig(cap_declared_by_disconnect=True)).sessions[0]
    assert capped.departure == capped.disconnect
    assert capped.deadline_source == "min_declared_disconnect"


def test_delivered_energy_is_a_flagged_proxy_and_never_overrides_explicit_zero():
    proxy = load_acn_export([record(userInputs=[{"kWhRequested": -1}])]).sessions[0]
    assert proxy.source_proxy and proxy.demand_kwh == 4
    assert proxy.deadline_source == "disconnect_proxy"
    zero = load_acn_export([record(userInputs=[{"kWhRequested": 0}])]).sessions[0]
    assert not zero.source_proxy and zero.demand_kwh == 0


def test_cleaning_reports_each_rejection_without_inventing_counts():
    raw = [record(), record(), {}, record("negative", kWhDelivered=-1),
           record("backwards", disconnect="2021-03-01T17:00:00Z"),
           record("noenergy", kWhDelivered=None), record("noloc", spaceID=None),
           record("other", siteID="unselected")]
    result = load_acn_export(raw)
    assert result.raw_count == 8
    assert len(result.sessions) == 1
    assert sum(result.rejected.values()) == 7
    assert result.rejected["duplicate_session"] == 1
    assert result.rejected["invalid_delivered_energy"] == 1
    assert result.rejected["invalid_duration"] == 1
    thresholded = load_acn_export([record()], ACNParserConfig(max_duration_seconds=3600))
    assert not thresholded.sessions


def test_cutoff_and_splits_use_los_angeles_arrival_dates():
    rows = [record("train", "2020-07-01T06:00:00Z", "2020-07-01T08:00:00Z"),
            record("val", "2020-07-01T07:00:00Z", "2020-07-01T09:00:00Z"),
            record("lastval", "2021-02-01T07:00:00Z", "2021-02-01T09:00:00Z"),
            record("test", "2021-02-01T08:00:00Z", "2021-02-01T10:00:00Z"),
            record("lastday", "2021-09-14T06:00:00Z", "2021-09-14T08:00:00Z"),
            record("past", "2021-09-14T07:00:00Z", "2021-09-14T09:00:00Z")]
    result = load_acn_export(rows)
    assert {s.session_id: s.split for s in result.sessions} == {
        "train": "train", "val": "validation", "lastval": "validation", "test": "test", "lastday": "test"}
    assert result.rejected["after_cutoff"] == 1


def test_sessions_group_into_local_days_before_shared_scenario_selection(tmp_path):
    rows = []
    for index in range(102):
        arrival = datetime(2021, 2, 2, 18, tzinfo=timezone.utc)+timedelta(days=index)
        for duplicate_day_member in range(2):
            rows.append(record(f"s{index}:{duplicate_day_member}", arrival.isoformat(),
                               (arrival+timedelta(hours=2)).isoformat()))
    days = group_session_days(load_acn_export(rows))
    assert len(days) == 102 and all(len(day.sessions) == 2 for day in days)
    selected = freeze_test_days(days)
    assert selected == freeze_test_days(days)
    assert len(set(selected.scenario_ids)) == 100
    assert selected.simulation_seeds == tuple(range(10001, 10101))
    assert selected.training_seeds == (1, 2, 3, 4, 5)
    assert all(len(ids) == 2 for ids in selected.session_ids)
    path = tmp_path/"selection.json"
    selected.save(path)
    assert FrozenDaySelection.load(path) == selected
    with pytest.raises(ValueError, match="enough distinct"):
        freeze_test_days(days[:2])


def test_location_vocabulary_is_train_only_and_unknown_is_frozen(tmp_path):
    train = load_acn_export([record(arrival="2019-05-01T17:00:00Z", disconnect="2019-05-01T19:00:00Z")])
    vocabulary = LocationVocabulary.fit(train.sessions)
    assert vocabulary.encode("space") == 1 and vocabulary.encode("unseen-test-space") == 0
    path = tmp_path/"vocabulary.json"
    vocabulary.save(path)
    assert LocationVocabulary.load(path) == vocabulary
    with pytest.raises(ValueError, match="training"):
        LocationVocabulary.fit(load_acn_export([record()]).sessions)
    with pytest.raises(ValueError, match="training"):
        fit_training_normalizer(group_session_days(load_acn_export([record()])), lambda _: None)


def test_ev_mapping_preserves_source_and_conservative_slots_and_local_clock():
    parsed = load_acn_export([record(arrival="2019-05-01T17:00:01Z", disconnect="2019-05-01T19:00:00Z")])
    day = group_session_days(parsed)[0]
    built = ev_scenario_from_day(day, (Location("parking", 0, 0),), {"space": "parking"},
                                 (Charger("C", "parking", 10, 10, 1, 1),),
                                 start_time=parse_timestamp("2019-05-01T17:00:00Z"), horizon=4, step_seconds=3600,
                                 recipient_capacity_kwh=20, initial_soc_proxy=.25,
                                 vocabulary=LocationVocabulary.fit(parsed.sessions), simulation_seed=10001,
                                 demand_multiplier=1)
    recipient = built.scenario.recipients[0]
    assert (recipient.arrival, recipient.departure) == (1, 2)
    assert recipient.source_proxy and recipient.demand == 4
    assert recipient.initial_energy == 5 and recipient.location_index == 1
    assert built.scenario.clock_offset_seconds == 10*3600
    assert recipient.raw_arrival_fraction == pytest.approx(1/(4*3600))
    assert recipient.declared_stay_seconds == 2*3600-1
    assert recipient.arrival_clock_seconds == 10*3600+1
    assert built.deadline_sources["s"] == "disconnect_proxy"
    assert evaluate(built.scenario, Plan(())).feasible


def test_original_request_metadata_survives_rolling_clipping_and_serialization():
    from rc_shapleynet.simulation import execute_interval
    parsed = load_acn_export([record(arrival="2019-05-01T17:00:01Z", disconnect="2019-05-01T19:00:00Z")])
    day = group_session_days(parsed)[0]
    built = ev_scenario_from_day(day, (Location("parking", 0, 0),), {"space": "parking"}, (),
                                 start_time=parse_timestamp("2019-05-01T17:00:00Z"), horizon=4,
                                 step_seconds=3600, recipient_capacity_kwh=20, initial_soc_proxy=0,
                                 vocabulary=LocationVocabulary.fit(parsed.sessions), simulation_seed=10001)
    first = built.scenario.recipients[0]
    updated = execute_interval(built.scenario, Plan(()), 1).next_scenario
    after = updated.recipients[0]
    assert after.arrival == 0 and after.departure == 1
    for field in ("raw_arrival_fraction", "declared_stay_seconds", "arrival_clock_seconds"):
        assert getattr(after, field) == getattr(first, field)
    restored = scenario_from_dict(to_dict(updated))
    assert restored.fingerprint() == updated.fingerprint()
    assert restored.recipients[0].arrival_clock_seconds == 36001


def test_legacy_inputs_freeze_their_initial_feature_fallback_once():
    from rc_shapleynet.schema import Recipient, Scenario
    from rc_shapleynet.simulation import execute_interval
    scenario = Scenario("EV", (Location("r", 0, 0),), (),
                        (Recipient("r", "r", 1, 4, 1, 0, 2),), 6, step_seconds=60,
                        clock_offset_seconds=10*3600)
    original = scenario.recipients[0]
    assert original.raw_arrival_fraction is None
    first = execute_interval(scenario, Plan(()), 2).next_scenario
    second = execute_interval(first, Plan(()), 1).next_scenario
    for updated in (first, second):
        recipient = updated.recipients[0]
        assert recipient.arrival == 0
        assert recipient.raw_arrival_fraction == pytest.approx(1/6)
        assert recipient.declared_stay_seconds == 3*60
        assert recipient.arrival_clock_seconds == 10*3600+60


def test_replay_clipping_is_an_explicit_reported_assumption():
    parsed = load_acn_export([record(arrival="2019-05-01T17:00:00Z", disconnect="2019-05-01T21:00:00Z")])
    args = dict(day=group_session_days(parsed)[0], locations=(Location("parking", 0, 0),),
                parking_map={"space": "parking"}, chargers=(),
                start_time=parse_timestamp("2019-05-01T17:00:00Z"), horizon=2, step_seconds=3600,
                recipient_capacity_kwh=20, initial_soc_proxy=0,
                vocabulary=LocationVocabulary.fit(parsed.sessions), simulation_seed=10001)
    with pytest.raises(ValueError, match="exceeds"):
        ev_scenario_from_day(**args)
    explicit = ev_scenario_from_day(**args, clip_departures=True)
    assert explicit.clipped_session_ids == ("s",)
    assert explicit.scenario.recipients[0].departure == 2


def test_energest_current_tick_units_and_complete_observations(tmp_path):
    path = tmp_path/"load.csv"
    path.write_text("node_id,slot,cpu_ticks,lpm_ticks,tx_ticks,rx_ticks\nr,0,18000,18000,18000,0\n", encoding="utf-8")
    calibration = EnergestCalibration(3, 10, 1, .1, 10, 8)
    assert read_energest_csv(path, calibration, 1, 3600)["r"][0] == pytest.approx(.00001665)
    with pytest.raises(ValueError, match="all documented"):
        read_energest_csv(path, calibration, 2, 3600)
    cumulative = tmp_path/"cumulative.csv"
    cumulative.write_text("node_id,slot,cpu_ticks,lpm_ticks,tx_ticks,rx_ticks\nr,0,100,200,0,0\nr,1,18100,18200,18000,0\n", encoding="utf-8")
    assert read_energest_csv(cumulative, calibration, 1, 3600, cumulative=True)["r"][0] == pytest.approx(.00001665)
    cumulative.write_text("node_id,slot,cpu_ticks,lpm_ticks,tx_ticks,rx_ticks\nr,0,60000,60000,0,0\nr,1,12464,12464,18000,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reset"):
        read_energest_csv(cumulative, calibration, 1, 3600, cumulative=True)
    assert read_energest_csv(cumulative, calibration, 1, 3600, cumulative=True,
                            counter_modulus=65536)["r"][0] == pytest.approx(.00001665)


def test_synthetic_partitions_and_geometry_are_disjoint_and_reproducible():
    partitions = synthetic_partitions(4, 2, 3)
    pairs = [pair for group in partitions.values() for pair in group]
    assert len({identifier for identifier, _ in pairs}) == len(pairs)
    assert len({seed for _, seed in pairs}) == len(pairs)
    args = dict(layout="critical-relay", node_count=8,
                chargers=(Charger("C", "depot", 10, 10, 1, 1),), horizon=12, step_seconds=3600,
                simulation_seed=20000, scenario_id="wrsn:train:00000", base_loads=(.01,)*12,
                initial_energy=.5, capacity=1, transfer_curve=((0, .8), (5, 0)),
                communication_radius=30, load_provenance="constructed homogeneous microfixture")
    scenario = wrsn_scenario(**args)
    assert scenario.fingerprint() == wrsn_scenario(**args).fingerprint()
    assert scenario.fingerprint() != wrsn_scenario(**{**args, "simulation_seed": 20001}).fingerprint()
    assert scenario.required_paths == (("N000", "N001", "N002"),)
    assert ("N000", "N001") in scenario.communication_edges
    assert ("N000", "N002") not in scenario.communication_edges
    assert service_fraction(scenario, scenario.chargers[0], "sensor:000", "N001") == 0
    assert scenario.recipients[0].load[0] == pytest.approx(.009)
    assert evaluate(scenario, Plan(())).feasible


@pytest.mark.parametrize("name", ["figure2", "wrsn_small"])
def test_constructed_inputs_roundtrip_and_pass_the_full_physical_checker(name):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root/"examples"/f"{name}.json").read_text(encoding="utf-8"))
    assert payload["constructed"] is True
    scenario, plan = scenario_from_dict(payload["scenario"]), plan_from_dict(payload["plan"])
    assert scenario.fingerprint() == scenario_from_dict(to_dict(scenario)).fingerprint()
    result = evaluate(scenario, plan)
    assert result.feasible, result.violations
