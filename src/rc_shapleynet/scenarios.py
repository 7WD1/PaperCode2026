"""Map supplied session/load records into explicit physical scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping
from zoneinfo import ZoneInfo
import math
import numpy as np

from .schema import Charger, Location, Recipient, Scenario
from .data import SessionDay, LocationVocabulary, parse_timestamp


@dataclass(frozen=True)
class EVScenarioBuild:
    scenario: Scenario
    included_session_ids: tuple[str, ...]
    excluded_session_ids: tuple[str, ...]
    clipped_session_ids: tuple[str, ...]
    deadline_sources: Mapping[str, str]


def _per_recipient(value: float | Mapping[str, float], identifier: str) -> float:
    return float(value[identifier]) if isinstance(value, Mapping) else float(value)


def ev_scenario_from_day(day: SessionDay, locations: tuple[Location, ...],
                         parking_map: Mapping[str, str], chargers: tuple[Charger, ...], *,
                         start_time: datetime, horizon: int, step_seconds: float,
                         recipient_capacity_kwh: float | Mapping[str, float],
                         initial_soc_proxy: float | Mapping[str, float],
                         vocabulary: LocationVocabulary, simulation_seed: int,
                         demand_multiplier: float = .9, clip_departures: bool = False,
                         road_distance: Mapping[str, float] | None = None,
                         road_slots: Mapping[str, int] | None = None,
                         return_deadline: int | None = None) -> EVScenarioBuild:
    """Create an EV replay with explicit map, battery proxies and time window.

    Requests arriving outside the supplied window are reported as excluded.
    A departure beyond it raises unless clipping is explicitly selected. Arrival
    rounds upward and departure downward, so quantization cannot expand service
    availability. Requested energy is never silently reduced to fit a battery.
    """
    if horizon <= 0 or step_seconds <= 0 or not math.isfinite(demand_multiplier) or demand_multiplier < 0:
        raise ValueError("Invalid replay horizon, step or demand multiplier")
    start = parse_timestamp(start_time)
    end = start + timedelta(seconds=horizon*step_seconds)
    local_start = start.astimezone(ZoneInfo(day.timezone))
    if local_start.utcoffset() != end.astimezone(ZoneInfo(day.timezone)).utcoffset():
        raise ValueError("Split a replay window at a daylight-saving transition to preserve local clock features")
    ids = {location.id for location in locations}
    included, excluded, clipped, recipients = [], [], [], []
    deadline_sources = {}
    for session in day.sessions:
        if not start <= session.arrival < end:
            excluded.append(session.session_id)
            continue
        departure = session.departure
        if departure > end:
            if not clip_departures:
                raise ValueError("A declared departure exceeds the explicit replay window; extend it or select clipping")
            departure = end
            clipped.append(session.session_id)
        arrival_slot = math.ceil((session.arrival-start).total_seconds()/step_seconds)
        departure_slot = math.floor((departure-start).total_seconds()/step_seconds)
        if arrival_slot >= departure_slot:
            excluded.append(session.session_id)
            continue
        if session.location_id not in parking_map or parking_map[session.location_id] not in ids:
            raise ValueError("Every included source parking ID needs an explicit physical map location")
        capacity = _per_recipient(recipient_capacity_kwh, session.session_id)
        soc = _per_recipient(initial_soc_proxy, session.session_id)
        if not math.isfinite(capacity) or capacity <= 0 or not math.isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("Battery capacity and initial SOC proxy must be explicit and physically valid")
        demand = session.demand_kwh*demand_multiplier
        initial = capacity*soc
        if demand > capacity-initial+1e-8:
            raise ValueError("Demand does not fit the selected capacity/SOC proxy; revise the explicit assumption")
        arrival_local = session.arrival.astimezone(ZoneInfo(day.timezone))
        recipients.append(Recipient(session.session_id, parking_map[session.location_id], arrival_slot,
                                    departure_slot, demand, initial, capacity,
                                    source_proxy=session.source_proxy, initial_soc_proxy=soc,
                                    weekday=arrival_local.weekday(),
                                    location_index=vocabulary.encode(session.location_id),
                                    raw_arrival_fraction=(session.arrival-start).total_seconds()/(horizon*step_seconds),
                                    declared_stay_seconds=(session.departure-session.arrival).total_seconds(),
                                    arrival_clock_seconds=(arrival_local.hour*3600+arrival_local.minute*60
                                                           +arrival_local.second+arrival_local.microsecond/1e6)))
        included.append(session.session_id)
        deadline_sources[session.session_id] = session.deadline_source
    clock = local_start.hour*3600 + local_start.minute*60 + local_start.second + local_start.microsecond/1e6
    scenario = Scenario("EV", locations, chargers, tuple(recipients), horizon,
                        step_seconds=step_seconds, simulation_seed=simulation_seed,
                        scenario_id=day.scenario_id, road_distance=road_distance or {},
                        road_slots=road_slots or {}, return_deadline=return_deadline,
                        clock_offset_seconds=clock)
    return EVScenarioBuild(scenario, tuple(included), tuple(excluded), tuple(clipped), deadline_sources)


def synthetic_partitions(train_count: int = 200, validation_count: int = 100,
                         test_count: int = 100, first_seed: int = 20000
                         ) -> dict[str, tuple[tuple[str, int], ...]]:
    """Freeze disjoint synthetic scenario IDs/seeds before any model selection."""
    if min(train_count, validation_count, test_count) < 0:
        raise ValueError("Partition sizes cannot be negative")
    result, cursor = {}, first_seed
    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        result[split] = tuple((f"wrsn:{split}:{j:05d}", cursor+j) for j in range(count))
        cursor += count
    return result


def wrsn_scenario(*, layout: str, node_count: int, chargers: tuple[Charger, ...],
                   horizon: int, step_seconds: float, simulation_seed: int,
                   scenario_id: str, base_loads: Mapping[str, tuple[float, ...]] | tuple[float, ...],
                   initial_energy: float | Mapping[str, float],
                   capacity: float | Mapping[str, float],
                   transfer_curve: tuple[tuple[float, float], ...],
                   communication_radius: float, load_provenance: str,
                   load_multiplier: float = .9, side_metres: float = 100,
                   sensing_radius: float = 10,
                   required_paths: tuple[tuple[str, ...], ...] | None = None,
                   coverage_points: tuple[tuple[float, float, float], ...] = (),
                   return_deadline: int | None = None) -> Scenario:
    """Generate geometry while retaining externally supplied slot-load vectors.

    ``base_loads`` contains kWh per physical slot. A shared vector is an explicit
    homogeneous-load assumption. ``load_provenance`` identifies calibration or a
    constructed fixture; this generator does not label its input as measured.
    Communication edges use their own radius, independent of charging transfer.
    For the critical-relay layout only, an omitted path means N000--N001--N002.
    Other layouts have no implied required path unless supplied explicitly.
    """
    if layout not in ("uniform", "clustered", "critical-relay"):
        raise ValueError("Unknown synthetic layout")
    if node_count < 1 or (layout == "critical-relay" and node_count < 3):
        raise ValueError("Invalid node count for the selected layout")
    if (not scenario_id or not load_provenance.strip() or side_metres <= 0 or communication_radius <= 0
            or not math.isfinite(load_multiplier) or load_multiplier < 0):
        raise ValueError("Scenario identity, explicit load provenance and physical scales are required")
    rng = np.random.Generator(np.random.PCG64(simulation_seed))
    if layout == "uniform":
        coordinates = rng.uniform(0, side_metres, (node_count, 2))
    elif layout == "clustered":
        centres = side_metres*np.asarray([[.25, .25], [.75, .25], [.5, .75]])
        coordinates = centres[rng.integers(0, 3, node_count)] + rng.normal(0, .06*side_metres, (node_count, 2))
        coordinates = np.clip(coordinates, 0, side_metres)
    else:
        spacing = side_metres/4
        if not spacing <= communication_radius < 2*spacing:
            raise ValueError("Critical relay construction needs spacing <= communication radius < 2*spacing")
        coordinates = np.empty((node_count, 2))
        coordinates[:3] = [[.25*side_metres, .5*side_metres], [.5*side_metres, .5*side_metres],
                            [.75*side_metres, .5*side_metres]]
        for index in range(3, node_count):
            left = index % 2 == 1
            centre = np.asarray([.125 if left else .875, .5])*side_metres
            point = centre+rng.normal(0, .04*side_metres, 2)
            point[0] = np.clip(point[0], 0 if left else .75*side_metres,
                               .25*side_metres if left else side_metres)
            point[1] = np.clip(point[1], 0, side_metres)
            coordinates[index] = point
        if required_paths is None:
            required_paths = (("N000", "N001", "N002"),)
    locations = [Location("depot", .5*side_metres, 0, True)]
    recipients = []
    for index, (x, y) in enumerate(coordinates):
        rid, lid = f"N{index:03d}", f"sensor:{index:03d}"
        loads = base_loads[rid] if isinstance(base_loads, Mapping) else base_loads
        if len(loads) != horizon or any(not math.isfinite(v) or v < 0 for v in loads):
            raise ValueError("Each supplied node trace must contain H finite nonnegative slot energies")
        locations.append(Location(lid, float(x), float(y)))
        recipients.append(Recipient(rid, lid, 0, horizon, 0,
                                    _per_recipient(initial_energy, rid), _per_recipient(capacity, rid),
                                    load=tuple(float(v)*load_multiplier for v in loads),
                                    sensing_radius=sensing_radius, location_index=index+1))
    edges = tuple((a.id, b.id) for i, a in enumerate(recipients) for j, b in enumerate(recipients)
                  if i < j and float(np.linalg.norm(coordinates[i]-coordinates[j])) <= communication_radius)
    return Scenario("WRSN", tuple(locations), chargers, tuple(recipients), horizon,
                    step_seconds=step_seconds, simulation_seed=simulation_seed, scenario_id=scenario_id,
                    communication_edges=edges, required_paths=required_paths or (),
                    transfer_curve=transfer_curve, coverage_points=coverage_points,
                    return_deadline=return_deadline)
