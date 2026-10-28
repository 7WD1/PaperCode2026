"""Typed, serializable inputs for the manuscript's fixed-reference model.

Slots have duration ``step_seconds``; energy is kWh and power is kW. An arc
occupies [start, end). Service is credited in its slot, before that slot's load.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Mapping
from types import MappingProxyType
import hashlib
import json
import math


def to_dict(value):
    """JSON-compatible dataclasses, including immutable reference mappings."""
    if is_dataclass(value):
        return {f.name:to_dict(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):
        return {str(k):to_dict(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [to_dict(v) for v in value]
    return value


@dataclass(frozen=True)
class Location:
    id: str
    x: float
    y: float
    depot: bool = False


@dataclass(frozen=True)
class Recipient:
    id: str
    location_id: str
    arrival: int
    departure: int
    demand: float
    initial_energy: float
    capacity: float
    load: tuple[float, ...] = ()
    source_proxy: bool = False
    initial_soc_proxy: float = 0.0
    weekday: int = 0
    sensing_radius: float = 10.0
    location_index: int = 0
    # Original request features remain distinct from rolling availability slots.
    raw_arrival_fraction: float | None = None
    declared_stay_seconds: float | None = None
    arrival_clock_seconds: float | None = None


@dataclass(frozen=True)
class Charger:
    id: str
    initial_location: str
    initial_energy: float
    capacity: float
    max_power: float
    speed_mps: float
    energy_per_metre: float = 0.0001
    auxiliary_kw: float = 0.0
    receiver_efficiency: float = 0.92
    depot_power: float = 0.0
    reserve_energy: float = 0.0
    ports: int = 1


@dataclass(frozen=True)
class Arc:
    origin: str
    destination: str
    start: int
    end: int


@dataclass(frozen=True)
class Service:
    recipient_id: str
    slot: int
    energy: float


@dataclass(frozen=True)
class Power:
    slot: int
    output_energy: float = 0.0
    depot_energy: float = 0.0


@dataclass(frozen=True)
class Route:
    charger_id: str
    arcs: tuple[Arc, ...] = ()
    services: tuple[Service, ...] = ()
    powers: tuple[Power, ...] = ()


@dataclass(frozen=True)
class Plan:
    routes: tuple[Route, ...]

    def route(self, charger_id: str) -> Route:
        matches = [r for r in self.routes if r.charger_id == charger_id]
        if len(matches) > 1:
            raise ValueError("A complete plan has at most one route per charger")
        return matches[0] if matches else Route(charger_id)

    def restrict(self, members: set[str] | frozenset[str]) -> Plan:
        """Disable future routes without reoptimizing retained timing."""
        return Plan(tuple(r for r in self.routes if r.charger_id in members))


@dataclass(frozen=True)
class Task:
    id: str
    route_ids: tuple[str, ...]
    recipient_ids: tuple[str, ...]
    kind: str = "service"


@dataclass(frozen=True)
class Scenario:
    domain: Literal["EV", "WRSN"]
    locations: tuple[Location, ...]
    chargers: tuple[Charger, ...]
    recipients: tuple[Recipient, ...]
    horizon: int
    step_seconds: float = 1.0
    simulation_seed: int = 10001
    scenario_id: str = "validation"
    tasks: tuple[Task, ...] = ()
    # Frozen directed road distances/durations. Missing entries use Euclidean
    # distance only when no road table was supplied.
    road_distance: Mapping[str, float] = field(default_factory=dict)
    road_slots: Mapping[str, int] = field(default_factory=dict)
    communication_edges: tuple[tuple[str, str], ...] = ()
    required_paths: tuple[tuple[str, ...], ...] = ()
    # Distance in metres -> total receiver-side fraction of transmitter energy.
    transfer_curve: tuple[tuple[float, float], ...] = ((0.0, 0.8), (5.0, 0.4), (10.0, 0.0))
    # Fixed spatial quadrature points (x,y,weight) for reported sensing coverage.
    coverage_points: tuple[tuple[float, float, float], ...] = ()
    energy_reference: float = 1000.0
    allow_simultaneous_depot_service: bool = False
    # q's safe-return policy: None resolves to H. Depots are safe locations;
    # without depots, each charger's own initial location is its safe location.
    return_deadline: int | None = None
    # Unavoidable in-progress travel at q belongs to the common reference,
    # including the empty plan. Explicit routes must preserve this prefix.
    committed_prefixes: tuple[Route, ...] = ()
    safe_locations: Mapping[str, str] = field(default_factory=dict)
    clock_offset_seconds: float = 0.0

    def __post_init__(self):
        if self.domain not in ("EV", "WRSN") or self.horizon <= 0 or self.step_seconds <= 0:
            raise ValueError("Invalid domain, horizon or physical step")
        numeric = [self.step_seconds, self.energy_reference, self.clock_offset_seconds]
        for location in self.locations:
            numeric.extend((location.x, location.y))
        for c in self.chargers:
            numeric.extend((c.initial_energy,c.capacity,c.max_power,c.speed_mps,c.energy_per_metre,
                            c.auxiliary_kw,c.receiver_efficiency,c.depot_power,c.reserve_energy))
            if min(c.energy_per_metre,c.auxiliary_kw,c.depot_power,c.reserve_energy) < 0 or c.ports < 1:
                raise ValueError("Invalid nonnegative charger parameters or port count")
        for r in self.recipients:
            numeric.extend((r.demand,r.initial_energy,r.capacity,r.initial_soc_proxy,r.sensing_radius,*r.load))
            numeric.extend(v for v in (r.raw_arrival_fraction, r.declared_stay_seconds,
                                        r.arrival_clock_seconds) if v is not None)
            if r.sensing_radius < 0 or not 0 <= r.weekday <= 6 or r.location_index < 0:
                raise ValueError("Invalid recipient metadata")
            if (r.raw_arrival_fraction is not None and not 0 <= r.raw_arrival_fraction <= 1 or
                r.declared_stay_seconds is not None and r.declared_stay_seconds < 0 or
                r.arrival_clock_seconds is not None and not 0 <= r.arrival_clock_seconds < 86400):
                raise ValueError("Invalid original arrival/stay metadata")
        numeric.extend(self.road_distance.values())
        numeric.extend(self.road_slots.values())
        numeric.extend(v for row in self.transfer_curve for v in row)
        numeric.extend(v for row in self.coverage_points for v in row)
        if not all(math.isfinite(float(v)) for v in numeric) or self.energy_reference <= 0:
            raise ValueError("Physical inputs must be finite with positive reference scale")
        if any(v < 0 for v in self.road_distance.values()) or any(
            not isinstance(v,int) or isinstance(v,bool) or v <= 0 for v in self.road_slots.values()):
            raise ValueError("Road distances must be nonnegative and travel slots positive integers")
        if not isinstance(self.horizon,int) or isinstance(self.horizon,bool):
            raise ValueError("The physical horizon is an integer slot count")
        if self.return_deadline is not None and self.return_deadline < self.horizon:
            raise ValueError("Safe-return deadline cannot precede the planning horizon")
        if (not self.transfer_curve or any(len(row) != 2 for row in self.transfer_curve) or
            any(d < 0 or not 0 <= eta <= 1 for d,eta in self.transfer_curve) or
            any(self.transfer_curve[j][0] >= self.transfer_curve[j+1][0]
                for j in range(len(self.transfer_curve)-1))):
            raise ValueError("Transfer calibration requires increasing distances and valid fractions")
        if any(len(p) != 3 or p[2] < 0 for p in self.coverage_points):
            raise ValueError("Coverage quadrature needs nonnegative area weights")
        for group in (self.locations, self.chargers, self.recipients, self.tasks):
            ids = [v.id for v in group]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate identifiers")
        locations = {v.id for v in self.locations}
        if not locations or any(c.initial_location not in locations for c in self.chargers):
            raise ValueError("Unknown initial charger location")
        for r in self.recipients:
            if r.location_id not in locations or not 0 <= r.arrival <= r.departure <= self.horizon:
                raise ValueError("Invalid recipient location or availability")
            if r.demand < 0 or r.capacity <= 0 or not 0 <= r.initial_energy <= r.capacity:
                raise ValueError("Invalid recipient energy")
            if r.load and (len(r.load) != self.horizon or min(r.load) < 0):
                raise ValueError("Load energies must be a nonnegative fixed horizon vector")
        recipient_ids={r.id for r in self.recipients}
        charger_ids={c.id for c in self.chargers}
        if set(self.safe_locations)-charger_ids or any(v not in locations for v in self.safe_locations.values()):
            raise ValueError("Safe locations must refer to known chargers and locations")
        if any(a not in recipient_ids or b not in recipient_ids for a,b in self.communication_edges):
            raise ValueError("Communication edge refers to an unknown sensor")
        if any(not p or any(r not in recipient_ids for r in p) for p in self.required_paths):
            raise ValueError("Required paths must contain known sensors")
        if any(set(t.route_ids)-charger_ids or set(t.recipient_ids)-recipient_ids for t in self.tasks):
            raise ValueError("Task refers to an unknown route or recipient")
        if len({r.charger_id for r in self.committed_prefixes}) != len(self.committed_prefixes):
            raise ValueError("Only one committed prefix is allowed per charger")
        if any(c.capacity <= 0 or c.max_power < 0 or c.speed_mps <= 0 or
               not 0 <= c.initial_energy <= c.capacity or not 0 < c.receiver_efficiency <= 1
               for c in self.chargers):
            raise ValueError("Invalid charger physics")
        for name in ('road_distance','road_slots','safe_locations'):
            object.__setattr__(self,name,MappingProxyType(dict(getattr(self,name))))

    @property
    def dt_hours(self) -> float:
        return self.step_seconds / 3600.0

    def distance(self, origin: str, destination: str) -> float:
        if origin == destination:
            return 0.0
        key = f"{origin}|{destination}"
        if key in self.road_distance:
            return float(self.road_distance[key])
        if self.road_distance:
            raise ValueError(f"Unreachable frozen road pair: {key}")
        loc = {p.id: p for p in self.locations}
        return math.hypot(loc[origin].x-loc[destination].x, loc[origin].y-loc[destination].y)

    def travel_slots(self, charger: Charger, origin: str, destination: str) -> int:
        if origin == destination:
            return 1
        key = f"{charger.id}|{origin}|{destination}"
        if key in self.road_slots:
            return int(self.road_slots[key])
        return max(1, math.ceil(self.distance(origin, destination) /
                                (charger.speed_mps * self.step_seconds)))

    def fingerprint(self) -> str:
        payload = json.dumps(to_dict(self), sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def scenario_from_dict(data: dict) -> Scenario:
    fields = dict(data)
    fields["locations"] = tuple(Location(**v) for v in fields["locations"])
    fields["chargers"] = tuple(Charger(**v) for v in fields["chargers"])
    fields["recipients"] = tuple(Recipient(**{**v, "load": tuple(v.get("load", ()))})
                                 for v in fields["recipients"])
    fields["tasks"] = tuple(Task(**{**v,"route_ids":tuple(v["route_ids"]),
                                      "recipient_ids":tuple(v["recipient_ids"])})
                              for v in fields.get("tasks", ()))
    fields['committed_prefixes']=plan_from_dict({'routes':fields.get('committed_prefixes',[]) }).routes
    for key in ("communication_edges", "required_paths", "transfer_curve", "coverage_points"):
        if key in fields:
            fields[key] = tuple(tuple(v) for v in fields[key])
    return Scenario(**fields)


def plan_from_dict(data: dict) -> Plan:
    return Plan(tuple(Route(r["charger_id"], tuple(Arc(**v) for v in r.get("arcs", ())),
                            tuple(Service(**v) for v in r.get("services", ())),
                            tuple(Power(**v) for v in r.get("powers", ())))
                      for r in data["routes"]))
