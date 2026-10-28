"""Optional real SUMO/TraCI transport and immutable common exogenous traces.

The physical energy checker remains in :mod:`physics`. A frozen network profile
is a documented free-flow routing input, not an observed charging benchmark.
API references: https://eclipse.dev/sumo/docs/Tools/Sumolib.html and
https://eclipse.dev/sumo/docs/TraCI/Interfacing_TraCI_from_Python.html.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any
from types import MappingProxyType
import hashlib
import importlib
import json
import math
import uuid

from .schema import Charger


@dataclass(frozen=True)
class RoadPosition:
    edge_id: str
    position: float = 0.0


@dataclass(frozen=True)
class FrozenRoadProfile:
    road_distance: Mapping[str, float]
    road_slots: Mapping[str, int]
    edge_paths: Mapping[str, tuple[str, ...]]
    network_sha256: str
    routing_basis: str

    def __post_init__(self):
        object.__setattr__(self, "road_distance", MappingProxyType(dict(self.road_distance)))
        object.__setattr__(self, "road_slots", MappingProxyType(dict(self.road_slots)))
        object.__setattr__(self, "edge_paths", MappingProxyType({k: tuple(v) for k, v in self.edge_paths.items()}))

    def save(self, path: str | Path) -> None:
        payload = {"road_distance": dict(self.road_distance), "road_slots": dict(self.road_slots),
                   "edge_paths": dict(self.edge_paths), "network_sha256": self.network_sha256,
                   "routing_basis": self.routing_basis}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenRoadProfile":
        fields = json.loads(Path(path).read_text(encoding="utf-8"))
        fields["edge_paths"] = {k: tuple(v) for k, v in fields["edge_paths"].items()}
        return cls(**fields)


def freeze_network(net_path: str | Path, positions: Mapping[str, RoadPosition],
                   chargers: tuple[Charger, ...], step_seconds: float,
                   vehicle_class: str = "passenger") -> FrozenRoadProfile:
    """Read a SUMO network and freeze directed station-to-station road profiles.

    SUMO's free-flow fastest path fixes the route. Travel time follows that route
    with each edge's speed capped by the charger speed. Thus these times do not
    claim to include realized congestion. Internal junction edges are requested.
    The same profile must be reused for every coalition of one reference.
    """
    if step_seconds <= 0:
        raise ValueError("Physical step must be positive")
    try:
        sumolib = importlib.import_module("sumolib")
    except ImportError as exc:
        raise ImportError("Install the SUMO optional dependencies to read a network") from exc
    net_path = Path(net_path)
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    distances: dict[str, float] = {}
    durations: dict[str, int] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for oid, origin in positions.items():
        edge_o = net.getEdge(origin.edge_id)
        if not 0 <= origin.position <= edge_o.getLength():
            raise ValueError(f"Origin position outside SUMO edge: {oid}")
        for did, destination in positions.items():
            if oid == did:
                continue
            edge_d = net.getEdge(destination.edge_id)
            if not 0 <= destination.position <= edge_d.getLength():
                raise ValueError(f"Destination position outside SUMO edge: {did}")
            path, _ = net.getFastestPath(edge_o, edge_d, vClass=vehicle_class,
                                        fromPos=origin.position, toPos=destination.position,
                                        withInternal=True)
            if not path:
                continue  # Absent pairs are unreachable; no Euclidean substitution.
            lengths = [float(edge.getLength()) for edge in path]
            lengths[0] -= origin.position
            lengths[-1] -= edge_d.getLength()-destination.position
            if any(length < -1e-8 for length in lengths):
                # A backwards station pair on one edge needs an explicit loop;
                # it is not replaced by a physically invalid negative segment.
                continue
            lengths = [max(0.0, value) for value in lengths]
            key = f"{oid}|{did}"
            distances[key] = sum(lengths)
            paths[key] = tuple(edge.getID() for edge in path)
            for charger in chargers:
                duration = sum(length/min(charger.speed_mps, float(edge.getSpeed()))
                               for length, edge in zip(lengths, path))
                durations[f"{charger.id}|{oid}|{did}"] = max(1, math.ceil(duration/step_seconds))
    if len(positions) > 1 and not distances:
        raise ValueError("No directed station pair is reachable in the SUMO network")
    return FrozenRoadProfile(distances, durations, paths,
                             hashlib.sha256(net_path.read_bytes()).hexdigest(),
                             "SUMO free-flow fastest path; edge speeds capped by charger speed")


@dataclass(frozen=True)
class FrozenTrace:
    """Cached snapshots from one actual run, shared without traffic reruns."""
    seed: int
    version: str
    snapshots: tuple[dict[str, Any], ...]

    def __post_init__(self):
        object.__setattr__(self, "snapshots", tuple(_freeze(row) for row in self.snapshots))

    def save(self, path: str | Path) -> None:
        payload = {"seed": self.seed, "version": self.version, "snapshots": _thaw(self.snapshots)}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenTrace":
        fields = json.loads(Path(path).read_text(encoding="utf-8"))
        fields["snapshots"] = tuple(fields["snapshots"])
        return cls(**fields)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


class SUMOAdapter:
    """Own a real libsumo process or a named TraCI connection.

    ``step`` delegates to SUMO; it never synthesizes a vehicle position. Captured
    background traces can subsequently be shared by fixed-reference evaluations.
    """
    _libsumo_active = False

    def __init__(self, config: str | Path, *, backend: str = "libsumo",
                 sumo_binary: str = "sumo", seed: int = 10001,
                 step_seconds: float = 1.0, extra_args: tuple[str, ...] = ()):
        if backend not in ("libsumo", "traci"):
            raise ValueError("Backend must be libsumo or traci")
        if not Path(config).is_file() or step_seconds <= 0:
            raise ValueError("A real SUMO configuration and positive step are required")
        self.config, self.backend, self.sumo_binary = str(Path(config).resolve()), backend, sumo_binary
        self.seed, self.step_seconds, self.extra_args = seed, step_seconds, extra_args
        self._api = None
        self._label = f"rc-{uuid.uuid4().hex}"

    def start(self) -> "SUMOAdapter":
        if self._api is not None:
            raise RuntimeError("SUMO connection is already running")
        try:
            module = importlib.import_module(self.backend)
        except ImportError as exc:
            raise ImportError(f"Install the optional {self.backend} package") from exc
        command = [self.sumo_binary, "-c", self.config, "--seed", str(self.seed),
                   "--step-length", str(self.step_seconds), *self.extra_args]
        if self.backend == "libsumo":
            if SUMOAdapter._libsumo_active:
                raise RuntimeError("libsumo exposes one global simulation; use named TraCI connections for concurrency")
            module.start(command)
            self._api = module
            SUMOAdapter._libsumo_active = True
        else:
            module.start(command, label=self._label)
            self._api = module.getConnection(self._label)
        return self

    @property
    def api(self):
        if self._api is None:
            raise RuntimeError("Start SUMO before using the adapter")
        return self._api

    @property
    def version(self) -> str:
        return str(self.api.getVersion()[1])

    def snapshot(self) -> dict[str, Any]:
        api = self.api
        return {"time": float(api.simulation.getTime()),
                "vehicles": {vid: {"position": tuple(map(float, api.vehicle.getPosition(vid))),
                                    "road_id": api.vehicle.getRoadID(vid),
                                    "lane_position": float(api.vehicle.getLanePosition(vid)),
                                    "speed": float(api.vehicle.getSpeed(vid)),
                                    "distance": float(api.vehicle.getDistance(vid))}
                             for vid in sorted(api.vehicle.getIDList())}}

    def step(self, until: float | None = None) -> dict[str, Any]:
        if until is None:
            self.api.simulationStep()
        else:
            if until <= self.api.simulation.getTime():
                raise ValueError("SUMO target time must advance")
            self.api.simulationStep(float(until))
        return self.snapshot()

    def add_vehicle(self, vehicle_id: str, edges: tuple[str, ...],
                    *, type_id: str = "DEFAULT_VEHTYPE", max_speed: float | None = None) -> None:
        if not edges:
            raise ValueError("A SUMO route needs real edge IDs")
        route_id = f"{self._label}-{vehicle_id}"
        self.api.route.add(route_id, list(edges))
        self.api.vehicle.add(vehicle_id, route_id, typeID=type_id, depart="now")
        if max_speed is not None:
            if max_speed <= 0:
                raise ValueError("Vehicle speed must be positive")
            self.api.vehicle.setMaxSpeed(vehicle_id, float(max_speed))

    def capture(self, steps: int) -> FrozenTrace:
        if steps < 0:
            raise ValueError("Capture length must be nonnegative")
        version = self.version
        return FrozenTrace(self.seed, version, tuple(self.step() for _ in range(steps)))

    def close(self) -> None:
        if self._api is not None:
            try:
                self._api.close()
            finally:
                self._api = None
                if self.backend == "libsumo":
                    SUMOAdapter._libsumo_active = False

    def __enter__(self) -> "SUMOAdapter":
        return self.start()

    def __exit__(self, *args) -> None:
        self.close()
