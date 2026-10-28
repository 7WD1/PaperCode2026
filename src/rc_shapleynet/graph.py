"""Plan-marked heterogeneous graphs and the paper's relation-wise encoder.

The graph is constructed from the complete plan. Coalition masks never enter
this module. All times use schema slots; raw waiting/slack features use minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .schema import Plan, Scenario
from .physics import effective_route

Relation = tuple[str, str, str]
RELATIONS: tuple[Relation, ...] = (
    ("charger", "compatible", "recipient"),
    ("recipient", "compatible_rev", "charger"),
    ("charger", "serves", "recipient"),
    ("recipient", "served_by", "charger"),
    ("charger", "located_at", "location"),
    ("recipient", "located_at", "location"),
    ("location", "travel", "location"),
    ("time", "continues", "time"),
    ("recipient", "window", "time"),
    ("charger", "scheduled", "time"),
    ("recipient", "communicates", "recipient"),
    ("charger", "shares_task", "charger"),
    ("charger", "power", "time"),
    ("charger", "departs", "location"),
    ("charger", "arrives", "location"),
    ("charger", "dwells", "location"),
    ("recipient", "forecast_load", "time"),
)
EDGE_DIM = 4


@dataclass(frozen=True)
class GraphTask:
    id: str
    route_indices: tuple[int, ...]
    recipient_indices: tuple[int, ...]
    # Full-plan committed-service-weighted, clipped deadline urgency per route.
    urgencies: tuple[float, ...]


@dataclass(frozen=True)
class PlanGraph:
    node_features: Mapping[str, Tensor]
    edge_index: Mapping[Relation, Tensor]
    edge_features: Mapping[Relation, Tensor]
    route_membership: tuple[Mapping[str, Tensor], ...]
    route_ids: tuple[str, ...]
    tasks: tuple[GraphTask, ...]
    recipient_location_ids: Tensor
    reference_id: str
    route_commitments: Tensor | None = None
    recipient_deadline_urgency: Tensor | None = None

    @property
    def input_dims(self) -> dict[str, int]:
        return {key: value.shape[1] for key, value in self.node_features.items()}

    def to(self, device: str | torch.device) -> PlanGraph:
        return replace(
            self,
            node_features={k: v.to(device) for k, v in self.node_features.items()},
            edge_index={k: v.to(device) for k, v in self.edge_index.items()},
            edge_features={k: v.to(device) for k, v in self.edge_features.items()},
            route_membership=tuple({k: v.to(device) for k, v in r.items()}
                                   for r in self.route_membership),
            recipient_location_ids=self.recipient_location_ids.to(device),
            route_commitments=(self.route_commitments.to(device)
                               if self.route_commitments is not None else None),
            recipient_deadline_urgency=(self.recipient_deadline_urgency.to(device)
                                       if self.recipient_deadline_urgency is not None else None),
        )


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-type training-set moments. Fit once on training graphs and freeze."""
    means: Mapping[str, Tensor]
    scales: Mapping[str, Tensor]
    edge_means: Mapping[Relation, Tensor]
    edge_scales: Mapping[Relation, Tensor]

    @classmethod
    def fit(cls, training_graphs: list[PlanGraph]) -> FeatureNormalizer:
        if not training_graphs:
            raise ValueError("Normalization requires nonempty training graphs")
        means, scales = {}, {}
        for kind in training_graphs[0].node_features:
            values = torch.cat([g.node_features[kind].detach().cpu()
                                for g in training_graphs], dim=0)
            if values.shape[0]:
                means[kind] = values.mean(0)
                scale = values.std(0, unbiased=False)
                scales[kind] = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            else:
                means[kind] = torch.zeros(values.shape[1])
                scales[kind] = torch.ones(values.shape[1])
        edge_means, edge_scales = {}, {}
        for relation in training_graphs[0].edge_features:
            values = torch.cat([g.edge_features[relation].detach().cpu()
                                for g in training_graphs], dim=0)
            if values.shape[0]:
                edge_means[relation] = values.mean(0)
                scale = values.std(0, unbiased=False)
                edge_scales[relation] = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            else:
                edge_means[relation] = torch.zeros(values.shape[1])
                edge_scales[relation] = torch.ones(values.shape[1])
        return cls(means, scales, edge_means, edge_scales)

    def transform(self, graph: PlanGraph) -> PlanGraph:
        features = {}
        for kind, values in graph.node_features.items():
            if kind not in self.means or self.means[kind].numel() != values.shape[1]:
                raise ValueError(f"Normalizer has no matching type: {kind}")
            features[kind] = ((values - self.means[kind].to(values)) /
                              self.scales[kind].to(values))
        edge_features = {relation: ((values-self.edge_means[relation].to(values)) /
                                    self.edge_scales[relation].to(values))
                         for relation, values in graph.edge_features.items()}
        return replace(graph, node_features=features, edge_features=edge_features)

    def state_dict(self) -> dict:
        return {"means": dict(self.means), "scales": dict(self.scales),
                "edge_means": dict(self.edge_means), "edge_scales": dict(self.edge_scales)}

    @classmethod
    def from_state_dict(cls, state: dict) -> FeatureNormalizer:
        return cls(state["means"], state["scales"], state["edge_means"], state["edge_scales"])


def _matrix(rows: list, width: int) -> Tensor:
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, width)


def _route_arc_runs(route) -> list[tuple[str, str, int, int]]:
    """Coalesce adjacent waits without collapsing any directed travel arc."""
    runs = []
    for arc in sorted(route.arcs, key=lambda a: (a.start, a.end)):
        waiting = arc.origin == arc.destination
        if (runs and waiting and runs[-1][0] == arc.origin
                and runs[-1][1] == arc.destination and runs[-1][3] == arc.start):
            old = runs[-1]
            runs[-1] = (old[0], old[1], old[2], arc.end)
        else:
            runs.append((arc.origin, arc.destination, arc.start, arc.end))
    return runs


def _route_event_times(route) -> set[int]:
    """Retain services, power intervals, and directed travel/wait boundaries."""
    events = {t for _, _, start, end in _route_arc_runs(route) for t in (start, end)}
    events.update(s.slot for s in route.services)
    for power in route.powers:
        if power.output_energy != 0 or power.depot_energy != 0:
            events.update((power.slot, power.slot+1))
    return events


def _load_runs(load: tuple[float, ...]) -> list[tuple[float, int, int]]:
    """Lossless constant-interval compression of a frozen forecast load vector."""
    runs = []
    for slot, value in enumerate(load):
        if runs and value == runs[-1][0]:
            old = runs[-1]
            runs[-1] = (value, old[1], slot+1)
        else:
            runs.append((value, slot, slot+1))
    return [run for run in runs if run[0] != 0]


def build_plan_graph(scenario: Scenario, plan: Plan, *,
                     normalizer: FeatureNormalizer | None = None) -> PlanGraph:
    """Construct all four node types from one frozen, unmasked plan.

    Explicit scenario tasks retain distant dependencies. Shared recipients also
    induce task links when the input does not name such a task. Resource-only
    dependencies can be supplied as schema ``Task`` objects with no recipients.
    ``location_index`` uses a dataset-wide training-frozen vocabulary.
    """
    if not scenario.chargers:
        raise ValueError("A route game needs at least one charger")
    locations = {v.id: v for v in scenario.locations}
    ci = {v.id: i for i, v in enumerate(scenario.chargers)}
    ri = {v.id: i for i, v in enumerate(scenario.recipients)}
    li = {v.id: i for i, v in enumerate(scenario.locations)}
    if any(r.charger_id not in ci for r in plan.routes):
        raise ValueError("Plan contains an unknown charger")
    if any(s.recipient_id not in ri for r in plan.routes for s in r.services):
        raise ValueError("Plan contains an unknown recipient")
    # Use the same unavoidable q-prefix/idle expansion as the physical checker.
    routes = [effective_route(scenario, plan, c.id) for c in scenario.chargers]
    event_times = {0, scenario.horizon}
    for recipient in scenario.recipients:
        event_times.update((recipient.arrival, recipient.departure))
        if scenario.domain == "WRSN":
            for _, start, end in _load_runs(recipient.load):
                event_times.update((start, end))
    for route in routes:
        event_times.update(_route_event_times(route))
    if min(event_times) < 0 or max(event_times) > scenario.horizon:
        raise ValueError("Plan event lies outside the reference horizon")
    times = sorted(event_times)
    ti = {t: i for i, t in enumerate(times)}
    def clock_phase(slot: int) -> float:
        local_seconds = (scenario.clock_offset_seconds + slot*scenario.step_seconds) % 86400
        return 2*math.pi*local_seconds/86400

    recipient_rows = []
    for r in scenario.recipients:
        rp = locations[r.location_id]
        nearest = min(scenario.chargers, key=lambda c: math.hypot(
            locations[c.initial_location].x-rp.x,
            locations[c.initial_location].y-rp.y))
        cp = locations[nearest.initial_location]
        own = routes[ci[nearest.id]]
        last = max((s.slot for route in routes for s in route.services
                    if s.recipient_id == r.id), default=r.arrival)
        busy = sum(s.slot == 0 for s in own.services) / max(1, nearest.ports)
        arrival_fraction = (r.raw_arrival_fraction if r.raw_arrival_fraction is not None
                            else r.arrival/scenario.horizon)
        declared_stay = (r.declared_stay_seconds if r.declared_stay_seconds is not None
                         else (r.departure-r.arrival)*scenario.step_seconds)
        arrival_phase = (2*math.pi*r.arrival_clock_seconds/86400
                         if r.arrival_clock_seconds is not None else clock_phase(r.arrival))
        recipient_rows.append([
            arrival_fraction, declared_stay/60,
            math.sin(arrival_phase), math.cos(arrival_phase),
            (r.demand if scenario.domain == "EV" else sum(r.load)),
            (r.initial_soc_proxy if scenario.domain == "EV"
             else r.initial_energy/r.capacity),
            float(r.source_proxy), float(r.weekday),
            math.hypot(cp.x-rp.x, cp.y-rp.y),
            nearest.initial_energy/nearest.capacity,
            float(busy > 0), (r.departure-last)*scenario.step_seconds/60,
        ])
    node_features = {
        "recipient": _matrix(recipient_rows, 12),
        "charger": _matrix([
            [c.initial_energy/c.capacity, c.capacity, c.max_power,
             c.reserve_energy/c.capacity, c.speed_mps, float(c.ports)]
            for c in scenario.chargers], 6),
        "location": _matrix([[v.x, v.y, float(v.depot)]
                             for v in scenario.locations], 3),
        "time": _matrix([[t/scenario.horizon,
                          math.sin(clock_phase(t)),
                          math.cos(clock_phase(t))] for t in times], 3),
    }
    edges: dict[Relation, list[tuple[int, int]]] = {k: [] for k in RELATIONS}
    edge_values: dict[Relation, list[list[float]]] = {k: [] for k in RELATIONS}

    def add(relation, source, target, features):
        edges[relation].append((source, target))
        edge_values[relation].append(features)

    for c in scenario.chargers:
        add(("charger", "located_at", "location"), ci[c.id], li[c.initial_location],
            [0, 0, 1, 0])
        for r in scenario.recipients:
            try:
                distance = scenario.distance(c.initial_location, r.location_id)
                duration = scenario.travel_slots(c, c.initial_location, r.location_id)
            except ValueError:  # unreachable in the frozen road graph
                continue
            compatible = float(duration < r.departure and c.max_power > 0)
            features = [distance, duration/scenario.horizon, compatible, 0]
            add(("charger", "compatible", "recipient"), ci[c.id], ri[r.id], features)
            add(("recipient", "compatible_rev", "charger"), ri[r.id], ci[c.id], features)
    for r in scenario.recipients:
        add(("recipient", "located_at", "location"), ri[r.id], li[r.location_id],
            [0, 0, 1, 0])
        for t in sorted({r.arrival, r.departure}):
            add(("recipient", "window", "time"), ri[r.id], ti[t],
                [0, t/scenario.horizon, 1, 0])
        if scenario.domain == "WRSN":
            for value, start, end in _load_runs(r.load):
                add(("recipient", "forecast_load", "time"), ri[r.id], ti[start],
                    [value, start/scenario.horizon, end/scenario.horizon, 1])
    selected_arcs = {(a.origin, a.destination) for route in routes for a in route.arcs}
    for origin in scenario.locations:
        for target in scenario.locations:
            if origin.id == target.id:
                continue
            try:
                distance = scenario.distance(origin.id, target.id)
            except ValueError:
                continue
            add(("location", "travel", "location"), li[origin.id], li[target.id],
                [distance, 0, 1, float((origin.id, target.id) in selected_arcs)])
    for a, b in zip(times[:-1], times[1:]):
        add(("time", "continues", "time"), ti[a], ti[b],
            [0, (b-a)/scenario.horizon, 1, 0])
    route_membership = []
    for i, route in enumerate(routes):
        recipients = sorted({ri[s.recipient_id] for s in route.services})
        places = {li[scenario.chargers[i].initial_location]}
        events = {0} | _route_event_times(route)
        for arc in route.arcs:
            places.update((li[arc.origin], li[arc.destination]))
        charger = scenario.chargers[i]
        for origin, destination, start, end in _route_arc_runs(route):
            energy = ((scenario.distance(origin, destination)*charger.energy_per_metre
                       if origin != destination else 0.0)
                      + (end-start)*scenario.dt_hours*charger.auxiliary_kw)
            features = [start/scenario.horizon, end/scenario.horizon, energy, 1]
            if origin == destination:
                add(("charger", "dwells", "location"), i, li[origin], features)
            else:
                add(("charger", "departs", "location"), i, li[origin], features)
                add(("charger", "arrives", "location"), i, li[destination], features)
        power_by_slot: dict[int, list[float]] = {}
        for power in route.powers:
            values = power_by_slot.setdefault(power.slot, [0.0, 0.0])
            values[0] += power.output_energy
            values[1] += power.depot_energy
        for slot, (output, depot) in sorted(power_by_slot.items()):
            if output or depot:
                add(("charger", "power", "time"), i, ti[slot],
                    [output, depot, slot/scenario.horizon, 1])
        for service in route.services:
            r = scenario.recipients[ri[service.recipient_id]]
            places.add(li[r.location_id])
            events.add(service.slot)
            features = [service.energy, service.slot/scenario.horizon, 1, 1]
            add(("charger", "serves", "recipient"), i, ri[r.id], features)
            add(("recipient", "served_by", "charger"), ri[r.id], i, features)
        for t in sorted(events):
            add(("charger", "scheduled", "time"), i, ti[t],
                [0, t/scenario.horizon, 1, 1])
        route_membership.append({
            "charger": torch.tensor([i], dtype=torch.long),
            "recipient": torch.tensor(recipients, dtype=torch.long),
            "location": torch.tensor(sorted(places), dtype=torch.long),
            "time": torch.tensor([ti[t] for t in sorted(events)], dtype=torch.long),
        })
    for a, b in scenario.communication_edges:
        if a not in ri or b not in ri:
            raise ValueError("Unknown communication endpoint")
        for source, target in ((a, b), (b, a)):
            add(("recipient", "communicates", "recipient"), ri[source], ri[target],
                [0, 0, 1, 0])

    named = [(t.id, t.route_ids, t.recipient_ids) for t in scenario.tasks]
    for r in scenario.recipients:
        serving = tuple(scenario.chargers[i].id for i, route in enumerate(routes)
                        if any(s.recipient_id == r.id and s.energy > 0 for s in route.services))
        if serving and not any(set(serving) <= set(t.route_ids)
                                       and r.id in t.recipient_ids for t in scenario.tasks):
            named.append((f"shared-recipient:{r.id}", serving, (r.id,)))
    tasks = []
    for name, route_ids, recipient_ids in named:
        if any(v not in ci for v in route_ids) or any(v not in ri for v in recipient_ids):
            raise ValueError("Task refers to an unknown route or recipient")
        members = tuple(sorted({ci[v] for v in route_ids}))
        recipients = tuple(sorted({ri[v] for v in recipient_ids}))
        urgency = []
        for i, route in enumerate(routes):
            contributions = [(s.energy, 1-min(1, max(0,
                scenario.recipients[ri[s.recipient_id]].departure/scenario.horizon)))
                for s in route.services if ri[s.recipient_id] in recipients and s.energy > 0]
            total = sum(v[0] for v in contributions)
            urgency.append(sum(w*u for w, u in contributions)/total if total else 0.0)
        tasks.append(GraphTask(name, members, recipients, tuple(urgency)))
        for a in members:
            for b in members:
                if a != b:
                    add(("charger", "shares_task", "charger"), a, b, [0, 0, 1, 1])
    commitments = torch.zeros(len(routes), len(scenario.recipients))
    for i, route in enumerate(routes):
        for service in route.services:
            recipient = scenario.recipients[ri[service.recipient_id]]
            if recipient.arrival <= service.slot < recipient.departure:
                commitments[i, ri[recipient.id]] += service.energy
    graph = PlanGraph(
        node_features,
        {k: torch.tensor(v, dtype=torch.long).reshape(-1, 2).T.contiguous()
         for k, v in edges.items()},
        {k: _matrix(v, EDGE_DIM) for k, v in edge_values.items()},
        tuple(route_membership), tuple(c.id for c in scenario.chargers), tuple(tasks),
        torch.tensor([r.location_index for r in scenario.recipients], dtype=torch.long),
        scenario.fingerprint(),
        commitments,
        torch.tensor([1-min(1, max(0, r.departure/scenario.horizon))
                      for r in scenario.recipients], dtype=torch.float32),
    )
    if graph.recipient_location_ids.numel() and graph.recipient_location_ids.min() < 0:
        raise ValueError("Location embedding indices must be nonnegative")
    return normalizer.transform(graph) if normalizer is not None else graph


class RecipientFeatureEncoder(nn.Module):
    """Twelve numerical fields plus one ID become 28 dense dimensions."""
    def __init__(self, num_locations: int, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_locations, embedding_dim)
        self.output_dim = 12 + embedding_dim

    def forward(self, numerical: Tensor, location_ids: Tensor) -> Tensor:
        if numerical.ndim != 2 or numerical.shape[1] != 12:
            raise ValueError("Recipient numerical input must have width twelve")
        if location_ids.dtype != torch.long or location_ids.shape != numerical.shape[:1]:
            raise ValueError("One int64 location index is required per recipient")
        return torch.cat((numerical, self.embedding(location_ids)), dim=-1)


class RelationalMessageLayer(nn.Module):
    """Mean per incoming relation, then relation sum plus typed self state."""
    def __init__(self, node_types: tuple[str, ...], relations: tuple[Relation, ...],
                 hidden_dim: int, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.relations = relations
        self.self_maps = nn.ModuleDict({kind: nn.Linear(hidden_dim, hidden_dim)
                                        for kind in node_types})
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2*hidden_dim+edge_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim)) for _ in relations])

    def forward(self, states: Mapping[str, Tensor], graph: PlanGraph) -> dict[str, Tensor]:
        updates = {kind: layer(states[kind]) for kind, layer in self.self_maps.items()}
        for relation, message in zip(self.relations, self.messages):
            source_type, _, target_type = relation
            source, target = graph.edge_index[relation]
            if not source.numel():
                continue
            values = message(torch.cat((states[source_type][source],
                                        states[target_type][target],
                                        graph.edge_features[relation]), dim=-1))
            aggregate = torch.zeros_like(states[target_type])
            aggregate.index_add_(0, target, values)
            count = values.new_zeros((states[target_type].shape[0], 1))
            count.index_add_(0, target, values.new_ones((target.numel(), 1)))
            updates[target_type] = updates[target_type] + aggregate/count.clamp_min(1)
        return {kind: torch.relu(value) for kind, value in updates.items()}


class TypedPlanEncoder(nn.Module):
    def __init__(self, input_dims: Mapping[str, int], relations: tuple[Relation, ...],
                 num_locations: int, hidden_dim: int = 128, layers: int = 2,
                 embedding_dim: int = 16, edge_dim: int = EDGE_DIM):
        super().__init__()
        if input_dims.get("recipient") != 12 or layers < 1:
            raise ValueError("Twelve scalar fields and at least one message layer are required")
        self.recipient_features = RecipientFeatureEncoder(num_locations, embedding_dim)
        self.projections = nn.ModuleDict({
            k: nn.Linear(self.recipient_features.output_dim if k == "recipient" else width,
                         hidden_dim) for k, width in input_dims.items()})
        self.message_layers = nn.ModuleList([
            RelationalMessageLayer(tuple(input_dims), relations, hidden_dim, edge_dim)
            for _ in range(layers)])

    def forward(self, graph: PlanGraph) -> tuple[Tensor, dict[str, Tensor]]:
        values = dict(graph.node_features)
        values["recipient"] = self.recipient_features(values["recipient"],
                                                       graph.recipient_location_ids)
        states = {k: torch.relu(self.projections[k](v)) for k, v in values.items()}
        for layer in self.message_layers:
            states = layer(states, graph)
        routes = []
        for membership in graph.route_membership:
            marked = [states[k][indices] for k, indices in membership.items() if indices.numel()]
            routes.append(torch.cat(marked, dim=0).mean(dim=0))
        return torch.stack(routes), states
