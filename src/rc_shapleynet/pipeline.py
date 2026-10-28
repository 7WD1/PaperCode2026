"""Offline physical labels for forecast-context graphs and cached trajectories.

Each supplied trajectory is evaluated explicitly. The default has one trajectory;
the function never duplicates observations to emulate a larger ensemble.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Sequence

import numpy as np
import torch

from .game import FixedRouteGame, interaction_task
from .graph import FeatureNormalizer, build_plan_graph
from .learning import interaction_support
from .physics import evaluate
from .recovery import probe
from .schema import Plan, Scenario, to_dict
from .training import PlanTrainingExample


class LabelGenerationError(ValueError):
    """A cached trajectory or retained physical plan cannot supply a valid label."""


def mask_stream_seed(simulation_seed: int, mask_index: int) -> int:
    if simulation_seed < 0 or mask_index < 0:
        raise ValueError("Simulation seed and mask index must be nonnegative")
    return simulation_seed*1000+mask_index


def coalition_masks(route_count: int, mask_count: int, simulation_seed: int) -> torch.Tensor:
    """Shared per-index PCG64 streams, with empty and full masks first.

    Nontrivial coalition sizes are sampled uniformly when at least two routes
    exist. Small games may repeat masks; repeated queries are cached physical
    evaluations, not new independent trajectory observations.
    """
    if route_count < 1 or mask_count < 2:
        raise ValueError("At least one route and two masks (empty/full) are required")
    masks = np.zeros((mask_count, route_count), dtype=np.float32)
    masks[1] = 1
    for index in range(2, mask_count):
        rng = np.random.Generator(np.random.PCG64(mask_stream_seed(simulation_seed, index)))
        if route_count == 1:
            masks[index, 0] = rng.integers(0, 2)
        else:
            count = int(rng.integers(1, route_count))
            masks[index, rng.choice(route_count, count, replace=False)] = 1
    return torch.from_numpy(masks)


def _fixed_payload(scenario: Scenario) -> dict:
    """Only the explicitly cached exogenous loads and identifying seed may vary."""
    payload = to_dict(scenario)
    payload.pop("simulation_seed")
    payload.pop("scenario_id")
    for recipient in payload["recipients"]:
        recipient.pop("load")
    return payload


def _cached_trajectories(context: Scenario,
                          trajectories: Sequence[Scenario] | None) -> tuple[Scenario, ...]:
    cached = (context,) if trajectories is None else tuple(trajectories)
    if not cached:
        raise LabelGenerationError("An explicitly supplied trajectory group cannot be empty")
    reference = _fixed_payload(context)
    fingerprints = []
    for index, scenario in enumerate(cached):
        actual = _fixed_payload(scenario)
        changed = [key for key, value in reference.items() if actual.get(key) != value]
        if changed:
            raise LabelGenerationError(
                f"Trajectory {index} changes fixed context fields: {', '.join(changed)}")
        fingerprints.append(scenario.fingerprint())
    if len(set(fingerprints)) != len(fingerprints):
        raise LabelGenerationError("Duplicate cached trajectory fingerprints cannot count as independent traces")
    return cached


def label_reference_id(context: Scenario,
                        trajectories: Sequence[Scenario] | None = None) -> str:
    """Identify a shared forecast reference and its actual cached ensemble.

    Candidate plans are intentionally absent. Permuting the same cached group
    leaves the reference unchanged.
    """
    cached = _cached_trajectories(context, trajectories)
    payload = {"context": context.fingerprint(),
               "trajectories": sorted(s.fingerprint() for s in cached)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def label_plan(context: Scenario, plan: Plan, *, mask_count: int = 16,
               trajectories: Sequence[Scenario] | None = None, order: int = 2,
               normalizer: FeatureNormalizer | None = None) -> PlanTrainingExample:
    """Generate coalition and recovery labels under one complete-route reference.

    ``context`` supplies forecast features only. ``trajectories`` supplies actual
    cached external loads and may contain one or eight explicitly provided traces.
    All initial states, requests, resources, roads, transfer rules, clocks, and
    physical assumptions remain fixed. Every sampled retained plan must pass the
    original checker in every trace; failure raises a diagnostic exception.

    Recovery repairs are confined to the members of each supported interaction,
    at their existing dwell slots. Other commitments are retained by ``probe``.
    Labels average restored service before dividing by the fixed commitment.
    """
    cached = _cached_trajectories(context, trajectories)
    reference_id = label_reference_id(context, cached)
    graph = replace(build_plan_graph(context, plan), reference_id=reference_id)
    masks = coalition_masks(len(graph.route_ids), mask_count, context.simulation_seed)
    physical_values = np.zeros(mask_count, dtype=np.float64)
    full_objectives = []
    for index, scenario in enumerate(cached):
        def checked_evaluator(actual, retained):
            result = evaluate(actual, retained)
            if not result.feasible:
                names = ",".join(r.charger_id for r in retained.routes) or "empty"
                raise LabelGenerationError(
                    f"Trajectory {index}, retained routes {names}: " + "; ".join(result.violations))
            return result

        game = FixedRouteGame(scenario, plan, evaluator=checked_evaluator)
        full_objectives.append(game(game.players)+game.baseline)
        for mask_index, mask in enumerate(masks):
            members = tuple(route for route, present in zip(graph.route_ids, mask.tolist()) if present)
            physical_values[mask_index] += game(members)/len(cached)

    recovery_labels = {}
    for support in interaction_support(graph, order):
        members = tuple(graph.route_ids[i] for i in support)
        task = interaction_task(context, plan, members)
        if task is None:
            continue  # There is no committed-service recovery target.
        for removed in members:
            results = []
            for index, scenario in enumerate(cached):
                try:
                    result = probe(scenario, plan, task, removed)
                except ValueError as error:
                    raise LabelGenerationError(
                        f"Trajectory {index}, recovery of {removed} in {members}: {error}") from error
                results.append(result)
            committed = results[0].committed_total
            if any(not math.isclose(result.committed_total, committed,
                                    rel_tol=1e-10, abs_tol=1e-12) for result in results):
                raise LabelGenerationError("Cached trajectories changed the fixed service commitment")
            restored_mean = sum(result.restored_total for result in results)/len(results)
            difficulty = 1-restored_mean/committed if committed > 0 else 0.0
            recovery_labels[members, removed] = max(0.0, min(1.0, difficulty))
    if normalizer is not None:
        graph = normalizer.transform(graph)
    return PlanTrainingExample(
        graph, masks, torch.tensor(physical_values, dtype=torch.float32), recovery_labels,
        reference_id, float(np.mean(full_objectives)),
    )


def label_plans(context: Scenario, plans: Sequence[Plan], *, mask_count: int = 16,
                trajectories: Sequence[Scenario] | None = None, order: int = 2,
                normalizer: FeatureNormalizer | None = None) -> tuple[PlanTrainingExample, ...]:
    """Label a candidate pool with exactly the same masks and cached trajectories."""
    cached = _cached_trajectories(context, trajectories)
    return tuple(label_plan(context, plan, mask_count=mask_count, trajectories=cached,
                            order=order, normalizer=normalizer) for plan in plans)
