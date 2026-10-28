"""Signed interaction prediction and fixed-context recovery allocation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import torch
from torch import Tensor, nn

from .graph import EDGE_DIM, PlanGraph, Relation, TypedPlanEncoder


def interaction_support(graph: PlanGraph, order: int = 2) -> tuple[tuple[int, ...], ...]:
    """All singletons plus task/resource-linked sets, without a distance cutoff."""
    if order < 1:
        raise ValueError("Interaction order must be positive")
    support = {(i,) for i in range(len(graph.route_ids))}
    for task in graph.tasks:
        for size in range(2, min(order, len(task.route_indices))+1):
            support.update(combinations(sorted(task.route_indices), size))
    return tuple(sorted(support, key=lambda value: (len(value), value)))


@dataclass(frozen=True)
class EncodedPlan:
    """One encoding of P; all coalition and allocation queries reuse it.

    ``support`` indexes ``route_ids``. Recovery, urgency, and share matrices have
    shape [interactions, routes], with zero entries outside each interaction.
    """
    coefficients: Tensor
    support: tuple[tuple[int, ...], ...]
    recovery_predictions: Tensor
    urgencies: Tensor
    membership: Tensor
    task_associated: Tensor
    route_ids: tuple[str, ...]
    reference_id: str

    def values(self, masks: Tensor) -> Tensor:
        masks = torch.as_tensor(masks, dtype=self.coefficients.dtype,
                                device=self.coefficients.device)
        single = masks.ndim == 1
        if single:
            masks = masks.unsqueeze(0)
        if masks.ndim != 2 or masks.shape[1] != len(self.route_ids):
            raise ValueError("Coalition masks must have one entry per route")
        if not torch.all((masks == 0) | (masks == 1)):
            raise ValueError("Coalition interventions require binary masks")
        products = torch.where(self.membership.unsqueeze(0), masks.unsqueeze(1),
                               torch.ones_like(masks).unsqueeze(1)).prod(dim=-1)
        result = products @ self.coefficients
        return result[0] if single else result

    @property
    def full_value(self) -> Tensor:
        return self.coefficients.sum()

    def shares(self, *, lambda_: float = .5, tau: float = .5,
               eta: float = .5) -> Tensor:
        if not 0 <= lambda_ <= 1 or tau <= 0 or eta < 0:
            raise ValueError("Require lambda in [0,1], tau>0, eta>=0")
        uniform = self.membership.to(self.coefficients.dtype)
        uniform = uniform/uniform.sum(dim=1, keepdim=True)
        scores = self.recovery_predictions + eta*self.urgencies
        scores = (scores/tau).masked_fill(~self.membership, -torch.inf)
        modified = (1-lambda_)*uniform + lambda_*torch.softmax(scores, dim=1)
        return torch.where(self.task_associated[:, None], modified, uniform)

    def allocation(self, *, lambda_: float = .5, tau: float = .5,
                   eta: float = .5) -> Tensor:
        return (self.shares(lambda_=lambda_, tau=tau, eta=eta)
                * self.coefficients[:, None]).sum(dim=0)

    @property
    def recovery_mask(self) -> Tensor:
        return self.membership & self.task_associated[:, None]


class RCShapleyNet(nn.Module):
    """Typed encoder, permutation-invariant signed head, and separate κ head."""
    def __init__(self, input_dims: Mapping[str, int], relations: tuple[Relation, ...],
                 num_locations: int, *, hidden_dim: int = 128, layers: int = 2,
                 order: int = 2, embedding_dim: int = 16, edge_dim: int = EDGE_DIM):
        super().__init__()
        if num_locations < 1 or hidden_dim < 1 or order < 1:
            raise ValueError("Invalid model dimensions")
        relations = tuple(tuple(v) for v in relations)
        self.configuration = {
            "input_dims": dict(input_dims), "relations": relations,
            "num_locations": num_locations, "hidden_dim": hidden_dim,
            "layers": layers, "order": order, "embedding_dim": embedding_dim,
            "edge_dim": edge_dim,
        }
        self.order = order
        self.encoder = TypedPlanEncoder(input_dims, relations, num_locations,
                                        hidden_dim, layers, embedding_dim, edge_dim)
        self.coefficient_head = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.recovery_head = nn.Sequential(
            nn.Linear(3*hidden_dim+1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward_plan(self, graph: PlanGraph) -> EncodedPlan:
        route_states, node_states = self.encoder(graph)
        support = interaction_support(graph, self.order)
        count = len(graph.route_ids)
        coefficients, recovery_rows, urgency_rows, membership_rows, associated = [], [], [], [], []
        for members in support:
            indices = torch.tensor(members, dtype=torch.long, device=route_states.device)
            pooled = route_states[indices].mean(dim=0)
            tasks = [task for task in graph.tasks if set(members) <= set(task.route_indices)]
            task_recipient_indices = sorted({r for task in tasks for r in task.recipient_indices})
            if task_recipient_indices:
                task_state = node_states["recipient"][task_recipient_indices].mean(dim=0)
            else:
                task_state = torch.zeros_like(pooled)
            cardinality = pooled.new_tensor([len(members)/self.order])
            context = torch.cat((pooled, task_state, cardinality))
            coefficients.append(self.coefficient_head(context).squeeze(-1))
            # The task head's pooled route context is symmetric; member identity
            # enters only the independent recovery head for that member.
            member_context = torch.cat((route_states[indices],
                pooled.expand(len(members), -1), task_state.expand(len(members), -1),
                cardinality.expand(len(members), 1)), dim=-1)
            predicted = self.recovery_head(member_context).squeeze(-1)
            row = route_states.new_zeros(count).scatter(0, indices, predicted)
            recovery_rows.append(row)
            mask = torch.zeros(count, dtype=torch.bool, device=route_states.device)
            mask[indices] = True
            membership_rows.append(mask)
            urgency = route_states.new_zeros(count)
            if tasks:
                for i in members:
                    if graph.route_commitments is not None and task_recipient_indices:
                        weights = graph.route_commitments[i, task_recipient_indices]
                        total = weights.sum()
                        if total > 0:
                            urgency[i] = (weights*graph.recipient_deadline_urgency[
                                task_recipient_indices]).sum()/total
                    else:
                        urgency[i] = sum(task.urgencies[i] for task in tasks)/len(tasks)
            urgency_rows.append(urgency)
            # A resource-only link retains a value coefficient but has no
            # committed-service recovery task, matching the physical game.
            associated.append(bool(task_recipient_indices))
        return EncodedPlan(
            torch.stack(coefficients), support, torch.stack(recovery_rows),
            torch.stack(urgency_rows), torch.stack(membership_rows),
            torch.tensor(associated, dtype=torch.bool, device=route_states.device),
            graph.route_ids, graph.reference_id,
        )

    def forward(self, graph: PlanGraph, masks: Tensor) -> Tensor:
        """Convenience query; use forward_plan once for repeated coalition queries."""
        return self.forward_plan(graph).values(masks)
