"""Supervised fixed-plan training with separate recovery targets and same-q ranks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .graph import FeatureNormalizer, PlanGraph
from .learning import EncodedPlan, RCShapleyNet

RecoveryLabels = Mapping[tuple[tuple[str, ...], str], float] | Tensor


@dataclass(frozen=True)
class PlanTrainingExample:
    graph: PlanGraph
    masks: Tensor
    coalition_targets: Tensor
    recovery_targets: RecoveryLabels
    reference_id: str
    physical_full_value: float

    def __post_init__(self):
        if self.reference_id != self.graph.reference_id:
            raise ValueError("Training reference must match the frozen graph reference")
        if (self.masks.ndim != 2 or self.masks.shape[1] != len(self.graph.route_ids)
                or self.coalition_targets.shape != self.masks.shape[:1]):
            raise ValueError("Masks and physical coalition targets have incompatible shapes")
        if not self.masks.shape[0] or not torch.isfinite(self.coalition_targets).all():
            raise ValueError("Finite physical coalition observations are required")
        if not math.isfinite(self.physical_full_value):
            raise ValueError("Ranking requires a finite physical plan value")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    ranking_margin: float = .05
    ranking_fraction: float = .2
    # Zero removes multi-coalition value supervision for the named ablation.
    coalition_weight: float = 1.0
    resource_weight: float = .5
    ranking_weight: float = .5
    sparsity_weight: float = 1e-3
    seed: int = 1
    device: str = "cpu"
    min_delta: float = 0.0

    def __post_init__(self):
        if (self.learning_rate <= 0 or self.weight_decay < 0 or self.batch_size < 1
                or self.epochs < 1 or self.patience < 1 or self.ranking_margin < 0
                or not 0 < self.ranking_fraction <= .5 or self.min_delta < 0
                or min(self.coalition_weight, self.resource_weight,
                       self.ranking_weight, self.sparsity_weight) < 0):
            raise ValueError("Invalid optimizer, ranking, or supervised loss configuration")


@dataclass
class LossTerms:
    total: Tensor
    coalition: Tensor
    resource: Tensor
    ranking: Tensor
    sparse: Tensor

    def detached(self) -> dict[str, float]:
        return {key: float(getattr(self, key).detach().cpu())
                for key in ("total", "coalition", "resource", "ranking", "sparse")}


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_mae: float
    history: list[dict[str, float]]
    stopped_early: bool


def make_ranking_pairs(examples: Sequence[PlanTrainingExample], *,
                       fraction: float = .2, seed: int = 1) -> list[tuple[int, int]]:
    """Randomly pair same-q top/bottom pools, selected by physical J.

    Rounding upward retains one positive and one negative for the paper's
    four-plan pool. The pools are disjoint. Equal physical values yield no pair.
    """
    if not 0 < fraction <= .5:
        raise ValueError("The ranking-pool fraction must lie in (0,.5]")
    groups: dict[str, list[int]] = {}
    for i, example in enumerate(examples):
        groups.setdefault(example.reference_id, []).append(i)
    rng = np.random.Generator(np.random.PCG64(seed))
    pairs = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: examples[i].physical_full_value)
        size = min(len(indices)//2, max(1, math.ceil(len(indices)*fraction)))
        bottom, top = ordered[:size], ordered[-size:]
        for good in top:
            bad = int(rng.choice(bottom))
            if examples[good].physical_full_value > examples[bad].physical_full_value:
                pairs.append((good, bad))
    return pairs


def _recovery_targets(encoded: EncodedPlan, labels: RecoveryLabels) -> Tensor:
    target = torch.full_like(encoded.recovery_predictions, torch.nan)
    if isinstance(labels, Tensor):
        if labels.shape != target.shape:
            raise ValueError("Recovery label tensor must have shape [interactions,routes]")
        return labels.to(target)
    rows = {frozenset(encoded.route_ids[i] for i in members): j
            for j, members in enumerate(encoded.support)}
    columns = {name: i for i, name in enumerate(encoded.route_ids)}
    for (members, route), value in labels.items():
        key = frozenset(members)
        if key not in rows or route not in key:
            raise ValueError("Recovery label does not refer to a supported interaction member")
        target[rows[key], columns[route]] = float(value)
    return target


def loss_for_examples(model: RCShapleyNet, examples: Sequence[PlanTrainingExample], *,
                      config: TrainingConfig | None = None,
                      ranking_pairs: list[tuple[int, int]] | None = None) -> LossTerms:
    """Encode each plan once, then use all its masks and independent κ labels."""
    if not examples:
        raise ValueError("A training batch cannot be empty")
    config = config or TrainingConfig()
    device = next(model.parameters()).device
    encoded = [model.forward_plan(e.graph.to(device)) for e in examples]
    errors, recovery_errors = [], []
    for output, example in zip(encoded, examples):
        errors.append((output.values(example.masks)-example.coalition_targets.to(device))**2)
        targets = _recovery_targets(output, example.recovery_targets)
        valid = output.recovery_mask & torch.isfinite(targets)
        if config.resource_weight > 0 and output.recovery_mask.any() and not valid.any():
            raise ValueError("Task-associated plans require independent physical recovery labels")
        if valid.any():
            if torch.any((targets[valid] < 0) | (targets[valid] > 1)):
                raise ValueError("Physical recovery difficulties must lie in [0,1]")
            recovery_errors.append((output.recovery_predictions[valid]-targets[valid])**2)
    zero = encoded[0].coefficients.sum()*0
    coalition = torch.cat(errors).mean()
    resource = torch.cat(recovery_errors).mean() if recovery_errors else zero
    pairs = (make_ranking_pairs(examples, fraction=config.ranking_fraction, seed=config.seed)
             if ranking_pairs is None else ranking_pairs)
    hinge = []
    for good, bad in pairs:
        if examples[good].reference_id != examples[bad].reference_id:
            raise ValueError("Ranking pairs must share the same frozen reference")
        if examples[good].physical_full_value <= examples[bad].physical_full_value:
            raise ValueError("A ranking positive must have strictly higher physical J")
        hinge.append(torch.relu(config.ranking_margin -
                                (encoded[good].full_value-encoded[bad].full_value)))
    ranking = torch.stack(hinge).mean() if hinge else zero
    sparse = torch.stack([output.coefficients.abs().sum() for output in encoded]).mean()
    total = (config.coalition_weight*coalition + config.resource_weight*resource
             + config.ranking_weight*ranking
             + config.sparsity_weight*sparse)
    return LossTerms(total, coalition, resource, ranking, sparse)


def _batches(examples: Sequence[PlanTrainingExample], batch_size: int,
             rng: np.random.Generator):
    """Keep each state's candidate pool together so ranks use its full pool."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    groups: dict[str, list[PlanTrainingExample]] = {}
    for example in examples:
        groups.setdefault(example.reference_id, []).append(example)
    if any(len(pool) > batch_size for pool in groups.values()):
        raise ValueError("Batch size must fit one complete same-reference plan pool")
    keys = list(groups)
    rng.shuffle(keys)
    batch = []
    for key in keys:
        if batch and len(batch)+len(groups[key]) > batch_size:
            yield batch
            batch = []
        batch.extend(groups[key])
    if batch:
        yield batch


@torch.no_grad()
def validation_coalition_mae(model: RCShapleyNet,
                              examples: Sequence[PlanTrainingExample]) -> float:
    if not examples:
        raise ValueError("Validation requires held-out examples")
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for example in examples:
        output = model.forward_plan(example.graph.to(device))
        error = (output.values(example.masks)-example.coalition_targets.to(device)).abs()
        total += float(error.sum().cpu())
        count += error.numel()
    return total/count


def fit(model: RCShapleyNet, training: Sequence[PlanTrainingExample],
        validation: Sequence[PlanTrainingExample], *, config: TrainingConfig | None = None,
        checkpoint: str | Path | None = None,
        normalizer: FeatureNormalizer | None = None) -> TrainingResult:
    """AdamW training; restore the best validation-MAE weights before return."""
    config = config or TrainingConfig()
    if not training or not validation or config.epochs < 1 or config.patience < 1:
        raise ValueError("Training/validation data and positive epoch/patience limits are required")
    train_q = {e.reference_id for e in training}
    if train_q & {e.reference_id for e in validation}:
        raise ValueError("Training and validation reference IDs must be disjoint")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.Generator(np.random.PCG64(config.seed))
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best, best_epoch, waiting, state = math.inf, 0, 0, None
    history = []
    for epoch in range(1, config.epochs+1):
        model.train()
        losses = []
        for batch in _batches(training, config.batch_size, rng):
            pairs = make_ranking_pairs(batch, fraction=config.ranking_fraction,
                                       seed=int(rng.integers(0, 2**31)))
            optimizer.zero_grad(set_to_none=True)
            terms = loss_for_examples(model, batch, config=config, ranking_pairs=pairs)
            if not torch.isfinite(terms.total):
                raise FloatingPointError("Nonfinite supervised loss")
            terms.total.backward()
            optimizer.step()
            losses.append(terms.detached())
        mae = validation_coalition_mae(model, validation)
        row = {name: sum(v[name] for v in losses)/len(losses) for name in losses[0]}
        row.update(epoch=epoch, validation_mae=mae)
        history.append(row)
        if mae < best-config.min_delta:
            best, best_epoch, waiting = mae, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
        if waiting >= config.patience:
            break
    if state is None:
        raise FloatingPointError("No finite validation result was obtained")
    model.load_state_dict(state)
    model.eval()
    result = TrainingResult(best_epoch, best, history, len(history) < config.epochs)
    if checkpoint is not None:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_configuration": model.configuration,
                    "state_dict": model.state_dict(), "training_configuration": asdict(config),
                    "training_result": asdict(result),
                    "normalizer": normalizer.state_dict() if normalizer is not None else None}, path)
    return result


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[RCShapleyNet, FeatureNormalizer | None, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = RCShapleyNet(**payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    state = payload.get("normalizer")
    return model, FeatureNormalizer.from_state_dict(state) if state is not None else None, payload
