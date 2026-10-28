"""Command-line access to verified plans, physical labels and fitted models."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

from .schema import Plan, plan_from_dict, scenario_from_dict, to_dict


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(payload, path=None):
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    if path is None:
        print(encoded)
    else:
        target = Path(path)
        if target.exists():
            raise FileExistsError("Use a new output path; existing records are preserved")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded+"\n", encoding="utf-8")


def _input(path):
    payload = _read(path)
    return scenario_from_dict(payload["scenario"]), plan_from_dict(payload.get("plan", {"routes": []}))


def _evaluation(result):
    return dict(feasible=result.feasible,
                objective=result.objective if math.isfinite(result.objective) else None,
                metrics=result.metrics, violations=list(result.violations))


def _train(args):
    """Load actual context/plan pools; fit moments on training graphs only."""
    import torch
    from .graph import FeatureNormalizer, RELATIONS
    from .learning import RCShapleyNet
    from .pipeline import label_plans
    from .training import TrainingConfig, fit

    source = Path(args.input)
    manifest = _read(source)
    options = dict(manifest.get("training", {}))
    aliases = dict(max_epochs="epochs", rank_weight="ranking_weight",
                   sparse_weight="sparsity_weight", rank_margin="ranking_margin",
                   rank_pool_fraction="ranking_fraction")
    for old, new in aliases.items():
        if old in options:
            if new in options:
                raise ValueError(f"Declare only one of {old} and {new}")
            options[new] = options.pop(old)
    configuration = TrainingConfig(**options)
    if "cpu_threads" in manifest:
        if not isinstance(manifest["cpu_threads"], int) or manifest["cpu_threads"] < 1:
            raise ValueError("cpu_threads must be a positive integer")
        torch.set_num_threads(manifest["cpu_threads"])
    torch.manual_seed(configuration.seed)  # Before random parameter initialization.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configuration.seed)

    def pools(entries):
        result = []
        for entry in entries:
            context = scenario_from_dict(entry["scenario"])
            plans = tuple(plan_from_dict(p) for p in entry["plans"])
            trajectories = (tuple(scenario_from_dict(s) for s in entry["trajectories"])
                            if "trajectories" in entry else None)
            result.extend(label_plans(context, plans,
                mask_count=manifest.get("mask_count", 16),
                order=manifest.get("model", {}).get("order", 2),
                trajectories=trajectories))
        return result

    train_contexts = {scenario_from_dict(p["scenario"]).fingerprint()
                      for p in manifest["training_pools"]}
    validation_contexts = {scenario_from_dict(p["scenario"]).fingerprint()
                           for p in manifest["validation_pools"]}
    if train_contexts & validation_contexts:
        raise ValueError("Training and validation forecast states must be disjoint")
    training, validation = pools(manifest["training_pools"]), pools(manifest["validation_pools"])
    if not training or not validation:
        raise ValueError("Both disjoint training and validation pools are required")
    normalizer = FeatureNormalizer.fit([e.graph for e in training])
    training = [replace(e, graph=normalizer.transform(e.graph)) for e in training]
    validation = [replace(e, graph=normalizer.transform(e.graph)) for e in validation]
    # Location-ID vocabulary is supplied/frozen by the data protocol. Reserve
    # enough rows for the declared IDs, including the explicit unknown index.
    declared = manifest.get("num_locations")
    if declared is None:
        raise ValueError("Declare num_locations for the frozen training vocabulary plus unknown ID")
    if any(e.graph.recipient_location_ids.numel() and
           int(e.graph.recipient_location_ids.max()) >= declared for e in training+validation):
        raise ValueError("Location IDs exceed the declared frozen vocabulary")
    model = RCShapleyNet(training[0].graph.input_dims, RELATIONS,
                         num_locations=declared, **manifest.get("model", {}))
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        raise FileExistsError("Use a new checkpoint path")
    result = fit(model, training, validation, config=configuration,
                 checkpoint=checkpoint, normalizer=normalizer)
    _write(asdict(result), args.out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the complete dedicated statistical interfaces and help pages.
    if argv and argv[0] in {"statistics", "correlation"}:
        if argv[0] == "statistics":
            from .statistics import main as statistical_main
        else:
            from .correlation import main as statistical_main
        return statistical_main(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "audit", "search"):
        p = commands.add_parser(name)
        p.add_argument("input", type=Path, help="JSON containing an explicit scenario and plan")
        p.add_argument("--out", type=Path)
        if name == "audit":
            p.add_argument("--max-players", type=int, default=10)
            p.add_argument("--lambda", dest="lambda_", type=float, default=.5)
            p.add_argument("--tau", type=float, default=.5)
            p.add_argument("--eta", type=float, default=.5)
        if name == "search":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--device", default="cpu")
            p.add_argument("--standard", action="store_true", help="Same architecture; uniform shares (lambda=0)")
            p.add_argument("--max-evaluations", type=int, default=200)
            p.add_argument("--time-limit", type=float, default=60)
            p.add_argument("--max-trials", type=int, default=100)
            p.add_argument("--seed", type=int, default=1)
    p = commands.add_parser("train")
    p.add_argument("input", type=Path, help="Explicit disjoint context/plan/trajectory pools")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p = commands.add_parser("run")
    p.add_argument("input", type=Path, help="Actual scenario inputs and module:factory method adapters")
    p.add_argument("--out", type=Path, required=True)
    commands.add_parser("statistics", help="Paired scalar inference; dedicated --help")
    commands.add_parser("correlation", help="E15 paired raw-score inference; dedicated --help")
    args = parser.parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "run":
        from .experiments import run_manifest
        print(json.dumps(run_manifest(args.input, args.out), indent=2))
        return

    from .physics import evaluate
    scenario, plan = _input(args.input)
    if args.command == "validate":
        result = evaluate(scenario, plan)
        _write(_evaluation(result), args.out)
        return 0 if result.feasible else 2
    if args.command == "audit":
        from .game import FixedRouteGame, mobius_dividends, physical_allocation
        game = FixedRouteGame(scenario, plan)
        values = game.enumerate(max_players=args.max_players)
        dividends = mobius_dividends(values)
        attribution = physical_allocation(scenario, plan, dividends,
                                          args.lambda_, args.tau, args.eta)
        _write(dict(reference_id=game.reference_id,
            coalitions=[dict(members=k, value=v) for k, v in values.items()],
            interactions=[dict(members=k, value=v, shares=attribution.shares[k],
                difficulties=attribution.difficulties[k]) for k, v in dividends.items()],
            allocation=attribution.allocation,
            efficiency_residual=sum(attribution.allocation.values())-game(game.players),
            physical_queries=game.physical_calls+attribution.recovery_checker_calls), args.out)
        return
    if args.command == "search":
        from .training import load_checkpoint
        from .search import LearnedGuide, SearchConfig, search
        model, normalizer, _ = load_checkpoint(args.checkpoint, device=args.device)
        guide = LearnedGuide(model, normalizer=normalizer, lambda_=0 if args.standard else .5)
        result = search(scenario, plan, guide, config=SearchConfig(
            max_evaluations=args.max_evaluations, time_limit=args.time_limit,
            max_trials=args.max_trials, seed=args.seed))
        _write(dict(plan=to_dict(result.best_plan),
            evaluation=_evaluation(result.best_evaluation),
            final_allocation=(dict(result.final_guidance.allocation) if result.final_guidance else None),
            evaluations=result.evaluations, elapsed_seconds=result.elapsed_seconds,
            stop_reason=result.stop_reason, trace=[asdict(t) for t in result.trace]), args.out)


if __name__ == "__main__":
    sys.exit(main())
