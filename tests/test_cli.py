"""Command interfaces exercise actual physical inputs, without benchmark claims."""
import json
from pathlib import Path

import pytest

from rc_shapleynet.cli import main
from rc_shapleynet.game import FixedRouteGame, mobius_dividends, physical_allocation
from rc_shapleynet.physics import evaluate
from rc_shapleynet.schema import plan_from_dict, scenario_from_dict
from rc_shapleynet.search import BudgetExhausted, SearchBudget


ROOT = Path(__file__).resolve().parents[1]


def test_validate_and_exact_audit_example(tmp_path, capsys):
    example = ROOT/"examples"/"figure2.json"
    assert main(["validate", str(example)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["feasible"]
    output = tmp_path/"audit.json"
    main(["audit", str(example), "--out", str(output)])
    result = json.loads(output.read_text())
    pair = next(v for v in result["interactions"] if v["members"] == ["A", "B"])
    assert pair["difficulties"] == pytest.approx({"A": 1, "B": .25}, abs=1e-7)
    assert abs(result["efficiency_residual"]) < 1e-10
    with pytest.raises(FileExistsError):
        main(["audit", str(example), "--out", str(output)])


def test_exact_physical_recovery_respects_shared_online_budget():
    payload = json.loads((ROOT/"examples"/"figure2.json").read_text())
    scenario, plan = scenario_from_dict(payload["scenario"]), plan_from_dict(payload["plan"])
    actual_calls = []
    def counted(s, p):
        actual_calls.append(p)
        return evaluate(s, p)
    budget = SearchBudget(max_evaluations=4, final_reserve_evaluations=0,
                          evaluator=counted)
    game = FixedRouteGame(scenario, plan, evaluator=budget.evaluate)
    dividends = mobius_dividends(game.enumerate())
    with pytest.raises(BudgetExhausted):
        physical_allocation(scenario, plan, dividends, evaluator=budget.evaluate)
    assert budget.evaluations == len(actual_calls) == 4


def test_statistics_help_routes_to_full_dedicated_parser(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["statistics", "--help"])
    assert caught.value.code == 0
    assert "--list-families" in capsys.readouterr().out
