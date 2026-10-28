# Paired statistical analysis

The package analyzes supplied episode summaries and matched attribution/target observations. Scalar service inference and correlation-difference inference have separate inputs and decisions. The release contains constructed software tests; original benchmark run logs were not supplied.

## Episode-summary input

`rc_shapleynet.statistics` requires one CSV row per method, metric, condition, independent scenario, simulation seed, and training repetition.

| Column | Meaning |
| --- | --- |
| `experiment_id` | Declared experiment, such as `E01`. |
| `condition_id` | Complete condition identifier, including relevant traffic/load/layout factors. |
| `domain` | Domain, such as `EV` or `WRSN`. |
| `profile` | Execution profile, such as `query_controlled`; wall-clock profiles remain separate. |
| `metric` | Endpoint with a consistent unit across matched methods. |
| `scenario_id` | Independent test scenario. |
| `simulation_seed` | Shared exogenous simulation draw for that scenario. |
| `training_seed` | Repeated model-training identifier. |
| `method` | Method or factorial-cell identifier. |
| `value` | Finite numerical episode summary. |

Identifiers are read as strings. Within each experiment/condition/domain/profile/metric context, pairing uses exactly `(scenario_id, simulation_seed, training_seed)`. Duplicate keys, unmatched methods, nonfinite values, multiple simulation seeds per scenario, and incomplete crossed grids are rejected. A single ambiguous `seed` field cannot replace the two distinct seed identifiers.

[configs/statistics.yaml](../configs/statistics.yaml) declares 100 scenarios and five training seeds by default: 500 matched episode differences give **100 independent scenario observations**. Additional simulation replicates need an explicitly defined clustering design. Different physical steps, search profiles, and metric units retain their own context labels.

## Estimands and tests

For each scenario, the code averages its five oriented training-seed differences. Positive differences favor the subject, including when a lower-is-better endpoint has been negated. It reports the overall mean and the median of these scenario-average differences.

The primary profile uses a two-sided sign-flip permutation test of the scenario mean with the Monte Carlo plus-one correction. Crossed percentile intervals independently resample scenario rows and training-seed columns with replacement. The same coordinates apply to both matched methods. Defaults are 2,000 bootstrap draws and 10,000 permutations.

The supplementary Wilcoxon test is two-sided, uses `zero_method="wilcox"`, drops zero scenario differences, and averages tied absolute ranks. The default `wilcoxon_method: approx` uses SciPy's asymptotic calculation without continuity correction. `wilcoxon_W` is `min(W_plus, W_minus)`. The matched-pairs rank-biserial correlation is

```text
RBC = (W_plus - W_minus) / (W_plus + W_minus)
```

All-zero differences return `W=0`, `p=1`, and `RBC=0`. RBC is a signed rank effect, separate from a standardized normal statistic. The permutation test requires sign exchangeability under the null. The signed-rank location interpretation requires the corresponding symmetry assumption on independent scenario differences.

Optional `difference_decimals` rounds episode contrasts once, before training-seed averaging. The resulting contrasts feed all estimates, intervals, and tests. The default performs no rounding and does not modify input CSV values.

For a four-cell difference-in-differences, the matched weighted contrast is formed before any resampling. This retains all four cells jointly. `E11_main` is the historical family name for the conditional `RC-interaction - Std-interaction` contrast; it is distinct from the four-cell `E11_interaction` contrast. A conditional contrast's interval or p value cannot be assigned to the factorial interaction.

## Holm families and decisions

Family members are explicitly listed in `configs/statistics.yaml`. Every member of each selected scalar family must be present before inference or output. The code does not shrink a family to fit available rows. Multiple selected families retain their own multiplicity scopes; `--metric` checks consistency and does not filter members.

| Family | Size | Procedure |
| --- | ---: | --- |
| `E01_std`, `E01_alns`, `E01_adap` | 6 each | Three EV demand rates × two traffic conditions. |
| `E02_std`, `E02_gdqn` | 9 each | Three WRSN loads × three layouts. |
| `E11_interaction` | 2 | Four-cell DID, one per domain. |
| `E11_main` | 2 | Fixed-neighborhood conditional contrast, one per domain. |
| `E12_mech` | 12 | Six named replacement methods × two domains. |
| `E15_rho` | 3 | Separate paired raw-score correlation procedure below. |
| `E16_perturb` | 6 | Six declared perturbation conditions. |

The six `E12_mech` baselines are `RC-no-difficulty`, `RC-no-urgency`, `RC-shuffled-scores`, `Global-fixed-weights`, `Direct-recovery-priority`, and `PASV-compatible`. These are input identifiers for real observations, rather than generated replacement scores. No scalar family is active by default.

The scalar JSON exposes two distinct eligibility decisions:

| Field | Decision |
| --- | --- |
| `original_primary_dagger_eligible` and compatibility alias `dagger_eligible` | Permutation Holm p < alpha, positive lower **mean** crossed-bootstrap interval endpoint, and declared minimum **mean** effect. |
| `wilcoxon_dagger_eligible` | Wilcoxon Holm p < alpha, positive lower **median** crossed-bootstrap interval endpoint, and declared minimum **median** effect. |

The manuscript's supplementary daggers use the Wilcoxon field. Default alpha is 0.05. Completion and coverage use fractions, with `minimum_effect: 0.01` and `effect_unit: fraction_rate`, corresponding to 1.0 percentage point. An absent threshold stops scalar analysis; non-rate endpoints require a threshold in their own units. RBC is reported independently of eligibility.

```text
python -m rc_shapleynet.statistics --list-families
python -m rc_shapleynet.statistics episodes.csv --config configs/statistics.yaml --family E01_std --validate-only --out outputs/validation.json
python -m rc_shapleynet.statistics episodes.csv --config configs/statistics.yaml --family E01_std --family E01_alns --out outputs/paired.json
```

The JSON records family membership, pairing identifiers, scenario/seed counts, rank sums, raw and adjusted p values, RBC, both estimates and intervals, runtime versions, and input/configuration hashes. A companion CSV is written by default. Bootstrap and permutation RNG streams derive separately from the declared seed and test ID. Subject, direction, sampling counts, and effect overrides remain explicit; a fraction-rate threshold must still satisfy the configured one-percentage-point rule.

## Paired attribution correlation

`rc_shapleynet.correlation` implements the separate raw-score analysis. Selecting `E15_rho` through the scalar statistics command produces an explicit unsupported-kind error; aggregate rho values cannot be passed to scalar Wilcoxon.

The correlation CSV has these columns:

| Column | Meaning |
| --- | --- |
| `scenario_id`, `simulation_seed`, `training_seed` | Shared scenario and repetition identifiers. |
| `route_id` | Route observation within that scenario/repetition. |
| `subset` | Predeclared evaluation subset. |
| `method` | Attribution method identifier. |
| `prediction` | Route attribution score. |
| `service_loss` | The same independent physical service-loss target for both methods. |

Pair keys are `(scenario_id, simulation_seed, training_seed, route_id, subset)`. The implementation checks duplicate observations, identical method keys, finite scores, common physical targets, one simulation seed per scenario, and a complete scenario-by-training-seed grid. Constant score or target vectors have undefined Spearman correlation and are rejected. Use one consistent domain, execution profile, and target definition per invocation; this raw-score format has no scalar-context selector.

`paired_scores(frame, subject, baseline, subset)` forms the paired input. `crossed_difference(data, resamples=2000, seed=20261028, confidence=.95)` computes pooled Spearman rho for each method and their difference. Each bootstrap draw independently samples scenario IDs and training-seed IDs, retaining all route rows for every sampled cell and preserving the shared target. Route rows remain repeated observations. The procedure reports actual scenario and training-seed counts; verify those against the intended design. It rejects runs with too many degenerate resamples.

The JSON reports `rho_subject`, `rho_baseline`, `difference`, a percentile `ci`, and finite-resample-corrected `p_raw`. Here `p_raw` is two-sided **percentile-bootstrap sign-tail evidence**. It is distinct from the scalar primary sign-flip permutation p value and does not create a percentage-point service dagger. The command applies Holm correction to the supplied subset family and writes `p_holm`.

The declared manuscript family is `all`, `heterogeneous`, and `symmetric`. Supply all three together when reporting `E15_rho`; changing `--subsets` changes the command's correction scope and must be declared as a different analysis.

```text
python -m rc_shapleynet.correlation route_scores.csv --subject RC-ShapleyNet --baseline Std-Shapley --subsets all heterogeneous symmetric --resamples 2000 --seed 20261028 --out outputs/correlation.json
```

Scalar and correlation outputs are computed from their supplied rows. Neither procedure reconstructs episode observations, W statistics, intervals, or physical targets from the manuscript's aggregate tables.
