"""Paired inference from actual episode summaries and explicit Holm families.

Pair keys: scenario_id, simulation_seed, training_seed. Training seeds are
repeated measurements. Wilcoxon and the preserved sign-flip permutation test
use one training-seed mean per scenario. A crossed bootstrap resamples scenario
and training-seed axes independently. No runs, W, or CIs are reconstructed from
aggregate tables. See docs/STATISTICS.md for the input contract.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon
import yaml

CONTEXT = ('experiment_id', 'condition_id', 'domain', 'profile', 'metric')
PAIR_KEYS = ('scenario_id', 'simulation_seed', 'training_seed')
ID_COLUMNS = CONTEXT + PAIR_KEYS + ('method',)
REQUIRED_COLUMNS = ID_COLUMNS + ('value',)


def default_config_path():
    """Use the editable preset or its byte-identical installed-wheel resource."""
    source = ROOT/'configs'/'statistics.yaml'
    if source.is_file():
        return source
    packaged = Path(__file__).resolve().parent/'resources'/'statistics.yaml'
    if packaged.is_file():
        return packaged
    raise FileNotFoundError('The packaged statistics preset is missing; supply --config explicitly.')


def holm(p_values):
    """Holm step-down adjustment with monotone cumulative maxima."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError('A nonempty finite p-value vector is required.')
    if ((p < 0) | (p > 1)).any():
        raise ValueError('P values must lie in [0,1].')
    order = np.argsort(p, kind='stable')
    result = np.empty_like(p)
    result[order] = np.minimum(np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]), 1.0)
    return result


def matched_rank_biserial(differences):
    """Drop zeros, average tied absolute ranks, keep signs; never use Z/sqrt(N)."""
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError('RBC requires a finite one-dimensional difference vector.')
    nonzero = d[d != 0]
    if not len(nonzero):
        return dict(W=0.0, W_plus=0.0, W_minus=0.0, rank_biserial=0.0,
                    n_nonzero=0, n_zero=len(d), n_tied_absolute_groups=0)
    ranks = rankdata(np.abs(nonzero), method='average')
    plus, minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    _, counts = np.unique(np.abs(nonzero), return_counts=True)
    return dict(W=min(plus, minus), W_plus=plus, W_minus=minus,
                rank_biserial=(plus-minus)/(plus+minus), n_nonzero=len(nonzero),
                n_zero=len(d)-len(nonzero), n_tied_absolute_groups=int(np.sum(counts > 1)))


def crossed_bootstrap(d, resamples, confidence, rng):
    """Percentile CIs for mean gain and median of scenario-mean gains.

    Each draw selects n scenario rows and s training-seed columns independently,
    both with replacement. Contrasts are formed first, so identical coordinates
    are applied jointly to every method/group, including all four DID cells.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or not np.isfinite(d).all() or min(d.shape) < 1:
        raise ValueError('Crossed bootstrap requires a complete finite 2-D grid.')
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError('Positive resamples and confidence in (0,1) are required.')
    n, s = d.shape
    means, medians = np.empty(resamples), np.empty(resamples)
    for start in range(0, resamples, 256):
        count = min(256, resamples-start)
        rows, cols = rng.integers(n, size=(count, n)), rng.integers(s, size=(count, s))
        scenario_means = d[rows[:, :, None], cols[:, None, :]].mean(axis=2)
        means[start:start+count] = scenario_means.mean(axis=1)
        medians[start:start+count] = np.median(scenario_means, axis=1)
    tail = (1-confidence)/2
    return dict(mean_ci=np.quantile(means, [tail, 1-tail]).tolist(),
                median_ci=np.quantile(medians, [tail, 1-tail]).tolist())


def sign_flip_permutation(scenario_differences, permutations, rng):
    """Preserved two-sided scenario sign-flip mean test, Monte Carlo +1 rule.

    Sign exchangeability is required under the null. An uncentered bootstrap
    tail proportion is not substituted for a null hypothesis test.
    """
    d = np.asarray(scenario_differences, dtype=float)
    if permutations < 1:
        raise ValueError('At least one permutation is required.')
    if np.all(d == 0):
        return 1.0
    observed, extreme = abs(float(d.mean())), 0
    for start in range(0, permutations, 256):
        count = min(256, permutations-start)
        signs = 2*rng.integers(2, size=(count, len(d)))-1
        extreme += int(np.count_nonzero(np.abs((signs*d).mean(axis=1)) >= observed))
    return (1+extreme)/(1+permutations)


def load_config(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig') as handle:
        config = json.load(handle) if path.suffix.lower() == '.json' else yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('statistics config schema_version must be 1.')
    if config.get('pair_keys') != list(PAIR_KEYS):
        raise ValueError('Pair keys must be scenario_id, simulation_seed, training_seed.')
    if not isinstance(config.get('families'), list):
        raise ValueError('Explicit families must be a list.')
    seen = set()
    for family in config['families']:
        fid, members = family.get('id'), family.get('members')
        if not isinstance(fid, str) or not fid or fid in seen:
            raise ValueError('Family IDs must be nonempty and unique.')
        seen.add(fid)
        if not isinstance(members, list) or not members:
            raise ValueError('Family '+fid+' requires explicit members.')
        if family.get('expected_size') != len(members):
            raise ValueError('Family '+fid+' does not match declared expected_size.')
        ids = [m.get('id') for m in members]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError('Member IDs must be present and unique within '+fid)
    return config


def validate_frame(df):
    missing = set(REQUIRED_COLUMNS)-set(df.columns)
    if missing:
        extra = ' A single seed cannot distinguish simulation and training.' if 'seed' in df else ''
        raise ValueError('Missing required columns: '+', '.join(sorted(missing))+'.'+extra)
    if df.empty:
        raise ValueError('The input contains no actual episode rows.')
    df = df.copy()
    for col in ID_COLUMNS:
        if df[col].isna().any():
            raise ValueError('Missing identifier in '+col)
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq('').any():
            raise ValueError('Empty identifier in '+col)
    try:
        df['value'] = pd.to_numeric(df['value'], errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Values must be numeric episode summaries.') from exc
    if not np.isfinite(df['value'].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite episode values are not permitted.')
    duplicate_keys = list(CONTEXT+PAIR_KEYS+('method',))
    if df.duplicated(duplicate_keys).any():
        bad = df.loc[df.duplicated(duplicate_keys, keep=False), duplicate_keys].iloc[0].to_dict()
        raise ValueError('Duplicate training-seed episode key: '+str(bad))
    # Multiple simulation replicates need a separately specified cluster model.
    mapping = df.groupby(list(CONTEXT)+['scenario_id'], dropna=False)['simulation_seed'].nunique()
    if mapping.gt(1).any():
        raise ValueError('Simulation-seed mismatch or multiple simulations per scenario: '+str(mapping[mapping.gt(1)].index[0]))
    return df


def read_runs(path):
    return validate_frame(pd.read_csv(path, dtype={col: 'string' for col in ID_COLUMNS}))


def selected_families(config, requested=None):
    wanted = list(requested or config.get('selected_families', []))
    if not wanted:
        raise ValueError('Select complete families with --family or selected_families; none are inferred from rows.')
    if len(set(wanted)) != len(wanted):
        raise ValueError('A family was selected more than once.')
    lookup = {f['id']: f for f in config['families']}
    unknown = set(wanted)-set(lookup)
    if unknown:
        raise ValueError('Unknown family IDs: '+', '.join(sorted(unknown)))
    return [copy.deepcopy(lookup[fid]) for fid in wanted]


def contrast_grid(df, member, defaults):
    mid, kind = member['id'], member.get('kind', 'paired')
    if kind not in ('paired', 'linear_contrast'):
        raise NotImplementedError(mid+': '+kind+' is not implemented by this scalar pipeline; a separate raw-score correlation procedure is required.')
    if member.get('requires_author_definition', False):
        raise ValueError(mid+': requires_author_definition; group/condition names must be supplied before analysis.')
    if member.get('minimum_effect') is None:
        raise ValueError(mid+': an explicit minimum_effect is required; an implicit zero cannot authorize a dagger.')
    threshold = float(member['minimum_effect'])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(mid+': minimum_effect must be a finite nonnegative value in the metric units.')
    if member.get('effect_unit') == 'fraction_rate' and not np.isclose(threshold, 0.01, rtol=0, atol=1e-15):
        raise ValueError(mid+': the declared fraction-rate practical threshold is 0.01 (1.0 percentage point).')
    context = member.get('context', {})
    if set(context) != set(CONTEXT) or any(v is None for v in context.values()):
        raise ValueError(mid+': every experiment/condition/domain/profile/metric field must be explicit.')
    frame = df
    for key in CONTEXT:
        frame = frame.loc[frame[key] == str(context[key])]
    if frame.empty:
        raise ValueError(mid+': missing context rows; the declared Holm family will not be reduced.')
    if kind == 'paired':
        subject, baseline = member.get('subject', defaults.get('subject')), member.get('baseline')
        if not subject or not baseline or subject == baseline:
            raise ValueError(mid+': distinct subject and baseline are required.')
        weights = {subject: 1.0, baseline: -1.0}
    else:
        weights = member.get('weights')
        if not isinstance(weights, dict) or len(weights) < 2:
            raise ValueError(mid+': explicit linear-contrast weights are required.')
        weights = {str(k): float(v) for k, v in weights.items()}
        if not np.isfinite(list(weights.values())).all() or not np.isclose(sum(weights.values()), 0, atol=1e-12):
            raise ValueError(mid+': contrast weights must be finite and sum to zero.')
        if not any(v > 0 for v in weights.values()) or not any(v < 0 for v in weights.values()):
            raise ValueError(mid+': a nontrivial signed contrast is required.')
    keys, arrays = None, []
    for method, weight in weights.items():
        values = frame.loc[frame['method'] == method, list(PAIR_KEYS)+['value']]
        if values.empty:
            raise ValueError(mid+': required method '+method+' is absent; Holm family remains incomplete.')
        values = values.set_index(list(PAIR_KEYS)).sort_index()
        if keys is None:
            keys = values.index
        elif not keys.equals(values.index):
            raise ValueError(mid+': unmatched scenario/simulation/training keys for '+method+
                             '; missing='+str(list(keys.difference(values.index)[:3]))+
                             '; extra='+str(list(values.index.difference(keys)[:3])))
        arrays.append(values['value'].to_numpy(dtype=float)*weight)
    combined = np.sum(arrays, axis=0)
    direction = member.get('direction', 'higher')
    if direction not in ('higher', 'lower'):
        raise ValueError(mid+': direction must be higher or lower.')
    if direction == 'lower':
        combined = -combined
    grid = pd.Series(combined, index=keys).unstack('training_seed').sort_index(axis=0).sort_index(axis=1)
    if grid.isna().any().any():
        raise ValueError(mid+': incomplete crossed scenario/training-seed grid.')
    if grid.index.get_level_values('scenario_id').nunique() != len(grid):
        raise ValueError(mid+': simulation replicates cannot be independent scenarios.')
    n, s = grid.shape
    if n != defaults['expected_scenarios']:
        raise ValueError(mid+': expected '+str(defaults['expected_scenarios'])+' independent scenarios, found '+str(n))
    if s != defaults['expected_training_seeds']:
        raise ValueError(mid+': expected '+str(defaults['expected_training_seeds'])+' training seeds, found '+str(s))
    if n < 2:
        raise ValueError(mid+': at least two independent scenarios are required.')
    return grid, weights, direction


def test_rng(seed, identity):
    words = np.frombuffer(hashlib.sha256(identity.encode()).digest()[:16], dtype='<u4').tolist()
    boot, perm = np.random.SeedSequence([seed]+words).spawn(2)
    return np.random.default_rng(boot), np.random.default_rng(perm)


def compute_member(grid, member, settings, weights, direction):
    d = grid.to_numpy(dtype=float)
    decimals = settings.get('difference_decimals')
    if decimals is not None:
        d = np.round(d, decimals=int(decimals))
    scenario_differences = d.mean(axis=1)
    ranks = matched_rank_biserial(scenario_differences)
    raw_p = 1.0
    if ranks['n_nonzero']:
        result = wilcoxon(scenario_differences, alternative='two-sided', zero_method='wilcox',
                          correction=False, method=settings.get('wilcoxon_method', 'approx'))
        raw_p = float(result.pvalue)
        if not np.isfinite(raw_p) or not np.isclose(float(result.statistic), ranks['W']):
            raise RuntimeError(member['id']+': Wilcoxon/rank audit did not agree.')
    boot_rng, perm_rng = test_rng(settings['random_seed'], member['id'])
    ci = crossed_bootstrap(d, settings['bootstrap_resamples'], settings['confidence'], boot_rng)
    pp = sign_flip_permutation(scenario_differences, settings['permutations'], perm_rng)
    return dict(test_id=member['id'], kind=member.get('kind', 'paired'), **member['context'],
                contrast_weights=weights, direction=direction, n_scenarios=d.shape[0],
                n_training_seeds=d.shape[1], n_matched_episode_summaries=int(d.size),
                wilcoxon_W=ranks['W'], wilcoxon_raw_p=raw_p, wilcoxon_p=raw_p,
                W_plus=ranks['W_plus'], W_minus=ranks['W_minus'], rank_biserial=ranks['rank_biserial'],
                n_nonzero_scenario_differences=ranks['n_nonzero'], n_zero_scenario_differences=ranks['n_zero'],
                n_tied_absolute_rank_groups=ranks['n_tied_absolute_groups'],
                median_difference=float(np.median(scenario_differences)),
                median_crossed_bootstrap_ci=ci['median_ci'], mean_difference=float(d.mean()),
                mean_crossed_bootstrap_ci=ci['mean_ci'], oriented_mean_difference=float(d.mean()),
                crossed_bootstrap_low=ci['mean_ci'][0], crossed_bootstrap_high=ci['mean_ci'][1],
                permutation_raw_p=pp, permutation_p=pp,
                minimum_mean_effect=float(member['minimum_effect']),
                minimum_median_effect=float(member['minimum_effect']),
                minimum_effect_unit=member.get('effect_unit', 'original_metric_units'),
                training_seed_ids=grid.columns.tolist(),
                scenario_simulation_ids=[list(x) for x in grid.index.tolist()])


def adjust_family(records, family_id, expected_size, alpha):
    if len(records) != expected_size:
        raise ValueError('Full declared Holm family '+family_id+' is required.')
    wp = holm([r['wilcoxon_raw_p'] for r in records])
    pp = holm([r['permutation_raw_p'] for r in records])
    adjusted = []
    for record, w, p in zip(records, wp, pp):
        row = copy.deepcopy(record)
        row.update(family_id=family_id, family_size=expected_size,
                   wilcoxon_holm_p=float(w), permutation_holm_p=float(p))
        # The original permutation/mean-CI primary decision rule is preserved.
        eligible = bool(p < alpha and row['mean_crossed_bootstrap_ci'][0] > 0
                        and row['mean_difference'] >= row['minimum_mean_effect'])
        row['original_primary_dagger_eligible'] = eligible
        row['dagger_eligible'] = eligible
        row['wilcoxon_dagger_eligible'] = bool(
            w < alpha and row['median_crossed_bootstrap_ci'][0] > 0
            and row['median_difference'] >= row['minimum_median_effect'])
        adjusted.append(row)
    return adjusted


def analyze(df, config, requested=None, validate_only=False):
    df, settings = validate_frame(df), config['defaults']
    needed = ('expected_scenarios', 'expected_training_seeds', 'bootstrap_resamples',
              'permutations', 'confidence', 'random_seed', 'alpha')
    if any(key not in settings for key in needed):
        raise ValueError('Config lacks required default inference settings.')
    if settings['bootstrap_resamples'] < 1 or settings['permutations'] < 1:
        raise ValueError('Bootstrap/permutation counts must be positive.')
    if not 0 < settings['confidence'] < 1 or not 0 < settings['alpha'] < 1:
        raise ValueError('Confidence and alpha must lie in (0,1).')
    prepared = []
    # Validate every selected family member before any inference or output.
    for family in selected_families(config, requested):
        group = [(member, *contrast_grid(df, member, settings)) for member in family['members']]
        prepared.append((family, group))
    if validate_only:
        return dict(status='validated_only', families=[dict(id=f['id'], size=len(g)) for f, g in prepared],
                    input_rows=len(df), inference_performed=False)
    results = []
    for family, group in prepared:
        records = [compute_member(grid, member, settings, weights, direction)
                   for member, grid, weights, direction in group]
        results.extend(adjust_family(records, family['id'], family['expected_size'], settings['alpha']))
    return dict(status='computed_from_episode_rows', data_kind=config.get('data_kind', 'actual_run_logs'),
                primary_profile='original: scenario sign-flip permutation + full-family Holm + positive mean crossed-bootstrap CI + minimum mean effect',
                supplementary_profile='scenario-mean Wilcoxon + full-family Holm + positive median crossed-bootstrap CI + minimum median effect; matched-pairs RBC',
                pair_keys=list(PAIR_KEYS), independent_unit='scenario_id with its fixed simulation_seed',
                repeated_measure='training_seed', confidence_interval_method='crossed percentile bootstrap of scenario and training-seed axes',
                settings=copy.deepcopy(settings),
                declared_families=[dict(id=f['id'], expected_size=f['expected_size'], members=[m['id'] for m in f['members']]) for f, _ in prepared],
                input_profiles=sorted(df['profile'].unique().tolist()), input_rows=len(df), results=results)


def write_outputs(output, payload, csv_output=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    if payload.get('results'):
        target = Path(csv_output) if csv_output else output.with_suffix('.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in payload['results']:
            row = {k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                   for k, v in record.items()}
            rows.append(row)
        with target.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('--config', type=Path, default=default_config_path(), help='Explicit YAML/JSON families.')
    parser.add_argument('--family', action='append', help='Select a complete family; repeat for several.')
    parser.add_argument('--list-families', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--csv-out', type=Path)
    # Original CLI overrides remain; --metric is a guard, never a family filter.
    parser.add_argument('--ours')
    parser.add_argument('--metric')
    parser.add_argument('--lower-is-better', action='store_true')
    parser.add_argument('--resamples', type=int)
    parser.add_argument('--permutations', type=int)
    parser.add_argument('--min-effect', type=float)
    parser.add_argument('--random-seed', type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.list_families:
        for family in config['families']:
            kinds = {m.get('kind', 'paired') for m in family['members']}
            status = ('use-correlation-command' if kinds-{'paired', 'linear_contrast'}
                      else 'author-definition-required' if any(m.get('requires_author_definition') for m in family['members']) else 'supported')
            print(family['id'], 'm='+str(family['expected_size']), ','.join(sorted(kinds)), status)
        return
    if args.input is None or args.out is None:
        parser.error('input and --out are required except with --list-families.')
    families = selected_families(config, args.family)
    metrics = {m.get('context', {}).get('metric') for f in families for m in f['members']}
    if args.metric and metrics != {args.metric}:
        raise ValueError('--metric cannot filter members from a declared Holm family.')
    if (args.lower_is_better or args.min_effect is not None) and len(metrics) != 1:
        raise ValueError('Direction/effect overrides require one declared metric.')
    chosen = {f['id'] for f in families}
    for family in config['families']:
        if family['id'] in chosen:
            for member in family['members']:
                if args.ours and member.get('kind', 'paired') == 'paired':
                    member['subject'] = args.ours
                if args.lower_is_better:
                    member['direction'] = 'lower'
                if args.min_effect is not None:
                    member['minimum_effect'] = args.min_effect
    for attr, key in [('resamples', 'bootstrap_resamples'), ('permutations', 'permutations'), ('random_seed', 'random_seed')]:
        if getattr(args, attr) is not None:
            config['defaults'][key] = getattr(args, attr)
    payload = analyze(read_runs(args.input), config, args.family, args.validate_only)
    cli = vars(args).copy()
    for key, value in list(cli.items()):
        if isinstance(value, Path):
            cli[key] = str(value)
    payload['provenance'] = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                                 config=str(args.config.resolve()), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                 python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, cli_arguments=cli)
    write_outputs(args.out, payload, args.csv_out)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
