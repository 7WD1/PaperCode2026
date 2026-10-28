"""Reproduce the controlled numerical checks of Theorems 1 and 2.

This is an enumerated signed-interaction schedule-value game, not a trained
scheduler or a physical benchmark.  Every value is obtained from the explicit
game; standard Shapley values are computed independently from marginal values
with factorial weights.  No measurements or confidence intervals are invented.

Run: python scripts/verify_theory.py
Dependencies: numpy, matplotlib, pymupdf (the last is used for PDF font audits).
All data, including the full 32-coalition games, are saved in data/theory/.
"""
from pathlib import Path
from itertools import combinations
import csv
import json
import math
import sys

PAPER = Path(__file__).resolve().parents[1]
for candidate in (PAPER / 'tools' / 'python-packages',
                  PAPER.parent / 'tools' / 'python-packages'):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pymupdf as fitz

SEED = 20261028
N = 5
GAMES = 250
LAM = 0.5
ETA = 0.25
TAUS = (0.1, 0.25, 0.5, 1.0, 2.0)
KS = (1, 2, 3)
EPSILONS = (0.0, 0.01, 0.03, 0.1)
PLOT_EPS = 0.03
FLOOR = 1e-16  # display only; the CSV and all checks use unfloored values
TOL = 2e-11
DATA = PAPER / 'data' / 'theory'
FIGS = PAPER / 'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

MEMBERS = {m: tuple(i for i in range(N) if m & (1 << i))
           for m in range(1, 1 << N)}
ORDER = {m: len(ids) for m, ids in MEMBERS.items()}
SUPPORT = {k: tuple(m for m, ids in MEMBERS.items()
                   if len(ids) <= k and ids[-1] - ids[0] + 1 == len(ids))
           for k in KS}


def values_from_dividends(dividends):
    """Zeta transform on all 32 coalitions, with v(empty)=0."""
    return np.array([sum(dividends[t] for t in MEMBERS if t & s == t)
                     for s in range(1 << N)], dtype=float)


def mobius_from_values(values):
    """Direct inclusion-exclusion, independent of the generating dividends."""
    out = np.zeros(1 << N)
    for mask in MEMBERS:
        sub = mask
        while True:
            out[mask] += (-1) ** (ORDER[mask] - sub.bit_count()) * values[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
    return out


def factorial_shapley(values):
    """Standard value from its factorial marginal formula, not dividend shares."""
    result = np.zeros(N)
    for i in range(N):
        for s in range(1 << N):
            if not s & (1 << i):
                k = s.bit_count()
                weight = math.factorial(k) * math.factorial(N-k-1) / math.factorial(N)
                result[i] += weight * (values[s | (1 << i)] - values[s])
    return result


def shares(kappa, urgency, tau, lam=LAM, equal_scores=False):
    result = np.zeros((1 << N, N))
    for t, ids in MEMBERS.items():
        scores = kappa[t, list(ids)] + ETA * urgency[t, list(ids)]
        if equal_scores:
            scores[:] = float(np.mean(scores))
        z = scores / tau
        expz = np.exp(z - z.max())
        p = expz / expz.sum()
        result[t, list(ids)] = (1-lam) / len(ids) + lam * p
    return result


def allocation(dividends, alpha, support=None):
    ts = tuple(MEMBERS) if support is None else support
    return np.sum(dividends[list(ts), None] * alpha[list(ts)], axis=0)


def draw_game(rng):
    """Explicit value components with service, overlap and completion motifs.

    Route labels A--E have a chain A-B-C-D-E for the sparse interaction support.
    Positive adjacent-route pairs encode complementarity; negative other pairs
    encode overlap/congestion. Small three-, four- and five-way dividends encode
    joint completion/connectivity thresholds. Values are arbitrary value units,
    not measured energy, completion rates, or fitted scheduler coefficients.
    """
    d = np.zeros(1 << N)
    for i in range(N):
        d[1 << i] = rng.uniform(2.0, 4.0)
    for i, j in combinations(range(N), 2):
        d[(1 << i) | (1 << j)] = (rng.uniform(0.8, 1.6) if j == i+1
                                else -rng.uniform(0.25, 0.8))
    for ids in combinations(range(N), 3):
        mask = sum(1 << i for i in ids)
        d[mask] = (rng.uniform(0.2, 0.6) if ids[-1]-ids[0] == 2
                   else rng.uniform(-0.12, 0.12))
    for ids in combinations(range(N), 4):
        d[sum(1 << i for i in ids)] = rng.uniform(-0.08, 0.16)
    d[(1 << N)-1] = rng.uniform(0.04, 0.12)
    return d


def write_csv(name, rows):
    with (DATA / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(x):
    return float(x)


def run_checks():
    rng = np.random.default_rng(SEED)
    structures, errors, games, examples = [], [], [], []
    max_mobius = 0.0
    max_reconstruction = 0.0
    max_individual_excess = -math.inf
    for game_id in range(GAMES):
        generated = draw_game(rng)
        v = values_from_dividends(generated)
        d = mobius_from_values(v)
        max_mobius = max(max_mobius, np.max(np.abs(d-generated)))
        max_reconstruction = max(max_reconstruction,
                                 np.max(np.abs(values_from_dividends(d)-v)))
        assert np.allclose(d, generated, rtol=0, atol=TOL)
        kappa = np.zeros((1 << N, N))
        urgency = np.zeros_like(kappa)
        score_direction = np.zeros_like(kappa)
        coefficient_direction = np.zeros(1 << N)
        for t, ids in MEMBERS.items():
            kappa[t, list(ids)] = rng.uniform(0.1, 0.9, len(ids))
            urgency[t, list(ids)] = rng.uniform(0.0, 1.0, len(ids))
            score_direction[t, list(ids)] = rng.uniform(-1.0, 1.0, len(ids))
            coefficient_direction[t] = rng.uniform(-1.0, 1.0)
        alpha = shares(kappa, urgency, 0.5)
        psi = allocation(d, alpha)
        phi = factorial_shapley(v)

        # The null-player test is constructed through coalition values, not by
        # inspecting its allocation: E is absent from every queried coalition.
        null_v = np.array([v[s & ~(1 << 4)] for s in range(1 << N)])
        null_d = mobius_from_values(null_v)
        null_psi = allocation(null_d, alpha)
        second_generated = draw_game(rng)
        second_v = values_from_dividends(second_generated)
        second_d = mobius_from_values(second_v)
        summed_d = mobius_from_values(v + second_v)
        additive_coefficients = rng.uniform(-1.0, 4.0, N)
        additive_v = np.array([sum(additive_coefficients[i] for i in MEMBERS.get(s, ()))
                               for s in range(1 << N)])
        additive_d = mobius_from_values(additive_v)
        # Proxy game uses fixed coefficients across masks, just as the theorem.
        proxy_c = d + PLOT_EPS * (1+np.abs(d)) * coefficient_direction
        proxy_k = np.clip(kappa + PLOT_EPS * score_direction, 0.0, 1.0)
        proxy_alpha = shares(proxy_k, urgency, 0.5)
        proxy_psi = allocation(proxy_c, proxy_alpha, SUPPORT[3])
        row = dict(game_id=game_id,
                   efficiency=number(abs(psi.sum()-v[-1])),
                   null_player=number(abs(null_psi[4])),
                   lambda_zero=number(np.max(np.abs(allocation(d, shares(kappa, urgency, 0.5, lam=0))-phi))),
                   equal_scores=number(np.max(np.abs(allocation(d, shares(kappa, urgency, 0.5, equal_scores=True))-phi))),
                   fixed_share_linearity=number(np.max(np.abs(allocation(summed_d, alpha)-psi-allocation(second_d, alpha)))),
                   additive_game=number(np.max(np.abs(allocation(additive_d, alpha)-additive_coefficients))),
                   proxy_efficiency=number(abs(proxy_psi.sum()-sum(proxy_c[t] for t in SUPPORT[3]))))
        structures.append(row)
        assert max(value for key, value in row.items() if key != 'game_id') < TOL
        examples.append(dict(game_id=game_id, value=number(v[-1]),
                             standard=phi.tolist(), resource_conditioned=psi.tolist()))
        games.append(dict(game_id=game_id, values=v.tolist(),
                          generating_dividends=generated.tolist(),
                          mobius_dividends=d.tolist(), kappa=kappa.tolist(),
                          urgency=urgency.tolist(),
                          coefficient_direction=coefficient_direction.tolist(),
                          recovery_direction=score_direction.tolist(),
                          linearity_second_values=second_v.tolist(),
                          additive_coefficients=additive_coefficients.tolist()))
        for tau in TAUS:
            a = shares(kappa, urgency, tau)
            true = allocation(d, a)
            for eps in EPSILONS:
                c = d + eps * (1+np.abs(d)) * coefficient_direction
                predicted_k = np.clip(kappa + eps * score_direction, 0.0, 1.0)
                ahat = shares(predicted_k, urgency, tau)
                for k in KS:
                    h = SUPPORT[k]
                    omitted = tuple(t for t in MEMBERS if t not in h)
                    pred = allocation(c, ahat, h)
                    individual_error = np.abs(pred-true)
                    coeff = float(np.sum(np.abs(c[list(h)]-d[list(h)])))
                    score_errors = {t: float(np.max(np.abs(predicted_k[t, list(MEMBERS[t])]
                                                         -kappa[t, list(MEMBERS[t])])))
                                    for t in h}
                    recovery = LAM / tau * sum(abs(d[t])*score_errors[t] for t in h)
                    truncation = float(np.sum(np.abs(d[list(omitted)])))
                    bound = coeff + recovery + truncation
                    individual_bound = np.sum(ahat[list(h)] * np.abs(c[list(h)]-d[list(h)])[:, None], axis=0)
                    individual_bound += np.sum(np.abs(ahat[list(h)]-a[list(h)]) * np.abs(d[list(h)])[:, None], axis=0)
                    individual_bound += np.sum(a[list(omitted)] * np.abs(d[list(omitted)])[:, None], axis=0)
                    exact_share_term = float(np.sum(np.abs(ahat[list(h)]-a[list(h)])
                                                    * np.abs(d[list(h)])[:, None]))
                    l1 = float(individual_error.sum())
                    excess = float(np.max(individual_error-individual_bound))
                    max_individual_excess = max(max_individual_excess, excess)
                    assert excess <= TOL
                    assert l1 <= bound + TOL
                    assert exact_share_term <= recovery + TOL
                    errors.append(dict(game_id=game_id, K=k, tau=tau,
                                       perturbation=eps, l1_error=l1,
                                       coefficient_term=coeff,
                                       recovery_lipschitz_term=number(recovery),
                                       omitted_dividend_term=truncation,
                                       total_bound=number(bound),
                                       exact_share_term=exact_share_term,
                                       error_bound_ratio=l1/bound,
                                       bound_slack=number(bound-l1),
                                       maximum_individual_excess=excess))
    summaries = []
    for eps in EPSILONS:
        for tau in TAUS:
            for k in KS:
                rows = [r for r in errors if r['K'] == k and r['tau'] == tau
                        and r['perturbation'] == eps]
                summary = dict(K=k, tau=tau, perturbation=eps, games=len(rows))
                for key in ('l1_error', 'coefficient_term', 'recovery_lipschitz_term',
                            'omitted_dividend_term', 'total_bound', 'exact_share_term'):
                    summary['mean_'+key] = float(np.mean([r[key] for r in rows]))
                summaries.append(summary)
    maxima = {key: max(row[key] for row in structures) for key in structures[0] if key != 'game_id'}
    result = dict(seed=SEED, games=GAMES, players=N, coalitions_per_game=1 << N,
                  floating_point='numpy float64', lambda_value=LAM, eta=ETA,
                  tau_values=TAUS, K_values=KS, perturbation_levels=EPSILONS,
                  plot_perturbation=PLOT_EPS, residual_display_floor=FLOOR,
                  support_rule='contiguous subsets of A-B-C-D-E with order <= K',
                  support_masks={str(k): list(v) for k, v in SUPPORT.items()},
                  perturbation_rule='c_T = delta_T + eps*(1+abs(delta_T))*z_T; kappa_hat = clip(kappa + eps*w,0,1); z,w uniform[-1,1], fixed per game',
                  game_description='controlled signed-interaction schedule-value games; not trained scheduler performance',
                  structural_maximum_absolute_residuals=maxima,
                  maximum_mobius_recovery_residual=float(max_mobius),
                  maximum_value_reconstruction_residual=float(max_reconstruction),
                  allocation_error_checks=len(errors),
                  individual_error_checks=N*len(errors),
                  bound_violations=sum(r['l1_error'] > r['total_bound']+TOL for r in errors),
                  maximum_error_bound_ratio=max(r['error_bound_ratio'] for r in errors),
                  minimum_global_bound_slack=min(r['bound_slack'] for r in errors),
                  maximum_individual_bound_excess=max_individual_excess,
                  absolute_check_tolerance=TOL,
                  all_checks_passed=True)
    write_csv('structure_checks.csv', structures)
    write_csv('error_checks.csv', errors)
    write_csv('error_plot_summary.csv', summaries)
    (DATA / 'games.json').write_text(json.dumps(dict(seed=SEED, games=games), indent=2), encoding='utf-8')
    (DATA / 'structure_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    return structures, summaries, examples, result


def setup_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'axes.labelsize': 9.5, 'axes.titlesize': 9.5,
                         'xtick.labelsize': 8.4, 'ytick.labelsize': 8.4,
                         'legend.fontsize': 8.4, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': 0.65,
                         'xtick.major.width': 0.65, 'ytick.major.width': 0.65,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def save_audited(fig, name):
    fig.savefig(FIGS / (name+'.pdf'))
    fig.savefig(FIGS / (name+'.png'), dpi=600)
    plt.close(fig)
    doc = fitz.open(FIGS / (name+'.pdf'))
    page = doc[0]
    spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
             for line in block['lines'] for span in line['spans'] if span['text'].strip()]
    minimum = min(span['size'] for span in spans)
    assert minimum >= 8.39, (name, minimum)
    assert all(page.rect.contains(fitz.Rect(span['bbox'])) for span in spans)
    info = dict(width_mm=page.rect.width*25.4/72, height_mm=page.rect.height*25.4/72,
                minimum_pdf_text_size_pt=minimum,
                font_names=sorted({span['font'] for span in spans}), png_dpi=600)
    doc.close()
    return info


def plot_structure(rows, examples):
    # Fixed margins preserve the exact 182 x 70 mm publication canvas.
    fig = plt.figure(figsize=(182/25.4, 70/25.4))
    ax = fig.add_axes([0.105, 0.245, 0.496, 0.60])
    bx = fig.add_axes([0.705, 0.245, 0.282, 0.60])
    keys = ('efficiency', 'null_player', 'lambda_zero', 'equal_scores',
            'fixed_share_linearity', 'additive_game', 'proxy_efficiency')
    labels = ('Total', 'Null', 'λ = 0', 'Uniform',
              'Linear', 'Additive', 'Proxy')
    colors = ('#0072B2', '#009E73', '#D55E00', '#CC79A7', '#0072B2', '#009E73', '#D55E00')
    jitter = np.linspace(-0.19, 0.19, GAMES)
    for x, (key, color) in enumerate(zip(keys, colors)):
        vals = np.array([r[key] for r in rows])
        ax.scatter(x+jitter, np.maximum(vals, FLOOR), s=4.5, color=color,
                   alpha=0.40, edgecolors='none', rasterized=False)
        ax.plot([x-.21, x+.21], [max(vals.max(), FLOOR)]*2, color=color, lw=1.6)
    ax.set_yscale('log')
    ax.set_ylim(6e-17, 7e-13)
    ax.set_yticks([1e-16, 1e-15, 1e-14, 1e-13], ['1e-16', '1e-15', '1e-14', '1e-13'])
    ax.minorticks_off()
    ax.set_xticks(range(len(keys)), labels)
    ax.set_xlim(-.45, len(keys)-.55)
    ax.set_ylabel('Absolute residual')
    ax.set_title('(a) Structural checks: 250 five-route games', pad=10, loc='left')
    ax.grid(axis='y', color='#d8dde4', lw=.55)
    ax.set_axisbelow(True)
    # A pre-specified representative game, not an example selected by effect size.
    ex = examples[0]
    x = np.arange(N)
    bx.bar(x-.185, ex['standard'], .35, color='#7cb4d3', label='Standard Shapley', edgecolor='white', lw=.3)
    bx.bar(x+.185, ex['resource_conditioned'], .35, color='#d87840', label='Resource-conditioned', edgecolor='white', lw=.3)
    bx.set_xticks(x, list('ABCDE'))
    bx.set_xlabel('Route')
    bx.set_ylabel('Allocated value')
    bx.set_title('(b) Fixed game 0', pad=10, loc='left')
    bx.set_ylim(0, max(*ex['standard'], *ex['resource_conditioned'])*1.72)
    bx.text(.03, .97, 'Both totals = '+format(ex['value'], '.3f'), transform=bx.transAxes,
            va='top', fontsize=9)
    bx.legend(loc='upper left', bbox_to_anchor=(-.04, .90), frameon=False,
              handlelength=1.05, labelspacing=.35, borderpad=.1)
    bx.grid(axis='y', color='#e2e6eb', lw=.5)
    bx.set_axisbelow(True)
    for a in (ax, bx):
        a.spines[['right', 'top']].set_visible(False)
    return save_audited(fig, 'theorem_structure')


def plot_error(rows):
    fig = plt.figure(figsize=(182/25.4, 70/25.4))
    ax = fig.add_axes([.081, .225, .433, .60])
    bx = fig.add_axes([.623, .225, .360, .60])
    palette = ('#0072B2', '#D55E00', '#009E73')
    for k, color in zip(KS, palette):
        selected = [r for r in rows if r['K'] == k and r['perturbation'] == PLOT_EPS]
        taus = [r['tau'] for r in selected]
        ax.plot(taus, [r['mean_l1_error'] for r in selected], 'o-', color=color,
                lw=1.5, ms=3.7)
        ax.plot(taus, [r['mean_total_bound'] for r in selected], 's--', color=color,
                lw=1.25, ms=3.0, markerfacecolor='white')
    ax.set_xscale('log')
    ax.set_xticks(TAUS, [str(t) for t in TAUS])
    ax.minorticks_off()
    ax.set_xlabel('Temperature τ')
    ax.set_ylabel('Mean L1 error / bound (value units)')
    ax.set_title('(a) Temperature and support order', pad=11, loc='left')
    ax.grid(color='#e2e6eb', lw=.5)
    order_legend = ax.legend([Line2D([], [], color=c, lw=2) for c in palette],
                            ['K = 1', 'K = 2', 'K = 3'], loc='upper center',
                            bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False,
                            handlelength=1.15, columnspacing=.8, handletextpad=.35)
    ax.add_artist(order_legend)
    ax.legend([Line2D([], [], color='#414b59', marker='o', lw=1.5, ms=3),
               Line2D([], [], color='#414b59', marker='s', linestyle='--', lw=1.2,
                      ms=3, markerfacecolor='white')],
              ['Actual error', 'Three-term bound'], loc='center right',
              bbox_to_anchor=(1, .65), frameon=False, handlelength=1.7,
              borderpad=.15, labelspacing=.3)
    ax.set_ylim(0, max(r['mean_total_bound'] for r in rows
                      if r['perturbation'] == PLOT_EPS)*1.22)
    selected = [r for r in rows if r['tau'] == .5 and r['perturbation'] == PLOT_EPS]
    xpos = np.arange(len(KS))
    bottoms = np.zeros(len(KS))
    terms = ('mean_coefficient_term', 'mean_recovery_lipschitz_term', 'mean_omitted_dividend_term')
    names = ('Dividend error', 'Recovery-score error', 'Omitted interactions')
    for key, color, label in zip(terms, ('#e6a13c', '#70a8c7', '#c0c9d1'), names):
        values = np.array([r[key] for r in selected])
        bx.bar(xpos, values, .53, bottom=bottoms, label=label, color=color,
               edgecolor='white', lw=.4)
        bottoms += values
    bx.plot(xpos, [r['mean_l1_error'] for r in selected], 'D', color='#293342',
            label='Actual error', ms=4.5)
    bx.set_xticks(xpos, ['K = '+str(k) for k in KS])
    bx.set_ylabel('Mean contribution to bound')
    bx.set_title('(b) Error sources at τ = 0.5', pad=11, loc='left')
    bx.set_ylim(0, max(bottoms)*1.65)
    bx.legend(loc='upper right', frameon=False, handlelength=1.15,
              labelspacing=.28, borderpad=.12)
    bx.grid(axis='y', color='#e2e6eb', lw=.5)
    bx.set_axisbelow(True)
    for a in (ax, bx):
        a.spines[['right', 'top']].set_visible(False)
    return save_audited(fig, 'theorem_error')


def main():
    setup_style()
    if '--plot-only' in sys.argv:
        with (DATA / 'structure_checks.csv').open(newline='', encoding='utf-8') as f:
            structural_rows = [{key: float(value) for key, value in row.items()}
                               for row in csv.DictReader(f)]
        with (DATA / 'error_plot_summary.csv').open(newline='', encoding='utf-8') as f:
            error_summary = [{key: float(value) for key, value in row.items()}
                             for row in csv.DictReader(f)]
        examples = json.loads((DATA / 'structure_examples.json').read_text(encoding='utf-8'))
        results = json.loads((DATA / 'summary.json').read_text(encoding='utf-8'))
    else:
        structural_rows, error_summary, examples, results = run_checks()
        (DATA / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    results['figure_audits'] = {
        'theorem_structure': plot_structure(structural_rows, examples),
        'theorem_error': plot_error(error_summary)}
    (DATA / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
