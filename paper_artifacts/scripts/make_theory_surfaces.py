"""Five reproducible 3-D audits of the resource-conditioned allocation theory.

Run: python scripts/make_theory_surfaces.py [--plot-only]
The source is the existing 250 enumerated games and their saved perturbation
directions.  These controlled value games are not EV/WRSN benchmark results.
Each facet joins evaluated grid vertices; no interpolated samples or uncertainty
intervals are generated.  CSV files preserve every game/grid calculation.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import sys

PAPER = Path(__file__).resolve().parents[1]
for directory in (PAPER / 'tools/python-packages', PAPER.parent / 'tools/python-packages'):
    if directory.exists():
        sys.path.insert(0, str(directory))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pymupdf as fitz
from PIL import Image

SPEC = importlib.util.spec_from_file_location('reference_theory', PAPER / 'scripts/verify_theory.py')
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)

DATA = PAPER / 'data/theory_surfaces'
FIGS = PAPER / 'figures'
SOURCE = PAPER / 'data/theory/games.json'
DATA.mkdir(parents=True, exist_ok=True)
TAUS = np.array([0.1, 0.2, 0.35, 0.5, 0.8, 1.25, 2.0])
LAMBDAS = np.linspace(0, 1, 7)
ZETAS = np.array([0., .01, .03, .05, .1, .2, .4])
XIS = np.linspace(0, 1, 7)
GAMMAS = np.array([0., .025, .05, .1, .2, .3, .45])
ZETA = .03
ETA = REF.ETA
N = REF.N
ALL = tuple(REF.MEMBERS)
CHAIN = REF.SUPPORT[2]
TOL = 2e-11
WIDTH_MM, HEIGHT_MM = 88.5, 62.0
NAMES = ('envelope', 'sensitivity', 'truncation', 'identifiability', 'sharpness')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_shares(kappa, urgency, tau, lam):
    """Vectorized evaluation of REF.shares; checked against it below."""
    alpha = np.zeros_like(kappa)
    for t, ids in REF.MEMBERS.items():
        ids = list(ids)
        z = (kappa[:, t, ids] + ETA * urgency[:, t, ids]) / tau
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        alpha[:, t, ids] = (1-lam) / len(ids) + lam * exp_z / exp_z.sum(axis=1, keepdims=True)
    return alpha


def batch_allocation(dividends, alpha, support=ALL):
    return np.sum(dividends[:, list(support), None] * alpha[:, list(support), :], axis=1)


def score_error(kappa, predicted, support):
    return np.stack([np.max(np.abs(predicted[:, t, list(REF.MEMBERS[t])]
                                  - kappa[:, t, list(REF.MEMBERS[t])]), axis=1)
                     for t in support], axis=1)


def grid_summary(name, rows, xkey, ykey, fields):
    summaries = []
    for y in sorted({r[ykey] for r in rows}):
        for x in sorted({r[xkey] for r in rows}):
            group = [r for r in rows if r[xkey] == x and r[ykey] == y]
            record = {xkey: x, ykey: y, 'games': len(group)}
            for field in fields:
                vals = np.array([r[field] for r in group], dtype=float)
                record['mean_' + field] = float(vals.mean())
                record['min_' + field] = float(vals.min())
                record['max_' + field] = float(vals.max())
            summaries.append(record)
    write_csv(DATA / (name + '_per_game.csv'), rows)
    write_csv(DATA / (name + '_grid.csv'), summaries)
    return summaries


def compute():
    raw = json.loads(SOURCE.read_text(encoding='utf-8'))
    games = raw['games']
    d = np.array([g['mobius_dividends'] for g in games])
    kappa = np.array([g['kappa'] for g in games])
    urgency = np.array([g['urgency'] for g in games])
    direction = np.array([g['recovery_direction'] for g in games])
    coeff_direction = np.array([g['coefficient_direction'] for g in games])
    values = np.array([g['values'] for g in games])
    ng = len(games)
    assert ng == 250 and d.shape == (250, 32)
    reference_residual = 0.
    for tau, lam in ((.1, 1.), (.5, .5), (2., 0.)):
        batch = batch_shares(kappa, urgency, tau, lam)
        for g in (0, 125, 249):
            exact = REF.shares(kappa[g], urgency[g], tau, lam=lam)
            reference_residual = max(reference_residual, float(np.max(np.abs(batch[g]-exact))))
            reference_residual = max(reference_residual, float(np.max(np.abs(
                batch_allocation(d, batch)[g] - REF.allocation(d[g], exact)))))
    assert reference_residual < TOL
    rows = {}
    grids = {}
    audit = {}
    tail = tuple(t for t in ALL if t not in CHAIN)

    # 1. Joint coefficient, score, and support error: the exact full allocation
    # and the learned retained allocation use their own appropriate shares.
    out = []
    predicted = np.clip(kappa + ZETA * direction, 0, 1)
    c = d + ZETA * (1 + np.abs(d)) * coeff_direction
    coefficient = np.sum(np.abs(c[:, CHAIN] - d[:, CHAIN]), axis=1)
    omitted = np.sum(np.abs(d[:, tail]), axis=1)
    e = score_error(kappa, predicted, CHAIN)
    for tau in TAUS:
        for lam in LAMBDAS:
            a = batch_shares(kappa, urgency, tau, lam)
            ahat = batch_shares(predicted, urgency, tau, lam)
            actual = np.sum(np.abs(batch_allocation(c, ahat, CHAIN)-batch_allocation(d, a)), axis=1)
            score_bound = lam/tau * np.sum(np.abs(d[:, CHAIN])*e, axis=1)
            exact_share = np.sum(np.abs(d[:, CHAIN, None])
                                 * np.abs(ahat[:, CHAIN, :]-a[:, CHAIN, :]), axis=(1, 2))
            bound = coefficient + score_bound + omitted
            for g in range(ng):
                out.append(dict(game_id=games[g]['game_id'], lambda_value=float(lam), tau=float(tau),
                                zeta=ZETA, K=2, l1_error=float(actual[g]),
                                coefficient_term=float(coefficient[g]),
                                exact_weighted_share_error=float(exact_share[g]),
                                recovery_lipschitz_term=float(score_bound[g]),
                                omitted_dividend_term=float(omitted[g]), total_bound=float(bound[g]),
                                bound_excess=float(actual[g]-bound[g])))
    fields = ('l1_error', 'coefficient_term', 'exact_weighted_share_error',
              'recovery_lipschitz_term', 'omitted_dividend_term', 'total_bound', 'bound_excess')
    grids['envelope'] = grid_summary('envelope', out, 'lambda_value', 'tau', fields)
    rows['envelope'] = out
    print('computed envelope', len(out), flush=True)

    # 2. Isolate the weighted share term, so signed cancellation does not conceal
    # the softmax Lipschitz inequality. e is the actual post-clipping L-infinity error.
    out = []
    for tau in TAUS:
        a = batch_shares(kappa, urgency, tau, .5)
        for zeta in ZETAS:
            predicted = np.clip(kappa + zeta * direction, 0, 1)
            ahat = batch_shares(predicted, urgency, tau, .5)
            e = score_error(kappa, predicted, CHAIN)
            exact = np.sum(np.abs(d[:, CHAIN, None])
                           * np.abs(ahat[:, CHAIN, :]-a[:, CHAIN, :]), axis=(1, 2))
            bound = .5/tau * np.sum(np.abs(d[:, CHAIN])*e, axis=1)
            for g in range(ng):
                out.append(dict(game_id=games[g]['game_id'], zeta=float(zeta), tau=float(tau),
                                lambda_value=.5, K=2, exact_weighted_share_error=float(exact[g]),
                                recovery_lipschitz_bound=float(bound[g]),
                                bound_excess=float(exact[g]-bound[g])))
    grids['sensitivity'] = grid_summary('sensitivity', out, 'zeta', 'tau',
        ('exact_weighted_share_error', 'recovery_lipschitz_bound', 'bound_excess'))
    rows['sensitivity'] = out
    print('computed sensitivity', len(out), flush=True)

    # 3. Complete-order support audit, deliberately different from the chain
    # support used by the default method. Tail coefficients alone scale with xi.
    out = []
    a = batch_shares(kappa, urgency, .5, .5)
    for xi in XIS:
        for order in range(1, N+1):
            kept = tuple(t for t in ALL if REF.ORDER[t] <= order)
            omitted_t = tuple(t for t in ALL if REF.ORDER[t] > order)
            scaled = d.copy()
            scaled[:, list(omitted_t)] *= xi
            full = batch_allocation(scaled, a)
            truncated = batch_allocation(scaled, a, kept)
            actual = np.sum(np.abs(truncated-full), axis=1)
            bound = np.sum(np.abs(scaled[:, list(omitted_t)]), axis=1)
            for g in range(ng):
                out.append(dict(game_id=games[g]['game_id'], K=order, xi=float(xi),
                                lambda_value=.5, tau=.5, support_size=len(kept),
                                omitted_interactions=len(omitted_t), l1_tail_error=float(actual[g]),
                                omitted_mass_bound=float(bound[g]),
                                bound_excess=float(actual[g]-bound[g])))
    grids['truncation'] = grid_summary('truncation', out, 'K', 'xi',
        ('l1_tail_error', 'omitted_mass_bound', 'bound_excess'))
    rows['truncation'] = out
    print('computed truncation', len(out), flush=True)

    # 4. Every proxy coalition value uses the exact full coefficient set. Its
    # predictions therefore remain independent of all corrupted resource scores.
    membership = np.array([[int(t & s == t) for s in range(32)] for t in ALL])
    proxy_values = d[:, ALL] @ membership
    value_error = np.max(np.abs(proxy_values-values), axis=1)
    out = []
    for zeta in ZETAS:
        predicted = np.clip(kappa + zeta * direction, 0, 1)
        for lam in LAMBDAS:
            a = batch_shares(kappa, urgency, .5, lam)
            ahat = batch_shares(predicted, urgency, .5, lam)
            actual = np.sum(np.abs(batch_allocation(d, ahat)-batch_allocation(d, a)), axis=1)
            efficiency = np.abs(np.sum(batch_allocation(d, ahat), axis=1)-values[:, -1])
            e = score_error(kappa, predicted, ALL)
            bound = lam/.5 * np.sum(np.abs(d[:, ALL])*e, axis=1)
            for g in range(ng):
                out.append(dict(game_id=games[g]['game_id'], lambda_value=float(lam),
                                zeta=float(zeta), tau=.5, support_size=31,
                                l1_allocation_error=float(actual[g]),
                                max_all_32_coalition_value_error=float(value_error[g]),
                                efficiency_residual=float(efficiency[g]),
                                recovery_lipschitz_bound=float(bound[g]),
                                bound_excess=float(actual[g]-bound[g])))
    grids['identifiability'] = grid_summary('identifiability', out, 'lambda_value', 'zeta',
        ('l1_allocation_error', 'max_all_32_coalition_value_error', 'efficiency_residual',
         'recovery_lipschitz_bound', 'bound_excess'))
    rows['identifiability'] = out
    assert max(r['max_all_32_coalition_value_error'] for r in out) < TOL
    assert max(r['efficiency_residual'] for r in out) < TOL
    print('computed identifiability', len(out), flush=True)

    # 5. Two-route analytic family. Centered scores and h=(zeta,-zeta)
    # remain admissible kappa values in [0,1] with urgency fixed to zero.
    out = []
    lam = .5
    for tau in TAUS:
        for gamma in GAMMAS:
            score = np.array([.5+gamma, .5-gamma])
            corrupted = score + np.array([ZETA, -ZETA])
            assert np.min(score) >= 0 and np.max(score) <= 1
            assert np.min(corrupted) >= 0 and np.max(corrupted) <= 1
            def two_shares(b):
                exp_z = np.exp((b-b.max())/tau)
                return (1-lam)/2 + lam * exp_z/exp_z.sum()
            exact = float(np.sum(np.abs(two_shares(corrupted)-two_shares(score))))
            limit = lam * ZETA / tau
            ratio = exact / limit
            analytic_ratio = tau/ZETA * (math.tanh((gamma+ZETA)/tau)-math.tanh(gamma/tau))
            assert abs(ratio-analytic_ratio) < TOL
            assert -TOL <= ratio <= 1+TOL
            out.append(dict(gamma=float(gamma), tau=float(tau), lambda_value=lam, zeta=ZETA,
                            exact_l1_share_change=exact, lipschitz_bound=limit,
                            sharpness_ratio=ratio, analytic_ratio=analytic_ratio,
                            analytic_residual=abs(ratio-analytic_ratio), ratio_excess=ratio-1))
    write_csv(DATA/'sharpness_grid.csv', out)
    grids['sharpness'] = out
    rows['sharpness'] = out
    print('computed sharpness', len(out), flush=True)

    for name in NAMES:
        field = 'ratio_excess' if name == 'sharpness' else 'bound_excess'
        audit[name] = dict(calculations=len(rows[name]), grid_vertices=len(grids[name]),
                           maximum_bound_excess=float(max(r[field] for r in rows[name])),
                           violations=int(sum(r[field] > TOL for r in rows[name])))
        assert audit[name]['violations'] == 0
    audit['identifiability']['maximum_value_error_all_32_masks'] = float(value_error.max())
    audit['identifiability']['maximum_efficiency_residual'] = max(r['efficiency_residual'] for r in rows['identifiability'])
    audit['identifiability']['maximum_mean_allocation_error'] = max(r['mean_l1_allocation_error'] for r in grids['identifiability'])
    positive_identification = [r['mean_l1_allocation_error'] for r in grids['identifiability']
                               if r['lambda_value'] > 0 and r['zeta'] > 0]
    audit['identifiability']['nonzero_mean_allocation_error_range'] = [
        min(positive_identification), max(positive_identification)]
    audit['sharpness']['minimum_ratio'] = min(r['sharpness_ratio'] for r in rows['sharpness'])
    audit['sharpness']['maximum_ratio'] = max(r['sharpness_ratio'] for r in rows['sharpness'])
    audit['sharpness']['maximum_analytic_residual'] = max(r['analytic_residual'] for r in rows['sharpness'])
    audit['sharpness']['true_score_range'] = [.5-float(GAMMAS.max()), .5+float(GAMMAS.max())]
    audit['sharpness']['predicted_score_range'] = [.5-float(GAMMAS.max())-ZETA, .5+float(GAMMAS.max())+ZETA]
    audit['sharpness']['zero_gamma_ratios'] = [r for r in rows['sharpness'] if r['gamma'] == 0]
    game_summaries = []
    for game in games:
        record = {'game_id': game['game_id'], 'surface_checks': {}}
        for name in NAMES[:-1]:
            subset = [r for r in rows[name] if r['game_id'] == game['game_id']]
            record['surface_checks'][name] = dict(
                grid_vertices=len(subset),
                maximum_bound_excess=float(max(r['bound_excess'] for r in subset)),
                bound_violations=int(sum(r['bound_excess'] > TOL for r in subset)))
        game_summaries.append(record)
    (DATA/'per_game_audit.json').write_text(json.dumps(game_summaries, indent=2), encoding='utf-8')
    result = dict(source_file='data/theory/games.json', source_sha256=digest(SOURCE),
                  source_seed=raw['seed'], games=ng, players=N, coalitions_per_game=32,
                  game_scope='controlled signed-interaction value games; not EV/WRSN benchmarks',
                  arithmetic='numpy float64', tolerance=TOL,
                  versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__,
                                pymupdf=fitz.VersionBind),
                  reference_implementation='scripts/verify_theory.py shares and allocation',
                  batch_reference_maximum_residual=reference_residual,
                  grids=dict(lambda_values=LAMBDAS.tolist(), tau_values=TAUS.tolist(),
                             zeta_values=ZETAS.tolist(), xi_values=XIS.tolist(),
                             sharpness_gamma_values=GAMMAS.tolist()),
                  envelope_support_masks=list(CHAIN), envelope_zeta=ZETA, eta=ETA,
                  support_rules=dict(envelope='contiguous order <=2 on A-B-C-D-E',
                                     sensitivity='contiguous order <=2 on A-B-C-D-E',
                                     truncation='all nonempty subsets with order <=K, K=1..5',
                                     identifiability='all 31 nonempty subsets',
                                     sharpness='one two-route softmax vector'),
                  coefficient_perturbation='delta + zeta*(1+abs(delta))*saved_coefficient_direction',
                  recovery_perturbation='clip(kappa + zeta*saved_recovery_direction,0,1)',
                  score_error='post-clipping max_i abs(kappa_hat_iT-kappa_iT); urgency fixed',
                  truncation_family='delta_xi(T)=delta(T) on support; xi*delta(T) off support',
                  sharpness_family='b=kappa=(.5+gamma,.5-gamma), urgency=0; h=(zeta,-zeta), no clipping; lambda=.5, zeta=.03',
                  aggregation='arithmetic mean across the same 250 games; no CI or error bars',
                  facets='polygonal mesh joining evaluated grid; vertices marked explicitly; no extra samples',
                  audits=audit)
    (DATA/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return grids, result


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.,
                         'axes.labelsize': 9., 'axes.titlesize': 9.5,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'legend.fontsize': 8.5, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .55,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def surface(name, rows, xkey, ykey, zkey, boundkey, xlabel, ylabel, title, zdesc,
            xticks, yticks, mean=True):
    xs = np.array(sorted({r[xkey] for r in rows}))
    ys = np.array(sorted({r[ykey] for r in rows}))
    lookup = {(r[xkey], r[ykey]): r for r in rows}
    x, y = np.meshgrid(xs, ys)
    z = np.array([[lookup[(xx, yy)][zkey] for xx in xs] for yy in ys])
    bound = None if boundkey is None else np.array([
        [lookup[(xx, yy)][boundkey] for xx in xs] for yy in ys])
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
    ax = fig.add_axes([.025, .16, .91, .77], projection='3d', computed_zorder=False)
    ax.view_init(elev=26, azim=-132)
    ax.set_box_aspect((1.1, 1., .69), zoom=.99)
    ax.plot_surface(x, y, z, cmap='viridis', vmin=float(z.min()), vmax=float(z.max()) or 1.,
                    linewidth=.38, edgecolor='#416175', shade=False, alpha=.91,
                    rstride=1, cstride=1, antialiased=True, zorder=2)
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), color='#123a4a', s=6.5,
               linewidths=.25, edgecolors='white', depthshade=False, zorder=3)
    if bound is not None:
        ax.plot_wireframe(x, y, bound, rstride=1, cstride=1,
                          color='#D55E00', linewidth=.67, alpha=.95, zorder=4)
    ax.set_xlabel(xlabel, labelpad=0.5)
    ax.set_ylabel(ylabel, labelpad=-2.5)
    # A horizontal z description avoids a long rotated label being clipped on a
    # single-column 3-D canvas. It names the vertical coordinate explicitly.
    # The manuscript caption supplies the figure title. Preserve the horizontal
    # vertical-coordinate description while spending the reduced canvas on axes.
    fig.text(.045, .967, 'z: ' + zdesc, fontsize=8.5, va='top')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    if name == 'sharpness':
        ax.set_zticks([0, .5, 1])
    ax.tick_params(axis='both', which='major', labelsize=8.5, pad=-1.5, length=2)
    ax.zaxis.set_tick_params(labelsize=8.5, pad=0)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(0, max(float(z.max()), 0. if bound is None else float(bound.max()))*1.075 or 1.)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.96, .97, .98, 1.))
        axis.pane.set_edgecolor('#d1d9e0')
        axis._axinfo['grid'].update(color='#d8dfe5', linewidth=.4)
    handles = [Line2D([], [], color='#245971', marker='o', ms=3, linewidth=.65,
                      markeredgecolor='white', markeredgewidth=.3)]
    labels = ['Evaluated mean' if mean else 'Evaluated ratio']
    if bound is not None:
        handles.append(Line2D([], [], color='#D55E00', linewidth=.85))
        labels.append('Theorem 2 bound')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,-.007),
               ncol=len(labels), frameon=False, handlelength=1., columnspacing=.8,
               handletextpad=.35, borderpad=0., fontsize=8.5)
    target = FIGS / ('theory_surface_' + name)
    fig.savefig(target.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(fig)
    doc = fitz.open(target.with_suffix('.pdf')); page = doc[0]
    spans = [s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[])
             for s in l['spans'] if s['text'].strip()]
    minimum = min(s['size'] for s in spans)
    outside = [s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
    assert minimum >= 8.39, (name, minimum)
    assert not outside, (name, outside)
    png = Image.open(target.with_suffix('.png'))
    alpha_range = png.getchannel('A').getextrema() if png.mode == 'RGBA' else None
    assert alpha_range in (None, (255, 255))
    return dict(width_mm=page.rect.width*25.4/72, height_mm=page.rect.height*25.4/72,
                minimum_pdf_font_pt=minimum, font_names=sorted({s['font']for s in spans}),
                outside_text=outside, png_dpi=600, png_pixels=list(png.size),
                png_alpha_range=alpha_range, grid_vertices=int(x.size),
                pdf_sha256=digest(target.with_suffix('.pdf')),
                png_sha256=digest(target.with_suffix('.png')))


def main():
    if '--plot-only' in sys.argv:
        result = json.loads((DATA/'audit.json').read_text(encoding='utf-8'))
        grids = {}
        for name in NAMES:
            with (DATA/(name+'_grid.csv')).open(newline='', encoding='utf-8') as f:
                grids[name] = [{key: float(value) for key, value in row.items()}
                               for row in csv.DictReader(f)]
    else:
        grids, result = compute()
    style()
    specs = {
        'envelope': ('lambda_value','tau','mean_l1_error','mean_total_bound',
                     'Mixing λ','Temperature τ','Joint error envelope','L1 error / bound (value units)',[0,.5,1],[.5,1,2]),
        'sensitivity': ('zeta','tau','mean_exact_weighted_share_error','mean_recovery_lipschitz_bound',
                        'Perturbation ζ','Temperature τ','Weighted share sensitivity','Weighted error (value units)',[0,.2,.4],[.5,1,2]),
        'truncation': ('K','xi','mean_l1_tail_error','mean_omitted_mass_bound',
                       'Order K','Tail scale ξ','Complete-order truncation','Tail error / mass (value units)',[1,3,5],[.25,.5,1]),
        'identifiability': ('lambda_value','zeta','mean_l1_allocation_error',None,
                            'Mixing λ','Perturbation ζ','Value-fit non-identifiability','Allocation L1 error (value units)',[0,.5,1],[.1,.2,.4]),
        'sharpness': ('gamma','tau','sharpness_ratio',None,
                      'Score offset γ','Temperature τ','Softmax constant sharpness','L1 change / (λζ/τ)',[0,.2,.45],[.5,1,2])}
    result['figure_audits'] = {}
    for name in NAMES:
        result['figure_audits'][name] = surface(name, grids[name], *specs[name], mean=name!='sharpness')
        print('saved', name, result['figure_audits'][name], flush=True)
    result['source_sha256_after_run'] = digest(SOURCE)
    assert result['source_sha256_after_run'] == result['source_sha256']
    result['script_sha256'] = digest(Path(__file__))
    result['data_file_hashes'] = {p.name:digest(p) for p in sorted(DATA.glob('*.csv'))}
    (DATA/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['audits'], indent=2))


if __name__ == '__main__':
    main()
