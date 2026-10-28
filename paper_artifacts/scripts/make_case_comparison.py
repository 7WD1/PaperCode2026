"""Draw the reported critical-relay execution and responsibility comparison.

Panels (a,b) use the author's execution checkpoints and specified path state.
Markers are observations and energy lines are visual guides between them.
Panels (c,d) preserve the exact task-dividend fractions and a separate recovery
probe denominator. The aggregate of C1 and C3 never implies individual RC shares.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import sys

PAPER = Path(__file__).resolve().parents[1]
for directory in (PAPER / 'tools/python-packages', PAPER.parent / 'tools/python-packages'):
    if directory.exists():
        sys.path.insert(0, str(directory))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import pymupdf as fitz
from PIL import Image

WIDTH_MM, HEIGHT_MM = 182.0, 70.0
SOURCE = PAPER / 'data/case_comparison.json'
EXECUTION = PAPER / 'data/case_execution.json'
TARGET = PAPER / 'figures/case_comparison'
QA = PAPER / 'data/case_comparison_qa'
ROUND_QA = QA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    execution = json.loads(EXECUTION.read_text(encoding='utf-8'))
    shares = data['responsibility']
    standard = tuple(Fraction(shares['standard'][key]) for key in
                     ('relay_route_exact', 'other_routes_aggregate_exact'))
    resource = tuple(Fraction(shares['resource_conditioned'][key]) for key in
                     ('relay_route_exact', 'other_routes_aggregate_exact'))
    recovery = data['local_recovery_probe']
    kappa = Fraction(recovery['recovery_difficulty_exact'])
    not_restored = Fraction(recovery['not_restored_fraction_exact'])
    restored = Fraction(recovery['restored_fraction_exact'])
    assert sum(standard) == sum(resource) == 1
    assert not_restored == kappa and restored == 1 - kappa
    assert not_restored + restored == 1
    transport = shares['redistribution']
    relay_retained = Fraction(transport['relay_to_relay_exact'])
    others_retained = Fraction(transport['others_to_others_exact'])
    transferred = Fraction(transport['others_to_relay_exact'])
    assert relay_retained == standard[0]
    assert others_retained + transferred == standard[1]
    assert relay_retained + transferred == resource[0]
    assert others_retained == resource[1]
    assert standard[0] == Fraction(1, 3)
    assert resource[0] > resource[1]
    times = execution['time_h']
    standard_energy = execution['energy_percent']['standard']
    rc_energy = execution['energy_percent']['resource_conditioned']
    path = execution['critical_path_availability']
    assert times == [0, 2, 4, 6, 8, 10, 12]
    assert all(len(series) == len(times) for method in
               (standard_energy, rc_energy) for series in method.values())
    assert standard_energy['N7'] == [35, 15, 0, 0, 0, 0, 0]
    assert path['standard'] == [1, 1, 0, 0, 0, 0, 0]
    assert path['resource_conditioned'] == [1] * len(times)
    assert all(value > 0 for route in rc_energy.values() for value in route)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH_MM), ylim=(0, HEIGHT_MM))
    ax.axis('off')
    ink, secondary = '#173746', '#4A626D'
    relay, other = '#087FAD', '#DDE9EE'
    difficult, recovered = '#C86924', '#F2DFCA'
    standard_color, rc_color = '#A53A35', '#28764F'

    def label(x, y, text, size=8.5, color=ink, weight='normal', ha='left'):
        return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                       ha=ha, va='center', clip_on=False)

    def stacked(x, y, width, height, values, colors, texts, textcolors):
        offset = 0.
        for value, color, text, textcolor in zip(values, colors, texts, textcolors):
            segment = width * float(value)
            ax.add_patch(Rectangle((x + offset, y), segment, height,
                                  facecolor=color, edgecolor='white', linewidth=.6))
            label(x + offset + segment / 2, y + height / 2, text,
                  size=9., color=textcolor, weight='bold', ha='center')
            offset += segment
        ax.add_patch(Rectangle((x, y), width, height, facecolor='none',
                              edgecolor='#C2D1D8', linewidth=.45))

    # Execution panels use the supplied two-hour checkpoints. No interpolated
    # sample, confidence interval, safety threshold, or global graph rate is added.
    label(2., 67., '(a) Relay N7 energy', size=9., weight='bold')
    ax.plot([13., 20.], [62.7, 62.7], color=standard_color, lw=1.25,
            linestyle=(0, (4, 2)), marker='o', markersize=2.5, markevery=[0])
    label(22., 62.7, 'Standard', color=standard_color)
    ax.plot([53., 60.], [62.7, 62.7], color=rc_color, lw=1.25,
            marker='s', markersize=2.5, markevery=[0])
    label(62., 62.7, 'RC', color=rc_color)

    def chart_axes(x_mm, y_mm, width_mm, height_mm):
        plot = fig.add_axes([x_mm / WIDTH_MM, y_mm / HEIGHT_MM,
                             width_mm / WIDTH_MM, height_mm / HEIGHT_MM])
        plot.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            plot.spines[side].set_color('#9AACB5')
            plot.spines[side].set_linewidth(.6)
        plot.tick_params(axis='both', labelsize=8.5, colors=secondary,
                         length=2.3, width=.6, pad=2.)
        plot.set_xlim(0, 12)
        plot.set_xticks([0, 4, 8, 12])
        plot.set_axisbelow(True)
        return plot

    energy_ax = chart_axes(13., 37., 73., 22.)
    energy_ax.set_ylim(-1., 40.)
    energy_ax.set_yticks([0, 20, 40])
    energy_ax.grid(axis='y', color='#E4EBEE', lw=.5)
    energy_ax.plot(times, standard_energy['N7'], color=standard_color,
                   lw=1.25, linestyle=(0, (4, 2)), marker='o', markersize=3.,
                   markeredgecolor='white', markeredgewidth=.4, clip_on=False)
    energy_ax.plot(times, rc_energy['N7'], color=rc_color, lw=1.35,
                   marker='s', markersize=3., markeredgecolor='white',
                   markeredgewidth=.4, clip_on=False)
    energy_ax.set_ylabel('N7 energy (%)', fontsize=8.5, color=ink, labelpad=3.)
    energy_ax.annotate('4 h outage', xy=(4, 0), xytext=(5.1, 6.5),
                       fontsize=8.5, color=standard_color,
                       arrowprops=dict(arrowstyle='->', color=standard_color,
                                       lw=.65, shrinkA=2., shrinkB=3.))

    label(2., 29.4, '(b) N3–N7–N12 path availability', size=9., weight='bold')
    path_ax = chart_axes(13., 10., 73., 13.8)
    path_ax.set_ylim(-.08, 1.08)
    path_ax.set_yticks([0, 1])
    path_ax.step(times, path['resource_conditioned'], where='post',
                 color=rc_color, lw=1.6, clip_on=False)
    # The paths coincide until 4 h: dashes show the Standard line over the
    # solid RC line at the same binary state, without a fabricated offset.
    path_ax.step(times, path['standard'], where='post', color=standard_color,
                 lw=1.35, linestyle=(0, (4, 2)), clip_on=False)
    path_ax.set_ylabel('Availability', fontsize=8.5, color=ink, labelpad=3.)
    path_ax.set_xlabel('Time (h)', fontsize=8.5, color=ink, labelpad=2.)
    path_ax.annotate('Outage from 4 h', xy=(4, 0), xytext=(5.1, .42),
                     fontsize=8.5, color=standard_color,
                     arrowprops=dict(arrowstyle='->', color=standard_color,
                                     lw=.65, shrinkA=2., shrinkB=3.))

    # Keep the allocation and recovery diagrams at their native physical font
    # sizes in the right column. Only ribbon thickness is reduced for the new
    # explicit ranking row; all mass ratios remain exact.
    dx, dy = 93.5, 5.

    def rlabel(x, y, text, **kwargs):
        return label(x + dx, y + dy, text, **kwargs)

    rlabel(2., 62., '(c) Task-dividend responsibility', size=9., weight='bold')
    rlabel(2., 58., 'Denominator: same three-route dividend', color=secondary)
    rlabel(20., 53.9, 'Standard', size=9., weight='bold', ha='center')
    rlabel(68.5, 53.9, 'RC', size=9., weight='bold', ha='center')
    rlabel(20., 49.6, shares['standard']['verdict'], color=standard_color,
           weight='bold', ha='center')
    rlabel(68.5, 49.6, shares['resource_conditioned']['verdict'], color=rc_color,
           weight='bold', ha='center')
    ax.annotate('', xy=(60. + dx, 53.9 + dy), xytext=(33. + dx, 53.9 + dy),
                arrowprops=dict(arrowstyle='->', color='#8CA0AA', lw=.75))

    # Every ribbon has one constant thickness proportional to an exact
    # fraction. Node gaps are empty space and never count as transported mass.
    left, right, node_width, scale = 20. + dx, 67. + dx, 2., 15.
    source_c2, source_others = 41. + dy, 28. + dy
    target_c2, target_others = 38.2 + dy, 28. + dy

    def ribbon(y0, y1, mass, color):
        x0, x1 = left + node_width, right
        height = scale * float(mass)
        bend = (x1 - x0) * .42
        points = [(x0, y0), (x0 + bend, y0), (x1 - bend, y1), (x1, y1),
                  (x1, y1 + height), (x1 - bend, y1 + height),
                  (x0 + bend, y0 + height), (x0, y0 + height), (x0, y0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(points, codes), facecolor=color,
                               edgecolor='white', lw=.5, alpha=.96))

    ribbon(source_c2, target_c2 + scale * float(transferred), relay_retained, relay)
    ribbon(source_others, target_others, others_retained, other)
    ribbon(source_others + scale * float(others_retained), target_c2,
           transferred, difficult)
    for x, bottom, value, color in (
            (left, source_c2, standard[0], relay),
            (left, source_others, standard[1], other),
            (right, target_c2, resource[0], relay),
            (right, target_others, resource[1], other)):
        ax.add_patch(Rectangle((x, bottom), node_width, scale * float(value),
                              facecolor=color, edgecolor='#B5C8D0', lw=.45))

    rlabel(2., 44.6, 'C2', size=9., weight='bold')
    rlabel(2., 41., '1/3', size=9.)
    rlabel(2., 35., 'C1 + C3')
    rlabel(2., 31.4, '2/3', size=9.)
    rlabel(73., 44.6, 'C2', size=9., weight='bold')
    rlabel(73., 41., '0.52', size=9.)
    rlabel(73., 34., 'C1 + C3')
    rlabel(73., 30.4, '0.48', size=9.)
    rlabel(2., 24.4, '14/75 transfer; standard routes: 1/3 each', color=secondary)
    ax.plot([2. + dx, 86.4 + dx], [21.4 + dy, 21.4 + dy],
            color='#D0DAE0', linewidth=.55)

    rlabel(2., 18.2, '(d) Fixed local recovery probe', size=9., weight='bold')
    rlabel(2., 14.25, 'Denominator: C2 committed service', color=secondary)
    probe_x, probe_width, probe_y = 2. + dx, 84.4, 6.5 + dy
    stacked(probe_x, probe_y, probe_width, 5.6,
            (not_restored, restored), (difficult, recovered),
            ('0.85', '0.15'), ('white', ink))
    label(probe_x + probe_width * float(not_restored) / 2, 2.6 + dy,
          'Not restored', ha='center')
    label(probe_x + probe_width * (float(not_restored) + float(restored) / 2),
          2.6 + dy, 'Restored', ha='center')

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    ROUND_QA.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(TARGET.with_suffix('.png'), dpi=600)
    plt.close(fig)

    doc = fitz.open(TARGET.with_suffix('.pdf'))
    page = doc[0]
    spans = [s for b in page.get_text('dict')['blocks']
             for line in b.get('lines', []) for s in line['spans'] if s['text'].strip()]
    minimum = min(s['size'] for s in spans)
    outside = [s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
    assert minimum >= 8.49 and not outside, (minimum, outside)
    image = Image.open(TARGET.with_suffix('.png'))
    alpha = image.getchannel('A').getextrema() if image.mode == 'RGBA' else None
    assert alpha in (None, (255, 255))
    for directory in (QA, ROUND_QA):
        page.get_pixmap(dpi=150).save(directory / 'native_preview.png')
        page.get_pixmap(dpi=96).save(directory / 'insertion_96dpi.png')
        panels = {'panel_a_zoom.png': (0, 0, 90, 37),
                  'panel_b_zoom.png': (0, 37, 90, 70),
                  'panel_c_zoom.png': (93.5, 0, 182, 44.5),
                  'panel_d_zoom.png': (93.5, 44.5, 182, 70)}
        for filename, bounds in panels.items():
            clip = fitz.Rect(*(value * 72 / 25.4 for value in bounds))
            page.get_pixmap(dpi=450, clip=clip).save(directory / filename)
    audit = {
        'source_sha256': digest(SOURCE), 'script_sha256': digest(Path(__file__)),
        'execution_source_sha256': digest(EXECUTION),
        'width_mm': page.rect.width * 25.4 / 72,
        'height_mm': page.rect.height * 25.4 / 72,
        'minimum_pdf_glyph_pt': minimum, 'outside_text': outside,
        'font_names': sorted({s['font'] for s in spans}),
        'png_dpi': 600, 'png_pixels': list(image.size), 'png_alpha_range': alpha,
        'exact_sum_checks': 'All three fraction pairs sum exactly to one',
        'alluvial_conservation': 'relay 1/3 -> 1/3; others 2/3 -> retained 0.48 plus transfer 14/75; target relay 0.52 = 1/3 + 14/75',
        'ribbon_interpretation': 'A correspondence between responsibility allocations, not physical energy, service, or time flow',
        'recovery_check': '0.15 = 1 - 0.85; the service and task-dividend denominators are separate',
        'comparison_check': 'Standard: each of C1,C2,C3 receives 1/3 (uniform tie). RC: C2=0.52 exceeds the aggregate 0.48 of C1 and C3 and therefore ranks first; no individual RC values are inferred.',
        'execution_check': 'Author-reported two-hour energy checkpoints for N3,N7,N12 are retained in case_execution.json. Only N7 is plotted. Standard N7 is zero from 4 through 12 h; all RC path nodes remain positive.',
        'path_scope': 'Binary availability of the named N3-N7-N12 path over 0-12 h, not whole-network connectivity or coverage.',
        'plot_interpolation': 'Markers identify the reported energy checkpoints; joining segments are visual guides. The path step states and outage time are author-supplied.',
        'pdf_sha256': digest(TARGET.with_suffix('.pdf')),
        'png_sha256': digest(TARGET.with_suffix('.png')),
        'text_spans': [{'text': s['text'], 'font_pt': s['size'], 'bbox': list(s['bbox'])} for s in spans]
    }
    for directory in (QA, ROUND_QA):
        (directory / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'text_spans'}, indent=2))


if __name__ == '__main__':
    main()
