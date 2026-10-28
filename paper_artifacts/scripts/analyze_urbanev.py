"""Audit a pinned public UrbanEV occupancy table and its prior sensitivity.

This calculation uses aggregate hourly occupied-pile series. It does not create
vehicle arrivals, departures, energy requests, or mobile-charging outcomes.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT/'tools'/'python-packages', ROOT.parent/'tools'/'python-packages'):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as fitz

SOURCE = ROOT/'data'/'external'/'urbanev'
OUT = ROOT/'data'/'urbanev_audit'
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    actual = hashlib.sha256((SOURCE/record['name']).read_bytes()).hexdigest()
    if actual != record['sha256']:
        raise ValueError('Source checksum mismatch: '+record['name'])
frame = pd.read_csv(SOURCE/'occupancy.csv')
info = pd.read_csv(SOURCE/'inf.csv')
time = pd.DatetimeIndex(pd.to_datetime(frame.pop('time')))
if time.has_duplicates or not time.is_monotonic_increasing:
    raise ValueError('Time index is not unique and increasing.')
if not time.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
    raise ValueError('The published hourly grid has a gap.')
capacity = info.groupby('TAZID')['charge_count'].sum().reindex(frame.columns.astype(int))
if capacity.isna().any() or (capacity <= 0).any():
    raise ValueError('Missing or nonpositive zone capacity.')
counts = frame.to_numpy(dtype=float)
caps = capacity.to_numpy(dtype=float)
missing = int(np.isnan(counts).sum())
if missing or not np.isfinite(counts).all():
    raise ValueError('This audit requires the complete published matrix.')
negative = int((counts < 0).sum())
above_capacity = int((counts > caps[None, :]+1e-9).sum())
if negative or above_capacity:
    raise ValueError('Published occupancy violates the declared capacity bounds.')
ratio = counts/caps[None, :]
q1, q3 = np.quantile(ratio, [.25, .75], axis=0)
iqr = q3-q1
lower = np.maximum(0., q1-1.5*iqr)
upper = np.minimum(1., q3+1.5*iqr)
# Winsorization is a sensitivity intervention, not a claim of erroneous peaks.
winsor = np.clip(ratio, lower[None, :], upper[None, :])
weighted = counts.sum(axis=1)/caps.sum()
equal_zone = ratio.mean(axis=1)
weighted_winsor = (winsor*caps[None, :]).sum(axis=1)/caps.sum()
profile = pd.DataFrame({'time': time, 'capacity_weighted': weighted,
                        'equal_zone': equal_zone,
                        'capacity_weighted_iqr': weighted_winsor})
profile['hour'] = time.hour
profile['month'] = time.strftime('%Y-%m')
hourly = profile.groupby('hour')[['capacity_weighted','equal_zone','capacity_weighted_iqr']].mean()
monthly = profile.groupby('month')[['capacity_weighted','equal_zone','capacity_weighted_iqr']].mean()
hourly.to_csv(OUT/'hourly_profiles.csv')
monthly.to_csv(OUT/'monthly_profiles.csv')
np.savetxt(OUT/'zone_capacity_and_mean.csv',
           np.column_stack([capacity.index,caps,ratio.mean(axis=0),winsor.mean(axis=0)]),
           delimiter=',',header='TAZID,capacity,mean_ratio,mean_ratio_iqr',comments='')
summary = {
    'repository_commit': manifest['commit'],
    'source': 'Pinned GitHub hourly occupied-pile table, divided by cleaned TAZ capacities.',
    'scope': 'Distribution-prior audit; no mobile-charging outcomes or synthetic sessions.',
    'timestamps': len(time), 'regions': len(caps), 'station_rows': len(info),
    'charging_piles': int(caps.sum()), 'cells': int(counts.size),
    'start': str(time[0]), 'end': str(time[-1]), 'missing_cells': missing,
    'negative_cells': negative, 'above_capacity_cells': above_capacity,
    'fractional_count_cells': int((abs(counts-np.round(counts))>1e-8).sum()),
    'capacity_weighted_mean_pct': float(weighted.mean()*100),
    'equal_zone_mean_pct': float(equal_zone.mean()*100),
    'weighting_shift_pp': float((equal_zone.mean()-weighted.mean())*100),
    'iqr_changed_cells': int((abs(ratio-winsor)>1e-12).sum()),
    'iqr_changed_cells_pct': float((abs(ratio-winsor)>1e-12).mean()*100),
    'iqr_capacity_weighted_mean_pct': float(weighted_winsor.mean()*100),
    'iqr_weighted_shift_pp': float((weighted_winsor.mean()-weighted.mean())*100),
    'maximum_hourly_weighting_profile_gap_pp': float(abs(hourly.equal_zone-hourly.capacity_weighted).max()*100),
    'capacity_weighted_hour_profile_min_pct': float(hourly.capacity_weighted.min()*100),
    'capacity_weighted_hour_profile_max_pct': float(hourly.capacity_weighted.max()*100),
    'peak_hour': int(hourly.capacity_weighted.idxmax()),
    'minimum_month': str(monthly.capacity_weighted.idxmin()),
    'maximum_month': str(monthly.capacity_weighted.idxmax()),
    'minimum_month_mean_pct': float(monthly.capacity_weighted.min()*100),
    'maximum_month_mean_pct': float(monthly.capacity_weighted.max()*100),
    'occupancy_ratio_quantiles': dict(zip(['q0','q25','q50','q75','q95','q99','q100'],
                                     map(float,np.quantile(ratio,[0,.25,.5,.75,.95,.99,1]))))}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
    'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':8.5,
    'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42})
fig, ax = plt.subplots(1,2,figsize=(182/25.4,80/25.4))
fig.subplots_adjust(left=.075,right=.99,bottom=.33,top=.84,wspace=.32)
for key,color,label in [('capacity_weighted','#0072B2','Capacity weighted'),
                         ('equal_zone','#D55E00','Equal zone'),
                         ('capacity_weighted_iqr','#009E73','IQR sensitivity')]:
    ax[0].plot(hourly.index,hourly[key]*100,label=label,color=color,lw=1.8,
               linestyle='--' if key=='capacity_weighted_iqr' else '-')
ax[0].set(xlabel='Hour of day',ylabel='Mean occupancy (%)',xticks=[0,6,12,18,23])
ax[0].set_title('(a) Hourly prior sensitivity',loc='left',pad=9)
xx=np.arange(len(monthly)); width=.36
ax[1].bar(xx-width/2,monthly.capacity_weighted*100,width,color='#0072B2',label='Published series')
ax[1].bar(xx+width/2,monthly.capacity_weighted_iqr*100,width,color='#009E73',label='IQR sensitivity')
ax[1].set_xticks(xx,[x[2:] for x in monthly.index],rotation=35,ha='right')
ax[1].set(xlabel='Month (YY-MM)',ylabel='Capacity-weighted occupancy (%)')
ax[1].set_title('(b) Month and preprocessing',loc='left',pad=9)
fig.legend(*ax[0].get_legend_handles_labels(),frameon=False,ncol=3,
           loc='lower center',bbox_to_anchor=(.53,.01))
for a in ax:
    a.spines[['top','right']].set_visible(False)
    a.grid(axis='y',alpha=.22);a.set_axisbelow(True)
for ext in ['pdf','png']:
    fig.savefig(ROOT/'figures'/f'urbanev_prior.{ext}',dpi=600,facecolor='white')
with fitz.open(ROOT/'figures'/'urbanev_prior.pdf') as document:
    sizes=[s['size'] for b in document[0].get_text('dict')['blocks']
           for line in b.get('lines',[]) for s in line['spans'] if s['text'].strip()]
    assert min(sizes)>=8., min(sizes)
print(json.dumps(summary,indent=2))
