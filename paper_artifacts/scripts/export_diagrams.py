"""Export the seven editable diagram PDFs on white at 600 DPI."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT/'tools'/'python-packages', ROOT.parent/'tools'/'python-packages'):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
import pymupdf as fitz

NAMES = ('motivation', 'problem_solution', 'overview', 'allocation', 'search',
         'experimental_environment', 'sensor_case')
audit = {}
for name in NAMES:
    with fitz.open(ROOT/'figures'/f'{name}.pdf') as document:
        assert len(document) == 1, name
        page = document[0]
        page.get_pixmap(dpi=600, alpha=False).save(ROOT/'figures'/f'{name}.png')
        sizes = [span['size'] for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']
                 if span['text'].strip()]
        audit[name] = {'width_mm': page.rect.width*25.4/72,
                       'height_mm': page.rect.height*25.4/72,
                       'minimum_native_glyph_pt': min(sizes), 'png_dpi': 600}
        assert min(sizes) >= 8.0, (name, min(sizes))
(ROOT/'data').mkdir(exist_ok=True)
(ROOT/'data'/'diagram_export_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('Exported seven diagrams; all native glyphs are at least 8 PDF pt.')
