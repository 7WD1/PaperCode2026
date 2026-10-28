# Editable figures and controlled analyses

These sources accompany the manuscript's mathematical and visual artifacts.
They use the same definitions as the main implementation. The enumerated theory
games are constructed calculations. The two case inputs retain the author-reported
execution checkpoints and responsibility fractions; plotting them does not rerun
the physical case. No benchmark observations are reconstructed from tables.

Install the optional plotting dependencies with `pip install -e '.[analysis]'`.
From the repository root, run the following in order:

```text
python paper_artifacts/scripts/make_toy.py
python paper_artifacts/scripts/verify_theory.py
python paper_artifacts/scripts/make_theory_surfaces.py
python paper_artifacts/scripts/analyze_urbanev.py
python paper_artifacts/scripts/make_case_comparison.py
```

The scripts write their calculations under `data/` and vector/600-DPI figures
under `figures/`, relative to this directory. The UrbanEV input is a pinned,
checksum-verified public snapshot under its included CC0 license. It supplies an
aggregate occupancy prior audit, rather than vehicle session labels.

Seven native diagrams are editable TikZ. Their standalone wrappers are in
`editable_figures/`. Compile them with pdfLaTeX and the TikZ/standalone packages.
On Windows, `editable_figures/rebuild.ps1` accepts `-PdfLaTeX` and `-Python` paths.
The overview's numeric literature labels match the submitted manuscript.

Source and data authorship of third-party public inputs remains with their
respective authors. The included UrbanEV manifest records its original source
and commit so the public-data audit can be reproduced.
