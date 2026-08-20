# Report

A short LaTeX report (`main.tex`) and a beamer deck (`slides.tex`) about CPE's practical
usefulness as a safety technique. Both share `lib/packages.tex` and `bib/refs.bib`.

## Build

```bash
./build.sh          # both PDFs into output/
./build.sh main     # just the report
./build.sh slides   # just the deck
```

TeX binaries are at `/Library/TeX/texbin/` (not on PATH).

## Content

One folder per section under `content/`, each holding:
- `notes.md` — the user's raw notes. **Source of truth.** The AI does not write these.
- `chapter.tex` — prose for the report, written from `notes.md`.
- `slides.tex` — the frames for that section, written from the same notes.

`main.tex` and `slides.tex` `\input` these, so notes, prose and slides for one section
stay side by side. Put each sentence on its own line in `.tex` files (better diffs,
invisible in the PDF).

`\todo[inline]{...}` marks gaps; switch the todonotes option in `lib/packages.tex` to
`[disable]` for a clean build.

## Plots

Each figure is a self-contained folder under `plots/`, with no shared code between
folders:
- `extract_data.py` — reads raw results from `experiments/` and writes `data.json`.
  Must record the repo's git commit and branch under an `experiment_repo` key.
- `plot_*.py` — reads only `data.json`, writes a `.png` next to it.

Scripts declare their own dependencies with a PEP 723 header and run outside the project
venv (the project's torch stack is not installable on this Mac):

```bash
uv run --no-project extract_data.py
uv run --no-project plot_scaling.py
```

`\graphicspath` includes `plots/`, so a figure is included as
`\includegraphics{pwlock_scaling/pwlock_scaling.png}`.

## Bibliography

`bib/refs.bib` is hand-maintained: no Zotero sync, no automated extraction. Full text of
a paper worth keeping around goes in `bib/<key>/<key>_fulltext.md` beside its PDF, as in
`bib/mack2026/` (the CPE paper). Never modify `_fulltext.md` files.
