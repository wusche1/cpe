# Report

A short LaTeX report (`main.tex`) and a beamer deck (`slides.tex`) on whether CPE is
practically useful as an AI-safety technique. Both documents are built from the same
`content/` sections, the same generated figures and tables in `exhibits/`, and the same
bibliography in `bib/refs.bib`.

This is a fast, deliberately crude write-up, not a submission-ready paper. Prefer getting
a correct number on the page over polishing the layout.

## Layout

```
main.tex              report root: \input's content/<section>/chapter.tex
slides.tex            deck root:   \input's content/<section>/slides.tex
lib/packages.tex      preamble shared by both documents
build.sh              builds one or both PDFs into output/
content/<section>/    notes.md + chapter.tex + slides.tex per section
exhibits/<name>/      one figure or table group: extract_data.py + a render script
bib/refs.bib          hand-maintained bibliography
bib/<key>/            per-paper PDF and extracted full text
output/               build artifacts (gitignored)
```

## Build

```bash
./build.sh          # both PDFs into output/
./build.sh main     # just the report
./build.sh slides   # just the deck
```

TeX binaries are at `/Library/TeX/texbin/` and are **not on PATH**; `build.sh` calls them
by absolute path. Each document gets pdflatex → bibtex → pdflatex × 2, so citations and
references resolve in one invocation. `bibtex` is allowed to fail (`|| true`) because a
document with no `\cite` yet is not an error.

After a build, check `output/main.log` for `Undefined control sequence`, `LaTeX Error`,
`Overfull`, and undefined citations. Ignore natbib's "There were undefined citations"
if it comes from the *first* pdflatex pass, before the `.bbl` exists; only the final
pass matters.

## Content

One folder per section under `content/`, each holding three files:

- `notes.md` — the user's raw notes. **Source of truth, and the user writes them, not
  you.** Do not edit these unless explicitly asked. If a claim is not in the notes (or in
  the repo's data or `lab_notebook/`), it does not go in the report.
- `chapter.tex` — report prose for that section, written from `notes.md`.
- `slides.tex` — the frames for that section, from the same notes.

Both document roots `\input` these, so notes, prose and slides for a section sit side by
side. Conventions:

- **Only write `chapter.tex` / `slides.tex` when asked.** The user drafts notes; the
  agent turns notes into LaTeX.
- **One sentence per line** in `.tex` files. Better diffs, invisible in the PDF.
- `\todo[inline]{...}` marks a gap. Set the todonotes option in `lib/packages.tex` to
  `[disable]` for a clean final build.
- Numbers in prose must match the generated tables and figures. Both ultimately come from
  `experiments/`; never retype a number that a `data.json` already holds.

## Exhibits (figures and tables)

Everything the documents show is generated from data, never typed by hand — **a table is
built exactly like a plot**, only its render step writes `.tex` instead of `.png`. The
folder is called `exhibits/` rather than `plots/` because it holds both.

One folder per exhibit, each **fully self-contained**, holding:

- `extract_data.py` — reads raw results from `experiments/`, writes `data.json`.
  Must record the repo's git commit and branch under an `experiment_repo` key.
- a render script — reads **only** `data.json`, writes the artifact beside it:
  `plot_*.py` → `.png`, `table_*.py` → `.tex`.

Folders **do not share code**, and each has its own `extract_data.py` even when two
exhibits draw on the same runs. The duplication is deliberate: an exhibit can be
re-rendered, edited or deleted without any chance of breaking another. Current folders:

```
exhibits/pwlock_scaling/   extract_data.py + plot_scaling.py  -> pwlock_scaling.png
exhibits/pwlock_strength/  extract_data.py + plot_strength.py -> pwlock_strength.png
exhibits/pwlock_tables/    extract_data.py + table_ladder.py  -> table_dense.tex, table_moe.tex
```

Splitting extraction from rendering matters just as much: re-styling an exhibit must
never re-read the experiment tree.

Scripts declare their own dependencies in a PEP 723 header and run **outside** the
project venv, because the repo's torch stack does not install on this Mac
(`pysqlite3-binary` has no macOS ARM wheel). Always run both steps of a folder:

```bash
cd exhibits/pwlock_tables
uv run --no-project extract_data.py     # experiments/ -> data.json
uv run --no-project table_ladder.py     # data.json -> table_dense.tex, table_moe.tex
```

When two exhibits cover the same runs, changing a run mapping in one means changing it in
the other. Cross-check them by comparing the shared values in their `data.json` files.

Note for zsh: `for s in plot_*.py table_*.py` aborts the whole loop when one glob has no
match. Run the scripts in a folder explicitly rather than globbing across folders.

### Including them

`\graphicspath` includes `exhibits/`, so a figure is:

```latex
\includegraphics[width=\textwidth]{pwlock_scaling/pwlock_scaling.png}
```

A generated `.tex` holds the **tabular only**. Caption, label and placement live in the
chapter that includes it, so regenerating a table never overwrites prose:

```latex
\begin{table}[h]
  \centering
  \small
  \input{exhibits/pwlock_tables/table_dense}
  \caption{...}
  \label{tab:pwlock-dense}
\end{table}
```

### Error bars

Recovery-style ratios get their error bars recomputed in `extract_data.py` from the raw
accuracies (binomial standard errors propagated by the delta method), not read from
result files: many runs ship no stderr, and recomputing puts every cell on the same
footing. The implementation in `exhibits/pwlock_scaling/extract_data.py` reproduces the
stderrs that *are* stored, which is what validates it. Reuse that function's shape rather
than inventing a new convention.

## Bibliography

`bib/refs.bib` is **hand-maintained**. There is no Zotero sync and no automated
extraction in this repo (unlike `../steganography-benchmark-paper`, which syncs from a
Zotero group). Do not build one; just edit the file.

### Adding a paper

1. **Get the BibTeX entry.** From arXiv's "Export BibTeX citation", the publisher, or a
   Zotero export. Do not hand-write an entry from memory — a fabricated author list or
   year is worse than no citation.
2. **Set the key to `<firstauthorlastname><year>`**, lowercase, e.g. `mack2026`,
   `beaglehole2026`. Add a letter suffix (`anthropic2026a`) if the key is taken.
3. **Paste it into `bib/refs.bib`.** Keep the fields that matter (author, title, year,
   venue or arXiv number, url/doi) and drop Zotero's local `file = {...}` paths. A short
   `note = {...}` saying why the paper is cited here is welcome; it shows up nowhere in
   the PDF but helps the next agent.
4. **Optionally add the full text** at `bib/<key>/<key>_fulltext.md`, with the PDF beside
   it as `bib/<key>/<key>.pdf`. Nothing in this repo generates these — copy the PDF in by
   hand and convert it with whatever tool is available. Worth doing for any paper whose
   claims the report leans on, since it lets an agent verify a citation without network
   access. See `bib/mack2026/` (the CPE paper) and `bib/beaglehole2026/`.
5. **Cite it and rebuild.** `./build.sh` runs bibtex, so a new citation resolves in one
   invocation.

### Citing

natbib with `plainnat`:

- `\citep{mack2026}` → "[Mack et al., 2026]" — the default for a parenthetical reference.
- `\citet{mack2026}` → "Mack et al. (2026)" — when the authors are the sentence's subject.

Rules:

- **Never modify a `_fulltext.md` file.** It is read-only source material.
- **Verify a claim against the paper before citing it for that claim.** If
  `bib/<key>/<key>_fulltext.md` exists, read the relevant part; grep it rather than
  reading all of it (they run to thousands of lines).
- Do not cite a key that is not in `refs.bib` — it renders as `[?]` and only shows up as
  a warning in the final pdflatex pass.

## Related repos and files

- `../lab_notebook/` — numbered entries with the authoritative experimental results,
  including which runs supersede which. Read the relevant entry before writing any
  results prose; entries are chronological and early sections are often explicitly marked
  wrong.
- `../experiments/<organism>/` — raw run outputs that `extract_data.py` reads.
- `../../steganography-benchmark-paper/` — the user's full paper setup (AAAI template,
  Zotero sync, RAG over the bibliography). This report deliberately uses a cut-down
  version of those conventions.
