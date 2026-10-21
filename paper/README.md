# paper

The manuscript (IEEE TCSVT) is drafted while E023 runs (user decision 2026-10-21: "실험을 진행 하면서 논문의 초안을 작성"). Until E023's analysis is committed, every value that depends on it is printed as a red `\pend{...}` placeholder; the wording of the claims follows the branch "condition (2) met" of `experiments/E023_localisation_baselines/PLAN.md` ("결과의 쓰임") and is rewritten from the committed verdict.

Rules:
- every number in the manuscript is generated from a file under `experiments/` by `python -m wvad.paper.numbers` (macros in `generated/numbers.tex`); no result is typed by hand;
- figures are produced by `python -m wvad.paper.figures` in the style of the earlier TCSVT manuscript (Times/STIX at placed size, 8 pt lettering, Okabe-Ito colours fixed per method, symbols only in method figures);
- references in `references.bib` are checked against a primary record (arXiv, DOI, proceedings);
- the LaTeX source, figures and generated macros are tracked; build products are git-ignored.
