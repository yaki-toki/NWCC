# paper

The manuscript (IEEE TCSVT) is kept out of the repository until it is submitted (user decision 2026-10-21: "깃헙에서 논문은 제출하기 전 까지 올라가지 않도록 해 주세요."). Everything in this folder except this file is git-ignored.

Rules for the local manuscript:
- every number is generated from a file under `experiments/` by `python -m wvad.paper.numbers` (macros in `generated/numbers.tex`); no result is typed by hand;
- figures are produced by `python -m wvad.paper.figures` in the style of the earlier TCSVT manuscript (Times/STIX at placed size, 8 pt lettering, Okabe-Ito colours fixed per method, symbols only in method figures);
- references in `references.bib` are checked against a primary record (arXiv, DOI, proceedings);
- E023-dependent values are red `\pend{...}` placeholders until its analysis is committed.
