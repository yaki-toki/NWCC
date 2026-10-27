# paper

The manuscript (IEEE TCSVT) is kept out of the repository until it is submitted (user decision 2026-10-21: "깃헙에서 논문은 제출하기 전 까지 올라가지 않도록 해 주세요."). Everything in this folder except this file, and the generators in `src/wvad/paper/`, are git-ignored.

Rules for the local manuscript:
- every number is generated from a file under `experiments/` by `python -m wvad.paper.numbers` (macros in `generated/numbers.tex`); no result is typed by hand;
- figures are produced by `python -m wvad.paper.figures` in the style of the earlier TCSVT manuscript (Times/STIX at placed size, 8 pt lettering, Okabe-Ito colours fixed per method, symbols only in method figures);
- references in `references.bib` are checked against a primary record (arXiv, DOI, proceedings);
- E023-dependent values are red `\pend{...}` placeholders until its analysis is committed.

Commands (local):
```bash
python -m wvad.paper.figures                 # paper/figs/*.pdf (+ .png previews): diagrams
python -m wvad.paper.evidence                # experiments/E024_localisation_method/figures_20261027/
                                             #   (record.json, per_video.csv, case_*.csv); decoded frames
                                             #   under F:/Vision/wvad_runs/e024_figures; needs UBnormal.zip,
                                             #   the stage-3 score files and the original UCF/XD videos
python -m wvad.paper.evidence_figs           # paper/figs/fig_{input,yield,cases,review,supp_cases}.pdf
python -m wvad.paper.result_figs             # paper/figs/fig_{contrasts,dev,base}.pdf from the committed result
                                             #   files (e024_stage3.json, e024_stage1.json, e023_summary.json)
python -m wvad.paper.numbers                 # paper/generated/numbers.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper && pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp   # supplementary material
```
