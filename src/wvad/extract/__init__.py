"""Dense snippet feature extraction of E021 (PLAN.md "구현").

    grid        snippet grid, evaluator windows, window assignment, m-picks, d16r draws, 1-s
                grid: the single source of every index rule (needs numpy only)
    video       production decoding (all frames 0..T-1 grabbed, used frames retrieved,
                failed frames replaced) and header reads
    preprocess  BGR -> RGB, resize, uint8 hand-off; 256-entry lookup-table normalisation
    encoders    frozen X-CLIP (base, large) and SigLIP encoders, pinned revisions, precision
    legacy      clip positions and the two decode rules of the earlier caches (gates only)
    dense       production extraction into one memory-mapped cache per (dataset, encoder)
    derive      32-unit caches d16c / d16r / d16m / d16m2 (and d16r1 / d16r2) from the snippets
    gate        extraction gates (x1)-(x6), the (x0) unit-test record, frame hashes, floors,
                environment records

Command line: ``python -m wvad.extract {gate,bench,header,dense,derive,check} ...``. Only
``grid`` is imported by the training, scoring and analysis code; the other modules need the
optional ``extract`` dependencies (opencv-python, transformers).
"""
