"""E023 comparison methods on the frozen VLM features (experiments/E023_localisation_baselines).

The interface, the registry and the shared rules are in ``wvad.baselines.common`` (its docstring
is the porting contract); recipes in ``recipes``, samplers in ``samplers``, the training loop in
``train`` (``python -m wvad.baselines.train``). Method modules (mil, rtfm, urdmu, vadclip_v;
replacements mgfn, clip_tsa) are imported lazily by ``get_method`` and never here.
``official`` (equivalence harnesses only) is never imported by the training or scoring code.
"""

from wvad.baselines.common import (
    KNOWN_METHODS,
    MAIN_METHODS,
    REPLACEMENT_METHODS,
    BaselineMethod,
    Batch,
    LossOutput,
    MethodError,
    available_methods,
    get_method,
)

__all__ = ["KNOWN_METHODS", "MAIN_METHODS", "REPLACEMENT_METHODS", "BaselineMethod", "Batch",
           "LossOutput", "MethodError", "available_methods", "get_method"]
