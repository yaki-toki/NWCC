"""Training losses."""

from wvad.losses.mil import (
    dense_ranking_loss,
    max_ranking_loss,
    pseudo_label_bce,
    smoothness_loss,
    sparsity_loss,
    video_bce,
)

__all__ = ["dense_ranking_loss", "max_ranking_loss", "pseudo_label_bce", "smoothness_loss",
           "sparsity_loss", "video_bce"]
