"""Training losses."""

from wvad.losses.mil import (
    dense_ranking_loss,
    max_ranking_loss,
    normal_segment_bce,
    pseudo_label_bce,
    smoothness_loss,
    sparsity_loss,
    topk_video_bce,
    video_bce,
)

__all__ = ["dense_ranking_loss", "max_ranking_loss", "normal_segment_bce", "pseudo_label_bce",
           "smoothness_loss", "sparsity_loss", "topk_video_bce", "video_bce"]
