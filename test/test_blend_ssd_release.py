from types import SimpleNamespace

import torch

from sglang.srt.utils.cache_blender_info import ContextBlendPool
from sglang.srt.utils.digest_index_manager import (
    DIGEST_INDEX_VERSION,
    DigestIndexManager,
)


def test_prepare_augmented_locs_for_request():
    raw_locs = [0, 2, 5, 7, 11, 13, 17]

    transformed = DigestIndexManager.prepare_augmented_locs_for_request(raw_locs)

    assert transformed == {
        "forward_locs": [0, 2, 7, 13],
        "original_locs": [0, 2, 5, 11],
        "keep_indices": [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16],
        "aug_sys_range": (0, 2),
        "aug_doc_ranges": [(2, 5), (7, 11)],
        "aug_zip_ranges": [(5, 7), (11, 13)],
    }


def test_build_all_indices_sets_non_query_context_ranges():
    DigestIndexManager.clear()
    ContextBlendPool.clear()
    blend_info = SimpleNamespace(
        att_params=SimpleNamespace(num_layers=2),
        qcompute_end=2,
        digest_ratio=0.5,
        digest_index_method="kvzip",
        critical_layers=[1],
        digest_original_chunk_loc_list=torch.tensor([0, 2, 6, 9]),
        context_n_sink=1,
    )

    DigestIndexManager.build_all_indices(blend_info, rotary_emb=None)

    metadata, indices_by_method = DigestIndexManager.export_payload()
    assert metadata["digest_index_version"] == DIGEST_INDEX_VERSION
    assert metadata["orig_chunk_ranges"] == [[0, 2], [2, 6], [6, 9]]
    assert metadata["total_tokens"] == 6
    assert "kvzip" in indices_by_method
    assert ContextBlendPool.orig_chunk_ranges == [(0, 2), (2, 6)]
