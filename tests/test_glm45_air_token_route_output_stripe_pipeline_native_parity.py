from __future__ import annotations

import pytest

from mlx_vq.kernels import nax


def test_token_route_output_stripe_pipeline_native_parity_report_passes() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    from benchmarks.prove_glm45_air_e8p_token_route_output_stripe_pipeline_native_parity import (
        build_report,
    )

    report = build_report(
        experts=3,
        tokens=13,
        routes=41,
        output_dims=70,
        input_dims=704,
        group_size=352,
        bn=64,
        bk=64,
        max_routes_per_token=8,
        seed=20261007,
        min_cosine=0.999999,
        max_abs_diff=6e-3,
    )

    assert (
        report["record_type"]
        == "glm45_air_e8p_token_route_output_stripe_pipeline_native_parity"
    )
    assert report["decision"] == "token_route_output_stripe_pipeline_native_parity_pass"
    assert report["passes_native_parity"] is True
    assert report["native_parity_claim"] is True
    assert report["speed_claim"] is False
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["shape"]["group_size"] == 352
    assert report["shape"]["max_routes_per_token"] == 8
    assert report["shape"]["max_routes_in_token"] <= 8
    assert report["contract"]["storage_constraint"] == (
        "compressed_e8p_token_route_output_stripe_pipelines"
    )
    assert report["contract"]["dispatch_grid"] == (
        "tokens_x_route_slots_x_output_stripes_x_kblock_stages"
    )
    assert report["contract"]["preserves_token_axis"] is True
    assert report["contract"]["preserves_route_slot_axis"] is True
    assert report["contract"]["preserves_output_stripe_axis"] is True
    assert report["contract"]["preserves_kblock_stage_axis"] is True
    assert report["contract"]["streams_compressed_codeword_tiles"] is True
    assert report["contract"][
        "streams_kblock_stages_inside_token_route_output_stripes"
    ] is True
    assert report["contract"]["accumulates_full_output_without_route_expansion"] is True
    assert report["contract"]["expands_route_microtiles"] is False
    assert report["contract"]["materializes_output_tile_local_full_lut"] is False
    assert report["contract"]["materializes_decoded_dense_rhs"] is False
    assert report["cosine"] >= 0.999999
    assert report["max_abs_diff"] <= 6e-3
