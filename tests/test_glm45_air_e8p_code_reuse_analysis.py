from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8p_packed_abs_grid


def _load_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_e8p_code_reuse.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_glm45_air_e8p_code_reuse", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_e8p_projection_artifact(
    root: Path,
    *,
    layer_index: int = 1,
    projection: str = "gate_proj",
    codes: np.ndarray,
    group_size: int = 64,
) -> Path:
    prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
    experts, output_dims, codewords = codes.shape
    input_dims = codewords * 8
    scales = np.ones((experts, output_dims, input_dims // group_size), dtype=np.float16)
    shard = root / f"layer-{layer_index:05d}-{projection}.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            f"{prefix}.codes": mx.array(codes),
            f"{prefix}.scales": mx.array(scales),
            "model.vq_codebook.e8": mx.array(e8p_packed_abs_grid()),
        },
    )
    return shard


def test_analyze_packed_e8p_code_reuse_marks_unique_tiles_as_low_reuse() -> None:
    analyzer = _load_analyzer()
    codes = np.arange(512, dtype=np.uint16).reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="gate_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    assert row["projection"] == "gate_proj"
    assert row["total_codewords"] == 512
    assert row["unique_codes"] == 512
    assert row["global_reuse_factor"] == 1.0
    assert row["tile_summary"]["median_unique_ratio"] == 1.0
    assert row["recommendation"] == "low_code_reuse_do_not_prioritize_codebook_cache"


def test_analyze_packed_e8p_code_reuse_reports_split_byte_factor_locality() -> None:
    analyzer = _load_analyzer()
    signs = (np.arange(512, dtype=np.uint16) % 256).reshape(1, 64, 8)
    abs_indices = (np.arange(512, dtype=np.uint16) % 4).reshape(1, 64, 8)
    codes = signs | (abs_indices << np.uint16(8))
    scales = np.ones((1, 64, 1), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="gate_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    factors = row["byte_factor_summary"]
    assert factors["sign_byte"]["global_unique_values"] == 256
    assert factors["sign_byte"]["tile_summary"]["median_unique_values"] == 256.0
    assert factors["sign_byte"]["tile_summary"]["median_unique_ratio"] == 0.5
    assert factors["abs_index"]["global_unique_values"] == 4
    assert factors["abs_index"]["tile_summary"]["median_unique_values"] == 4.0
    assert factors["abs_index"]["tile_summary"]["median_unique_ratio"] == 0.007812
    assert factors["parity"]["global_unique_values"] == 2
    assert row["factor_recommendation"] == "split_byte_layout_worth_microbench"


def test_analyze_packed_e8p_code_reuse_reports_sign_nibble_factor_locality() -> None:
    analyzer = _load_analyzer()
    signs = (np.arange(512, dtype=np.uint16) % 256).reshape(1, 64, 8)
    abs_indices = (np.arange(512, dtype=np.uint16) % 151).reshape(1, 64, 8)
    codes = signs | (abs_indices << np.uint16(8))
    scales = np.ones((1, 64, 1), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="gate_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    factors = row["sign_nibble_factor_summary"]
    assert factors["sign_low_nibble"]["global_unique_values"] == 16
    assert factors["sign_low_nibble"]["tile_summary"]["median_unique_values"] == 16.0
    assert factors["sign_low_nibble"]["tile_summary"]["median_unique_ratio"] == 0.03125
    assert factors["sign_high_nibble"]["global_unique_values"] == 16
    assert factors["sign_high_nibble"]["tile_summary"]["median_unique_values"] == 16.0
    assert factors["sign_high_nibble"]["tile_summary"]["median_unique_ratio"] == 0.03125
    assert factors["abs_index"]["tile_summary"]["median_unique_values"] == 151.0
    assert row["sign_nibble_recommendation"] == "sign_nibble_abs_index_layout_worth_microbench"


def test_analyze_packed_e8p_code_reuse_reports_codeword_position_locality() -> None:
    analyzer = _load_analyzer()
    rows = np.arange(64, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    signs = (rows + words * np.uint16(17)) & np.uint16(0xFF)
    abs_indices = ((rows % np.uint16(4)) + words * np.uint16(16)) & np.uint16(0xFF)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="gate_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    factors = row["codeword_position_factor_summary"]
    abs_summary = factors["abs_index"]["position_summary"]
    assert abs_summary["position_count"] == 8
    assert abs_summary["median_unique_values_per_position"] == 4.0
    assert abs_summary["median_unique_ratio_per_position"] == 0.0625
    assert abs_summary["median_reuse_factor_per_position"] == 16.0
    assert factors["sign_low_nibble"]["position_summary"][
        "median_unique_ratio_per_position"
    ] == 0.25
    assert row["codeword_position_recommendation"] == (
        "codeword_position_sign_nibble_abs_index_layout_worth_microbench"
    )


def test_analyze_packed_e8p_code_reuse_reports_expert_kblock_locality() -> None:
    analyzer = _load_analyzer()
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    signs = (rows % np.uint16(16)) + (words * np.uint16(0))
    abs_indices = (rows % np.uint16(4)) + (words // np.uint16(8)) * np.uint16(32)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 128, 16)
    scales = np.ones((1, 128, 2), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="gate_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    factors = row["expert_kblock_factor_summary"]
    abs_summary = factors["abs_index"]["expert_kblock_summary"]
    assert abs_summary["expert_kblock_count"] == 2
    assert abs_summary["median_unique_values_per_expert_kblock"] == 4.0
    assert abs_summary["median_unique_ratio_per_expert_kblock"] == 0.003906
    assert abs_summary["median_reuse_factor_per_expert_kblock"] == 256.0
    assert factors["sign_byte"]["expert_kblock_summary"][
        "median_unique_ratio_per_expert_kblock"
    ] == 0.015625
    assert row["expert_kblock_recommendation"] == (
        "expert_kblock_factor_reuse_worth_kernel_family_probe"
    )


def test_analyze_packed_e8p_code_reuse_marks_repeated_tiles_as_promising() -> None:
    analyzer = _load_analyzer()
    repeated = np.tile(np.array([3, 7, 11, 19, 3, 7, 11, 19], dtype=np.uint16), (64, 1))
    codes = repeated.reshape(1, 64, 8)
    scales = np.ones((1, 64, 1), dtype=np.float16)

    row = analyzer.analyze_projection_code_reuse(
        projection="up_proj",
        codes=codes,
        scales=scales,
        group_size=64,
    )

    assert row["total_codewords"] == 512
    assert row["unique_codes"] == 4
    assert row["global_reuse_factor"] == 128.0
    assert row["tile_summary"]["median_unique_codes"] == 4.0
    assert row["recommendation"] == "code_reuse_promising_for_codebook_cache_probe"


def test_analyze_artifact_code_reuse_loads_selected_layer_projection(tmp_path) -> None:
    analyzer = _load_analyzer()
    codes = np.arange(512, dtype=np.uint16).reshape(1, 64, 8)
    _write_e8p_projection_artifact(
        tmp_path,
        layer_index=1,
        projection="down_proj",
        codes=codes,
    )

    summary = analyzer.analyze_artifact_code_reuse(
        artifact_dir=tmp_path,
        layer_index=1,
        projections=["down_proj"],
    )

    assert summary["record_type"] == "glm45_air_e8p_code_reuse_analysis"
    assert summary["schema_version"] == 1
    assert summary["artifact_dir"] == str(tmp_path)
    assert summary["layer_index"] == 1
    assert summary["projection_count"] == 1
    assert summary["overall_recommendation"] == "low_code_reuse_do_not_prioritize_codebook_cache"
    assert summary["projections"][0]["projection"] == "down_proj"
    assert summary["projections"][0]["layout"]["bk"] == 64


def test_analyze_artifact_code_reuse_emits_sign_nibble_schedule_requirements(tmp_path) -> None:
    analyzer = _load_analyzer()
    signs = (np.arange(512, dtype=np.uint16) % 256).reshape(1, 64, 8)
    abs_indices = (np.arange(512, dtype=np.uint16) % 151).reshape(1, 64, 8)
    codes = signs | (abs_indices << np.uint16(8))
    _write_e8p_projection_artifact(
        tmp_path,
        layer_index=1,
        projection="gate_proj",
        codes=codes,
    )

    summary = analyzer.analyze_artifact_code_reuse(
        artifact_dir=tmp_path,
        layer_index=1,
        projections=["gate_proj"],
    )

    requirements = summary["sign_nibble_schedule_requirements"]
    assert requirements["decision"] == "microbench_sign_nibble_abs_index_layout"
    assert requirements["factor_signal"] == "sign_nibble_abs_index_reuse_present"
    assert requirements["best_sign_nibble_reuse"]["sign_low_nibble"]["median_unique_ratio"] == 0.03125
    assert requirements["best_sign_nibble_reuse"]["sign_high_nibble"]["median_unique_ratio"] == 0.03125
    assert "split_sign_byte_into_low_high_nibble_masks" in requirements["required_layout_features"]
    assert "avoid_lane_local_barrier_fragment_buffer" in requirements["required_kernel_features"]


def test_analyze_artifact_code_reuse_emits_codeword_position_schedule_requirements(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    rows = np.arange(64, dtype=np.uint16)[:, None]
    words = np.arange(8, dtype=np.uint16)[None, :]
    signs = (rows + words * np.uint16(17)) & np.uint16(0xFF)
    abs_indices = ((rows % np.uint16(4)) + words * np.uint16(16)) & np.uint16(0xFF)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 64, 8)
    _write_e8p_projection_artifact(
        tmp_path,
        layer_index=1,
        projection="gate_proj",
        codes=codes,
    )

    summary = analyzer.analyze_artifact_code_reuse(
        artifact_dir=tmp_path,
        layer_index=1,
        projections=["gate_proj"],
    )

    requirements = summary["codeword_position_schedule_requirements"]
    assert requirements["decision"] == "microbench_codeword_position_factor_layout"
    assert requirements["factor_signal"] == "codeword_position_factor_reuse_present"
    assert requirements["best_codeword_position_reuse"]["abs_index"][
        "median_unique_ratio_per_position"
    ] == 0.0625
    assert "group_rhs_by_codeword_slot_inside_bk64" in requirements["required_layout_features"]
    assert "decode_factor_rows_at_codeword_position_scope" in requirements["required_kernel_features"]


def test_analyze_artifact_code_reuse_emits_expert_kblock_schedule_requirements(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    rows = np.arange(128, dtype=np.uint16)[:, None]
    words = np.arange(16, dtype=np.uint16)[None, :]
    signs = (rows % np.uint16(16)) + (words * np.uint16(0))
    abs_indices = (rows % np.uint16(4)) + (words // np.uint16(8)) * np.uint16(32)
    codes = (signs | (abs_indices << np.uint16(8))).reshape(1, 128, 16)
    _write_e8p_projection_artifact(
        tmp_path,
        layer_index=1,
        projection="gate_proj",
        codes=codes,
        group_size=64,
    )

    summary = analyzer.analyze_artifact_code_reuse(
        artifact_dir=tmp_path,
        layer_index=1,
        projections=["gate_proj"],
    )

    requirements = summary["expert_kblock_schedule_requirements"]
    assert requirements["decision"] == "probe_expert_kblock_factor_decode_reuse"
    assert requirements["factor_signal"] == "expert_kblock_factor_reuse_present"
    assert requirements["best_expert_kblock_reuse"]["abs_index"][
        "median_unique_ratio_per_expert_kblock"
    ] == 0.003906
    assert "reuse_factor_decode_across_output_tiles_per_expert_kblock" in requirements[
        "required_kernel_features"
    ]
    assert "fixed_codeword_position_output_column_layout" in requirements["rejected_next_steps"]


def test_split_byte_schedule_requirements_reject_staged_b_after_benchmark() -> None:
    analyzer = _load_analyzer()
    reuse_summary = {
        "projections": [
            {
                "projection": "gate_proj",
                "factor_recommendation": "split_byte_layout_worth_microbench",
                "byte_factor_summary": {
                    "sign_byte": {
                        "tile_summary": {
                            "median_unique_ratio": 0.390625,
                            "median_reuse_factor": 2.56,
                        },
                    },
                    "abs_index": {
                        "tile_summary": {
                            "median_unique_ratio": 0.296875,
                            "median_reuse_factor": 3.368421,
                        },
                    },
                    "parity": {
                        "tile_summary": {
                            "median_unique_ratio": 0.003906,
                            "median_reuse_factor": 256.0,
                        },
                    },
                },
            }
        ]
    }
    benchmark_analysis = {
        "all_parity_pass": False,
        "all_lane_s_pass": False,
        "comparisons": [
            {
                "candidate_variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 6.293916667345911,
                "ratio_to_q2": 5.623682477617159,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_fp16_sorted_steel_raw",
                "candidate_ms_per_iter": 7.117860999035959,
                "ratio_to_q2": 6.359885631481166,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 7.738708334121232,
                "ratio_to_q2": 6.91461942668832,
                "tokens": 128,
                "projection": "gate_up",
            },
        ],
    }

    requirements = analyzer._split_byte_schedule_requirements(
        reuse_summary=reuse_summary,
        benchmark_analysis=benchmark_analysis,
    )

    assert requirements["decision"] == "reject_staged_b_split_byte_tiled"
    assert requirements["observed_split_byte_tiled"]["ratio_to_steel"] == 1.087224
    assert requirements["observed_split_byte_tiled"]["ratio_to_packed_tiled"] == 1.229554
    assert "abs_index_row_reuse_outside_staged_b" in requirements["required_schedule_features"]
    assert "sign_mask_reuse_outside_staged_b" in requirements["required_schedule_features"]
    assert "same_staged_b_split_byte_tiled" in requirements["rejected_next_steps"]


def test_split_byte_schedule_requirements_reject_scalar_and_staged_factor_reuse() -> None:
    analyzer = _load_analyzer()
    reuse_summary = {
        "projections": [
            {
                "projection": "gate_proj",
                "factor_recommendation": "split_byte_layout_worth_microbench",
                "byte_factor_summary": {
                    "sign_byte": {
                        "tile_summary": {
                            "median_unique_ratio": 0.390625,
                            "median_reuse_factor": 2.56,
                        },
                    },
                    "abs_index": {
                        "tile_summary": {
                            "median_unique_ratio": 0.296875,
                            "median_reuse_factor": 3.368421,
                        },
                    },
                    "parity": {
                        "tile_summary": {
                            "median_unique_ratio": 0.003906,
                            "median_reuse_factor": 256.0,
                        },
                    },
                },
            }
        ]
    }
    benchmark_analysis = {
        "all_parity_pass": False,
        "all_lane_s_pass": False,
        "comparisons": [
            {
                "candidate_variant": "nax_e8p_fp16_sorted_steel_raw",
                "candidate_ms_per_iter": 5.791319330455735,
                "ratio_to_q2": 4.33288370289177,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
                "candidate_ms_per_iter": 6.974763673497364,
                "ratio_to_q2": 5.2182996875152625,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 7.707861329739292,
                "ratio_to_q2": 5.766780388735522,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 9.357902657939121,
                "ratio_to_q2": 7.001289620933623,
                "tokens": 128,
                "projection": "gate_up",
            },
        ],
    }

    requirements = analyzer._split_byte_schedule_requirements(
        reuse_summary=reuse_summary,
        benchmark_analysis=benchmark_analysis,
    )

    assert requirements["decision"] == "reject_scalar_and_staged_factor_reuse"
    assert requirements["observed_factor_reuse_native"]["ratio_to_steel"] == 1.204348
    assert requirements["observed_factor_reuse_native"]["ratio_to_q2"] == 5.2182996875152625
    assert requirements["observed_factor_reuse_tiled"]["ratio_to_steel"] == 1.61585
    assert "non_staged_tensorops_shared_decode" in requirements["required_schedule_features"]
    assert "no_scalar_direct_route_output_traversal" in requirements["required_schedule_features"]
    assert "scalar_direct_factor_reuse_traversal" in requirements["rejected_next_steps"]
    assert "same_staged_b_factor_reuse_tiled" in requirements["rejected_next_steps"]


def test_split_byte_schedule_requirements_reject_current_shared_decode_factor_reuse() -> None:
    analyzer = _load_analyzer()
    reuse_summary = {
        "projections": [
            {
                "projection": "gate_proj",
                "factor_recommendation": "split_byte_layout_worth_microbench",
                "byte_factor_summary": {
                    "sign_byte": {
                        "tile_summary": {
                            "median_unique_ratio": 0.390625,
                            "median_reuse_factor": 2.56,
                        },
                    },
                    "abs_index": {
                        "tile_summary": {
                            "median_unique_ratio": 0.296875,
                            "median_reuse_factor": 3.368421,
                        },
                    },
                    "parity": {
                        "tile_summary": {
                            "median_unique_ratio": 0.003906,
                            "median_reuse_factor": 256.0,
                        },
                    },
                },
            }
        ]
    }
    benchmark_analysis = {
        "all_parity_pass": False,
        "all_lane_s_pass": False,
        "comparisons": [
            {
                "candidate_variant": "nax_e8p_fp16_sorted_steel_raw",
                "candidate_ms_per_iter": 5.852221996368219,
                "ratio_to_q2": 3.3274105695161307,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
                "candidate_ms_per_iter": 4.881333336622144,
                "ratio_to_q2": 2.7753902958034695,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 5.827138995906959,
                "ratio_to_q2": 3.313149073472103,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
                "candidate_ms_per_iter": 18.064208338425185,
                "ratio_to_q2": 10.270806164311384,
                "tokens": 128,
                "projection": "gate_up",
            },
        ],
    }

    requirements = analyzer._split_byte_schedule_requirements(
        reuse_summary=reuse_summary,
        benchmark_analysis=benchmark_analysis,
    )

    assert requirements["decision"] == "reject_current_shared_decode_factor_reuse"
    assert requirements["observed_factor_reuse_shared_decode"]["ratio_to_steel"] == 3.086726
    assert requirements["observed_factor_reuse_shared_decode"]["ratio_to_factor_reuse_native"] == 3.700671
    assert requirements["observed_factor_reuse_shared_decode"]["ratio_to_factor_reuse_tiled"] == 3.100013
    assert "current_non_staged_shared_decode_factor_reuse" in requirements["rejected_next_steps"]
    assert "broader_expert_output_factor_decode_reuse" in requirements["required_schedule_features"]
    assert "change_rhs_layout_or_kernel_family" in requirements["required_schedule_features"]


def test_split_byte_schedule_requirements_reject_shared_n_factor_reuse() -> None:
    analyzer = _load_analyzer()
    reuse_summary = {
        "projections": [
            {
                "projection": "gate_proj",
                "factor_recommendation": "split_byte_layout_worth_microbench",
                "byte_factor_summary": {
                    "sign_byte": {
                        "tile_summary": {
                            "median_unique_ratio": 0.390625,
                            "median_reuse_factor": 2.56,
                        },
                    },
                    "abs_index": {
                        "tile_summary": {
                            "median_unique_ratio": 0.296875,
                            "median_reuse_factor": 3.368421,
                        },
                    },
                    "parity": {
                        "tile_summary": {
                            "median_unique_ratio": 0.003906,
                            "median_reuse_factor": 256.0,
                        },
                    },
                },
            }
        ]
    }
    benchmark_analysis = {
        "all_parity_pass": False,
        "all_lane_s_pass": False,
        "comparisons": [
            {
                "candidate_variant": "nax_e8p_fp16_sorted_steel_raw",
                "candidate_ms_per_iter": 6.017721995400886,
                "ratio_to_q2": 5.660705123487832,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
                "candidate_ms_per_iter": 5.378055667582278,
                "ratio_to_q2": 5.058988649717058,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
                "candidate_ms_per_iter": 6.155347335152328,
                "ratio_to_q2": 5.790165485140986,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
                "candidate_ms_per_iter": 18.31377800165986,
                "ratio_to_q2": 17.22726590620917,
                "tokens": 128,
                "projection": "gate_up",
            },
            {
                "candidate_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
                "candidate_ms_per_iter": 12.246152997249737,
                "ratio_to_q2": 11.51961839837857,
                "tokens": 128,
                "projection": "gate_up",
            },
        ],
    }

    requirements = analyzer._split_byte_schedule_requirements(
        reuse_summary=reuse_summary,
        benchmark_analysis=benchmark_analysis,
    )

    assert requirements["decision"] == "reject_shared_n_factor_reuse"
    assert requirements["observed_factor_reuse_shared_n"]["ratio_to_shared_decode"] == 0.668685
    assert requirements["observed_factor_reuse_shared_n"]["ratio_to_factor_reuse_native"] == 2.27706
    assert requirements["observed_factor_reuse_shared_n"]["ratio_to_factor_reuse_tiled"] == 1.989515
    assert "lane_local_shared_n_factor_reuse" in requirements["rejected_next_steps"]
    assert "avoid_lane_local_barrier_fragment_buffer" in requirements["required_schedule_features"]
    assert "change_rhs_layout_or_kernel_family" in requirements["required_schedule_features"]
