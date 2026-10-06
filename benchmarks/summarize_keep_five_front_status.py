from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_WORKSTREAM1 = Path(
    "artifacts/quality/glm45-air-workstream1-free-levers-fullsplit-verdict-20260703.json"
)
DEFAULT_WORKSTREAM1_FIXED_BUMP = Path(
    "artifacts/quality/glm45-air-workstream1-bump-fixed-loader-verdict-20260704.json"
)
DEFAULT_TRACKB = Path(
    "artifacts/quality/glm45-air-e8p-trackb-frontier-summary-20260703.json"
)
DEFAULT_TRACKB_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-trackb-route-slot-codeword-stream-design-20260703.json"
)
DEFAULT_TRACKB_SOURCE_GUARD = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-source-guardrail-20260703.json"
)
DEFAULT_TRACKB_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-codeword-stream-native-parity-20260703.json"
)
DEFAULT_TRACKB_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-codeword-stream-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-codeword-stream-speed-packet-20260703.json"
)
DEFAULT_TRACKB_SUCCESSOR_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-mma-codeword-tile-design-20260703.json"
)
DEFAULT_TRACKB_SUCCESSOR_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-mma-codeword-tile-native-parity-20260703.json"
)
DEFAULT_TRACKB_SUCCESSOR_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-mma-codeword-tile-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_SUCCESSOR_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-slot-mma-codeword-tile-speed-packet-20260703.json"
)
DEFAULT_TRACKB_NEXT_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-active-route-tile-codeword-outer-product-design-20260703.json"
)
DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-active-route-tile-codeword-outer-product-native-parity-20260703.json"
)
DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-active-route-tile-codeword-outer-product-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-active-route-tile-codeword-outer-product-speed-packet-20260703.json"
)
DEFAULT_TRACKB_EXPERT_COHORT_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-expert-cohort-codeword-broadcast-design-20260703.json"
)
DEFAULT_TRACKB_EXPERT_COHORT_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-cohort-codeword-broadcast-native-parity-20260703.json"
)
DEFAULT_TRACKB_EXPERT_COHORT_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-cohort-codeword-broadcast-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_EXPERT_COHORT_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-expert-cohort-codeword-broadcast-speed-packet-20260703.json"
)
DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-batch-segmented-codeword-reduce-design-20260703.json"
)
DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-batch-segmented-codeword-reduce-native-parity-20260703.json"
)
DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-batch-segmented-codeword-reduce-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-batch-segmented-codeword-reduce-speed-packet-20260703.json"
)
DEFAULT_TRACKB_COMPONENT_STREAM_PARTIAL_REDUCTION_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-component-stream-partial-reduction-design-20260703.json"
)
DEFAULT_TRACKB_COMPONENT_STREAM_PARTIAL_REDUCTION_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-component-stream-partial-reduction-native-parity-20260703.json"
)
DEFAULT_TRACKB_COMPONENT_STREAM_TENSOROPS_REJECTION = Path(
    "artifacts/quality/glm45-air-e8p-component-stream-tensorops-m1024-m4096-structure-rejection-20260703.json"
)
DEFAULT_TRACKB_NEXT_FAMILY_GATE = Path(
    "artifacts/quality/glm45-air-e8p-trackb-next-family-gate-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-codeword-stream-design-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-codeword-stream-native-parity-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-codeword-stream-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-codeword-stream-speed-packet-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-design-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-native-parity-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-speed-packet-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-design-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-native-parity-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-speed-packet-20260703.json"
)
DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-design-20260703.json"
)
DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-native-parity-20260703.json"
)
DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-speed-packet-20260703.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-design-20260703.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-native-parity-20260703.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-speed-packet-20260703.json"
)
DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-codeword-lut-accumulate-design-20260703.json"
)
DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-codeword-lut-accumulate-native-parity-20260703.json"
)
DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-codeword-lut-accumulate-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-codeword-lut-accumulate-speed-packet-20260703.json"
)
DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-rowwise-codeword-tile-accumulate-design-20260703.json"
)
DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-rowwise-codeword-tile-accumulate-native-parity-20260703.json"
)
DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-rowwise-codeword-tile-accumulate-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-rowwise-codeword-tile-accumulate-speed-packet-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-output-tile-local-codeword-lut-design-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-tile-local-codeword-lut-native-parity-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-tile-local-codeword-lut-artifact-parity-20260703.json"
)
DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-output-tile-local-codeword-lut-speed-packet-20260703.json"
)
DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-microtile-codeword-block-reduce-design-20260703.json"
)
DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-microtile-codeword-block-reduce-native-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-microtile-codeword-block-reduce-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-microtile-codeword-block-reduce-speed-packet-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-kblock-wavefront-codeword-scan-design-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-kblock-wavefront-codeword-scan-native-parity-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-kblock-wavefront-codeword-scan-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-kblock-wavefront-codeword-scan-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-route-output-stripe-pipeline-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-route-output-stripe-pipeline-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-route-output-stripe-pipeline-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-route-output-stripe-pipeline-speed-packet-20260704.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-scale-slot-stream-design-20260704.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-scale-slot-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-scale-slot-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-expert-kblock-scale-slot-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-scale-group-route-block-reduce-design-20260704.json"
)
DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-scale-group-route-block-reduce-native-parity-20260704.json"
)
DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-scale-group-route-block-reduce-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-scale-group-route-block-reduce-speed-packet-20260704.json"
)
DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-block-output-group-stream-design-20260704.json"
)
DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-block-output-group-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-block-output-group-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-block-output-group-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-output-group-pretransposed-codeword-stream-design-20260704.json"
)
DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-group-pretransposed-codeword-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-output-group-pretransposed-codeword-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-output-group-pretransposed-codeword-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-kblock-output-group-route-fused-stream-design-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-kblock-output-group-route-fused-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-kblock-output-group-route-fused-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-kblock-output-group-route-fused-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-route-tile-output-swizzle-stream-design-20260704.json"
)
DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-tile-output-swizzle-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-route-tile-output-swizzle-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-route-tile-output-swizzle-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-topk-output-tile-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-topk-output-tile-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-topk-output-tile-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-topk-output-tile-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-block-output-group-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-block-output-group-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-block-output-group-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-block-output-group-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-output-stripe-group-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-output-stripe-group-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-output-stripe-group-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-output-stripe-group-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-expert-output-block-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-expert-output-block-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-expert-output-block-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-expert-output-block-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-kblock-accumulator-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-kblock-accumulator-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-kblock-accumulator-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-kblock-accumulator-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-output-group-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-output-group-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-output-group-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-output-group-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-group-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-group-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-group-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-group-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-codeword-group-pipeline-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-scale-slot-broadcast-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-scale-slot-broadcast-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-scale-slot-broadcast-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-scale-slot-broadcast-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-route-bucket-codeword-reduce-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-kblock-microtile-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-kblock-microtile-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-kblock-microtile-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-kblock-microtile-stream-speed-packet-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_DESIGN = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-tile-fused-stream-design-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_NATIVE_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-tile-fused-stream-native-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_ARTIFACT_PARITY = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-tile-fused-stream-artifact-parity-20260704.json"
)
DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_SPEED_PACKET = Path(
    "artifacts/quality/glm45-air-e8p-token-pair-slot-topk-output-tile-fused-stream-speed-packet-20260704.json"
)
DEFAULT_QWEN_GATE = Path(
    "artifacts/quality/qwen36-35b-a3b-family-gate-hardened-20260704.json"
)
DEFAULT_GLM52_LAYER3 = Path(
    "artifacts/quality/glm52-vq-layer3-source-artifact-diagnostic-20260703.json"
)
DEFAULT_GLM52_LAYER77 = Path(
    "artifacts/quality/glm52-vq-layer77-source-artifact-diagnostic-20260703.json"
)
DEFAULT_CACHE_SOURCE_SCAN = Path(
    "artifacts/quality/glm45-air-single-host-teacher-source-candidate-scan-20260704.json"
)
DEFAULT_SINGLE_HOST_CACHE_ATTEMPT = Path(
    "artifacts/quality/glm45-air-cleanroom-single-host-local-sequential-prefix-split23-lower2-4-6-8-10-12-14-16-18-20-22-upper25-27-29-31-33-35-37-39-41-43-45-stageviews-headproc-systemlimits-20260704.json"
)
DEFAULT_LAYER_SPLIT_RECOMMENDATION = Path(
    "artifacts/quality/glm45-air-rank0-logits-layer-split-recommendation-20260704.json"
)
DEFAULT_RDMA_TOPOLOGY_AUDIT = Path(
    "artifacts/quality/glm45-air-rdma-topology-audit-20260704-goal-continuation5-readonly.json"
)
DEFAULT_MODEL_CARD = Path("docs/research/GLM45_AIR_KEEP_MODEL_CARD_DRAFT.md")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _transport_flags(*payloads: dict[str, Any]) -> dict[str, bool]:
    return {
        "peer2_used": any(bool(payload.get("peer2_used")) for payload in payloads),
        "rdma_jaccl_touched": any(
            bool(payload.get("rdma_jaccl_touched")) for payload in payloads
        ),
    }


def _rdma_topology_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    local = payload.get("local") if isinstance(payload.get("local"), dict) else {}
    peer = payload.get("peer") if isinstance(payload.get("peer"), dict) else {}
    local_interface = (
        local.get("interface") if isinstance(local.get("interface"), dict) else {}
    )
    peer_interface = (
        peer.get("interface") if isinstance(peer.get("interface"), dict) else {}
    )
    local_route = local.get("route") if isinstance(local.get("route"), dict) else {}
    peer_route = peer.get("route") if isinstance(peer.get("route"), dict) else {}
    local_rdma = local.get("rdma") if isinstance(local.get("rdma"), dict) else {}
    peer_rdma = peer.get("rdma") if isinstance(peer.get("rdma"), dict) else {}
    return {
        "decision": payload.get("decision"),
        "direct_path_ready": bool(payload.get("direct_path_ready")),
        "link_layer_ready": bool(payload.get("link_layer_ready")),
        "direct_ip_ready": bool(payload.get("direct_ip_ready")),
        "direct_route_ready": bool(payload.get("direct_route_ready")),
        "direct_arp_ready": bool(payload.get("direct_arp_ready")),
        "rdma_ipv4_gid_ready": bool(payload.get("rdma_ipv4_gid_ready")),
        "ready_for_uc_pingpong": bool(payload.get("ready_for_uc_pingpong")),
        "ready_for_jaccl_preflight": bool(payload.get("ready_for_jaccl_preflight")),
        "local_interface_status": local_interface.get("status"),
        "peer_interface_status": peer_interface.get("status"),
        "local_route_interface": local_route.get("interface"),
        "peer_route_interface": peer_route.get("interface"),
        "local_rdma_port_active": bool(local_rdma.get("port_active")),
        "peer_rdma_port_active": bool(peer_rdma.get("port_active")),
    }


def _workstream1_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision")
    if decision == "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable":
        return "bump_lm_head_embed_fixed_loader_measured_not_promotable"
    if decision == "reject_free_p1_levers_metric_backed":
        return "metric_backed_free_levers_rejected"
    return "needs_attention"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


QWEN_PLACEHOLDER_BENCHMARK_BASELINES = {
    "",
    "qwen_source_or_qwen_control_runtime",
}


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _qwen_family_thresholds(qwen: dict[str, Any]) -> dict[str, Any] | None:
    thresholds = qwen.get("qwen_family_thresholds")
    return thresholds if isinstance(thresholds, dict) else None


def _qwen_eval_required_row_count(qwen: dict[str, Any]) -> int | None:
    thresholds = _qwen_family_thresholds(qwen)
    if thresholds is None:
        return None
    eval_gate = thresholds.get("eval_gate")
    if not isinstance(eval_gate, dict):
        return None

    minimum_total = _int_value(eval_gate.get("minimum_total_clean_rows"))
    if minimum_total is not None:
        return minimum_total

    minimum_per_split = _int_value(eval_gate.get("minimum_clean_rows_per_split"))
    required_splits = eval_gate.get("required_splits")
    if minimum_per_split is None or not isinstance(required_splits, list):
        return None
    return minimum_per_split * len(required_splits)


def _qwen_benchmark_reference(qwen: dict[str, Any]) -> str | None:
    thresholds = _qwen_family_thresholds(qwen)
    if thresholds is None:
        return None
    benchmark_gate = thresholds.get("benchmark_gate")
    if not isinstance(benchmark_gate, dict):
        return None
    reference = benchmark_gate.get("comparison_baseline")
    return reference.strip() if isinstance(reference, str) else None


def _qwen_has_same_machine_reference(qwen: dict[str, Any]) -> bool:
    thresholds = _qwen_family_thresholds(qwen)
    if thresholds is None:
        return False
    benchmark_gate = thresholds.get("benchmark_gate")
    if not isinstance(benchmark_gate, dict):
        return False
    reference = _qwen_benchmark_reference(qwen)
    if reference is None or reference in QWEN_PLACEHOLDER_BENCHMARK_BASELINES:
        return False
    normalized_reference = reference.replace("-", "_").lower()
    return bool(benchmark_gate.get("same_machine_reference")) or (
        "same_machine" in normalized_reference
    )


def _qwen_hardening_requirements(qwen: dict[str, Any]) -> list[str]:
    thresholds = _qwen_family_thresholds(qwen)
    if thresholds is None:
        return []

    requirements: list[str] = []
    required_rows = _qwen_eval_required_row_count(qwen)
    if required_rows is None or required_rows < 64:
        requirements.append("qwen_eval_prompt_pack_at_least_64_rows")
    if not _qwen_has_same_machine_reference(qwen):
        requirements.append("qwen_benchmark_same_machine_reference_ratio")
    return requirements


def build_status(
    *,
    workstream1_path: Path = DEFAULT_WORKSTREAM1,
    workstream1_fixed_bump_path: Path | None = DEFAULT_WORKSTREAM1_FIXED_BUMP,
    trackb_path: Path = DEFAULT_TRACKB,
    trackb_design_path: Path | None = DEFAULT_TRACKB_DESIGN,
    trackb_source_guard_path: Path | None = DEFAULT_TRACKB_SOURCE_GUARD,
    trackb_native_parity_path: Path | None = DEFAULT_TRACKB_NATIVE_PARITY,
    trackb_artifact_parity_path: Path | None = DEFAULT_TRACKB_ARTIFACT_PARITY,
    trackb_speed_packet_path: Path | None = DEFAULT_TRACKB_SPEED_PACKET,
    trackb_successor_design_path: Path | None = DEFAULT_TRACKB_SUCCESSOR_DESIGN,
    trackb_successor_native_parity_path: Path | None = (
        DEFAULT_TRACKB_SUCCESSOR_NATIVE_PARITY
    ),
    trackb_successor_artifact_parity_path: Path | None = (
        DEFAULT_TRACKB_SUCCESSOR_ARTIFACT_PARITY
    ),
    trackb_successor_speed_packet_path: Path | None = (
        DEFAULT_TRACKB_SUCCESSOR_SPEED_PACKET
    ),
    trackb_next_design_path: Path | None = None,
    trackb_active_route_tile_native_parity_path: Path | None = None,
    trackb_active_route_tile_artifact_parity_path: Path | None = None,
    trackb_active_route_tile_speed_packet_path: Path | None = None,
    trackb_expert_cohort_design_path: Path | None = None,
    trackb_expert_cohort_native_parity_path: Path | None = None,
    trackb_expert_cohort_artifact_parity_path: Path | None = None,
    trackb_expert_cohort_speed_packet_path: Path | None = None,
    trackb_route_batch_segmented_design_path: Path | None = None,
    trackb_route_batch_segmented_native_parity_path: Path | None = None,
    trackb_route_batch_segmented_artifact_parity_path: Path | None = None,
    trackb_route_batch_segmented_speed_packet_path: Path | None = None,
    trackb_component_stream_partial_reduction_design_path: Path | None = None,
    trackb_component_stream_partial_reduction_native_parity_path: Path | None = None,
    trackb_component_stream_tensorops_rejection_path: Path | None = None,
    trackb_next_family_gate_path: Path | None = None,
    trackb_token_cohort_codeword_stream_design_path: Path | None = None,
    trackb_token_cohort_codeword_stream_native_parity_path: Path | None = None,
    trackb_token_cohort_codeword_stream_artifact_parity_path: Path | None = None,
    trackb_token_cohort_codeword_stream_speed_packet_path: Path | None = None,
    trackb_token_cohort_mma_codeword_tile_design_path: Path | None = None,
    trackb_token_cohort_mma_codeword_tile_native_parity_path: Path | None = None,
    trackb_token_cohort_mma_codeword_tile_artifact_parity_path: Path | None = None,
    trackb_token_cohort_mma_codeword_tile_speed_packet_path: Path | None = None,
    trackb_output_stationary_codeword_tile_design_path: Path | None = None,
    trackb_output_stationary_codeword_tile_native_parity_path: Path | None = None,
    trackb_output_stationary_codeword_tile_artifact_parity_path: Path | None = None,
    trackb_output_stationary_codeword_tile_speed_packet_path: Path | None = None,
    trackb_input_stationary_codeword_tile_design_path: Path | None = None,
    trackb_input_stationary_codeword_tile_native_parity_path: Path | None = None,
    trackb_input_stationary_codeword_tile_artifact_parity_path: Path | None = None,
    trackb_input_stationary_codeword_tile_speed_packet_path: Path | None = None,
    trackb_expert_kblock_codeword_factor_reuse_design_path: Path | None = None,
    trackb_expert_kblock_codeword_factor_reuse_native_parity_path: Path | None = None,
    trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path: Path
    | None = None,
    trackb_expert_kblock_codeword_factor_reuse_speed_packet_path: Path
    | None = None,
    trackb_route_codeword_lut_accumulate_design_path: Path | None = None,
    trackb_route_codeword_lut_accumulate_native_parity_path: Path | None = None,
    trackb_route_codeword_lut_accumulate_artifact_parity_path: Path | None = None,
    trackb_route_codeword_lut_accumulate_speed_packet_path: Path | None = None,
    trackb_rowwise_codeword_tile_accumulate_design_path: Path | None = None,
    trackb_rowwise_codeword_tile_accumulate_native_parity_path: Path
    | None = None,
    trackb_rowwise_codeword_tile_accumulate_artifact_parity_path: Path
    | None = None,
    trackb_rowwise_codeword_tile_accumulate_speed_packet_path: Path | None = None,
    trackb_output_tile_local_codeword_lut_design_path: Path | None = None,
    trackb_output_tile_local_codeword_lut_native_parity_path: Path
    | None = None,
    trackb_output_tile_local_codeword_lut_artifact_parity_path: Path
    | None = None,
    trackb_output_tile_local_codeword_lut_speed_packet_path: Path | None = None,
    trackb_route_microtile_codeword_block_reduce_design_path: Path | None = None,
    trackb_route_microtile_codeword_block_reduce_native_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_NATIVE_PARITY,
    trackb_route_microtile_codeword_block_reduce_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_ARTIFACT_PARITY,
    trackb_route_microtile_codeword_block_reduce_speed_packet_path: Path
    | None = DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_SPEED_PACKET,
    trackb_kblock_wavefront_codeword_scan_design_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_DESIGN,
    trackb_kblock_wavefront_codeword_scan_native_parity_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_NATIVE_PARITY,
    trackb_kblock_wavefront_codeword_scan_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_ARTIFACT_PARITY,
    trackb_kblock_wavefront_codeword_scan_speed_packet_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_SPEED_PACKET,
    trackb_token_route_output_stripe_pipeline_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_DESIGN,
    trackb_token_route_output_stripe_pipeline_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_NATIVE_PARITY,
    trackb_token_route_output_stripe_pipeline_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_ARTIFACT_PARITY,
    trackb_token_route_output_stripe_pipeline_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_SPEED_PACKET,
    trackb_expert_kblock_scale_slot_stream_design_path: Path
    | None = DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_DESIGN,
    trackb_expert_kblock_scale_slot_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_NATIVE_PARITY,
    trackb_expert_kblock_scale_slot_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_ARTIFACT_PARITY,
    trackb_expert_kblock_scale_slot_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_SPEED_PACKET,
    trackb_scale_group_route_block_reduce_design_path: Path | None = None,
    trackb_scale_group_route_block_reduce_native_parity_path: Path
    | None = DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_NATIVE_PARITY,
    trackb_scale_group_route_block_reduce_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_ARTIFACT_PARITY,
    trackb_scale_group_route_block_reduce_speed_packet_path: Path
    | None = DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_SPEED_PACKET,
    trackb_route_block_output_group_stream_design_path: Path | None = None,
    trackb_route_block_output_group_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    trackb_route_block_output_group_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    trackb_route_block_output_group_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    trackb_output_group_pretransposed_codeword_stream_design_path: Path
    | None = None,
    trackb_output_group_pretransposed_codeword_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_NATIVE_PARITY,
    trackb_output_group_pretransposed_codeword_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_ARTIFACT_PARITY,
    trackb_output_group_pretransposed_codeword_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_SPEED_PACKET,
    trackb_kblock_output_group_route_fused_stream_design_path: Path | None = None,
    trackb_kblock_output_group_route_fused_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_NATIVE_PARITY,
    trackb_kblock_output_group_route_fused_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_ARTIFACT_PARITY,
    trackb_kblock_output_group_route_fused_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_SPEED_PACKET,
    trackb_route_tile_output_swizzle_stream_design_path: Path
    | None = DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_DESIGN,
    trackb_route_tile_output_swizzle_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_NATIVE_PARITY,
    trackb_route_tile_output_swizzle_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_ARTIFACT_PARITY,
    trackb_route_tile_output_swizzle_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_SPEED_PACKET,
    trackb_token_topk_output_tile_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_DESIGN,
    trackb_token_topk_output_tile_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_NATIVE_PARITY,
    trackb_token_topk_output_tile_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_ARTIFACT_PARITY,
    trackb_token_topk_output_tile_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_SPEED_PACKET,
    trackb_token_block_output_group_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_DESIGN,
    trackb_token_block_output_group_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    trackb_token_block_output_group_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    trackb_token_block_output_group_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    trackb_token_output_stripe_group_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_DESIGN,
    trackb_token_output_stripe_group_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_NATIVE_PARITY,
    trackb_token_output_stripe_group_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_ARTIFACT_PARITY,
    trackb_token_output_stripe_group_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_SPEED_PACKET,
    trackb_token_expert_output_block_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_DESIGN,
    trackb_token_expert_output_block_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_NATIVE_PARITY,
    trackb_token_expert_output_block_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_ARTIFACT_PARITY,
    trackb_token_expert_output_block_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_SPEED_PACKET,
    trackb_token_pair_kblock_accumulator_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_DESIGN,
    trackb_token_pair_kblock_accumulator_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_NATIVE_PARITY,
    trackb_token_pair_kblock_accumulator_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_kblock_accumulator_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_SPEED_PACKET,
    trackb_token_pair_output_group_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_DESIGN,
    trackb_token_pair_output_group_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    trackb_token_pair_output_group_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_output_group_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    trackb_token_pair_slot_topk_output_group_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_DESIGN,
    trackb_token_pair_slot_topk_output_group_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_output_group_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    trackb_token_pair_slot_topk_codeword_group_pipeline_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_DESIGN,
    trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_NATIVE_PARITY,
    trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_SPEED_PACKET,
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_DESIGN,
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_NATIVE_PARITY,
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_SPEED_PACKET,
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_DESIGN,
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_NATIVE_PARITY,
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_SPEED_PACKET,
    trackb_token_pair_slot_topk_kblock_microtile_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_DESIGN,
    trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_NATIVE_PARITY,
    trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_SPEED_PACKET,
    trackb_token_pair_slot_topk_output_tile_fused_stream_design_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_DESIGN,
    trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_NATIVE_PARITY,
    trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_ARTIFACT_PARITY,
    trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path: Path
    | None = DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_SPEED_PACKET,
    qwen_gate_path: Path = DEFAULT_QWEN_GATE,
    glm52_layer3_path: Path = DEFAULT_GLM52_LAYER3,
    glm52_layer77_path: Path = DEFAULT_GLM52_LAYER77,
    cache_source_scan_path: Path = DEFAULT_CACHE_SOURCE_SCAN,
    single_host_cache_attempt_path: Path | None = None,
    layer_split_recommendation_path: Path | None = None,
    rdma_topology_audit_path: Path | None = None,
    model_card_path: Path = DEFAULT_MODEL_CARD,
) -> dict[str, Any]:
    workstream1 = _read_json(workstream1_path)
    workstream1_fixed_bump = (
        _read_json(workstream1_fixed_bump_path)
        if workstream1_fixed_bump_path is not None
        and workstream1_fixed_bump_path.is_file()
        else None
    )
    trackb = _read_json(trackb_path)
    trackb_design = (
        _read_json(trackb_design_path)
        if trackb_design_path is not None and trackb_design_path.is_file()
        else None
    )
    trackb_source_guard = (
        _read_json(trackb_source_guard_path)
        if trackb_source_guard_path is not None and trackb_source_guard_path.is_file()
        else None
    )
    trackb_native_parity = (
        _read_json(trackb_native_parity_path)
        if trackb_native_parity_path is not None and trackb_native_parity_path.is_file()
        else None
    )
    trackb_artifact_parity = (
        _read_json(trackb_artifact_parity_path)
        if trackb_artifact_parity_path is not None
        and trackb_artifact_parity_path.is_file()
        else None
    )
    trackb_speed_packet = (
        _read_json(trackb_speed_packet_path)
        if trackb_speed_packet_path is not None and trackb_speed_packet_path.is_file()
        else None
    )
    trackb_successor_design = (
        _read_json(trackb_successor_design_path)
        if trackb_successor_design_path is not None
        and trackb_successor_design_path.is_file()
        else None
    )
    trackb_successor_native_parity = (
        _read_json(trackb_successor_native_parity_path)
        if trackb_successor_native_parity_path is not None
        and trackb_successor_native_parity_path.is_file()
        else None
    )
    trackb_successor_artifact_parity = (
        _read_json(trackb_successor_artifact_parity_path)
        if trackb_successor_artifact_parity_path is not None
        and trackb_successor_artifact_parity_path.is_file()
        else None
    )
    trackb_successor_speed_packet = (
        _read_json(trackb_successor_speed_packet_path)
        if trackb_successor_speed_packet_path is not None
        and trackb_successor_speed_packet_path.is_file()
        else None
    )
    trackb_next_design = (
        _read_json(trackb_next_design_path)
        if trackb_next_design_path is not None and trackb_next_design_path.is_file()
        else None
    )
    trackb_active_route_tile_native_parity = (
        _read_json(trackb_active_route_tile_native_parity_path)
        if trackb_active_route_tile_native_parity_path is not None
        and trackb_active_route_tile_native_parity_path.is_file()
        else None
    )
    trackb_active_route_tile_artifact_parity = (
        _read_json(trackb_active_route_tile_artifact_parity_path)
        if trackb_active_route_tile_artifact_parity_path is not None
        and trackb_active_route_tile_artifact_parity_path.is_file()
        else None
    )
    trackb_active_route_tile_speed_packet = (
        _read_json(trackb_active_route_tile_speed_packet_path)
        if trackb_active_route_tile_speed_packet_path is not None
        and trackb_active_route_tile_speed_packet_path.is_file()
        else None
    )
    trackb_expert_cohort_design = (
        _read_json(trackb_expert_cohort_design_path)
        if trackb_expert_cohort_design_path is not None
        and trackb_expert_cohort_design_path.is_file()
        else None
    )
    trackb_expert_cohort_native_parity = (
        _read_json(trackb_expert_cohort_native_parity_path)
        if trackb_expert_cohort_native_parity_path is not None
        and trackb_expert_cohort_native_parity_path.is_file()
        else None
    )
    trackb_expert_cohort_artifact_parity = (
        _read_json(trackb_expert_cohort_artifact_parity_path)
        if trackb_expert_cohort_artifact_parity_path is not None
        and trackb_expert_cohort_artifact_parity_path.is_file()
        else None
    )
    trackb_expert_cohort_speed_packet = (
        _read_json(trackb_expert_cohort_speed_packet_path)
        if trackb_expert_cohort_speed_packet_path is not None
        and trackb_expert_cohort_speed_packet_path.is_file()
        else None
    )
    trackb_route_batch_segmented_design = (
        _read_json(trackb_route_batch_segmented_design_path)
        if trackb_route_batch_segmented_design_path is not None
        and trackb_route_batch_segmented_design_path.is_file()
        else None
    )
    trackb_route_batch_segmented_native_parity = (
        _read_json(trackb_route_batch_segmented_native_parity_path)
        if trackb_route_batch_segmented_native_parity_path is not None
        and trackb_route_batch_segmented_native_parity_path.is_file()
        else None
    )
    trackb_route_batch_segmented_artifact_parity = (
        _read_json(trackb_route_batch_segmented_artifact_parity_path)
        if trackb_route_batch_segmented_artifact_parity_path is not None
        and trackb_route_batch_segmented_artifact_parity_path.is_file()
        else None
    )
    trackb_route_batch_segmented_speed_packet = (
        _read_json(trackb_route_batch_segmented_speed_packet_path)
        if trackb_route_batch_segmented_speed_packet_path is not None
        and trackb_route_batch_segmented_speed_packet_path.is_file()
        else None
    )
    trackb_component_stream_partial_reduction_design = (
        _read_json(trackb_component_stream_partial_reduction_design_path)
        if trackb_component_stream_partial_reduction_design_path is not None
        and trackb_component_stream_partial_reduction_design_path.is_file()
        else None
    )
    trackb_component_stream_partial_reduction_native_parity = (
        _read_json(trackb_component_stream_partial_reduction_native_parity_path)
        if trackb_component_stream_partial_reduction_native_parity_path is not None
        and trackb_component_stream_partial_reduction_native_parity_path.is_file()
        else None
    )
    trackb_component_stream_tensorops_rejection = (
        _read_json(trackb_component_stream_tensorops_rejection_path)
        if trackb_component_stream_tensorops_rejection_path is not None
        and trackb_component_stream_tensorops_rejection_path.is_file()
        else None
    )
    trackb_next_family_gate = (
        _read_json(trackb_next_family_gate_path)
        if trackb_next_family_gate_path is not None
        and trackb_next_family_gate_path.is_file()
        else None
    )
    trackb_token_cohort_codeword_stream_design = (
        _read_json(trackb_token_cohort_codeword_stream_design_path)
        if trackb_token_cohort_codeword_stream_design_path is not None
        and trackb_token_cohort_codeword_stream_design_path.is_file()
        else None
    )
    trackb_token_cohort_codeword_stream_native_parity = (
        _read_json(trackb_token_cohort_codeword_stream_native_parity_path)
        if trackb_token_cohort_codeword_stream_native_parity_path is not None
        and trackb_token_cohort_codeword_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_cohort_codeword_stream_artifact_parity = (
        _read_json(trackb_token_cohort_codeword_stream_artifact_parity_path)
        if trackb_token_cohort_codeword_stream_artifact_parity_path is not None
        and trackb_token_cohort_codeword_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_cohort_codeword_stream_speed_packet = (
        _read_json(trackb_token_cohort_codeword_stream_speed_packet_path)
        if trackb_token_cohort_codeword_stream_speed_packet_path is not None
        and trackb_token_cohort_codeword_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_cohort_mma_codeword_tile_design = (
        _read_json(trackb_token_cohort_mma_codeword_tile_design_path)
        if trackb_token_cohort_mma_codeword_tile_design_path is not None
        and trackb_token_cohort_mma_codeword_tile_design_path.is_file()
        else None
    )
    trackb_token_cohort_mma_codeword_tile_native_parity = (
        _read_json(trackb_token_cohort_mma_codeword_tile_native_parity_path)
        if trackb_token_cohort_mma_codeword_tile_native_parity_path is not None
        and trackb_token_cohort_mma_codeword_tile_native_parity_path.is_file()
        else None
    )
    trackb_token_cohort_mma_codeword_tile_artifact_parity = (
        _read_json(trackb_token_cohort_mma_codeword_tile_artifact_parity_path)
        if trackb_token_cohort_mma_codeword_tile_artifact_parity_path is not None
        and trackb_token_cohort_mma_codeword_tile_artifact_parity_path.is_file()
        else None
    )
    trackb_token_cohort_mma_codeword_tile_speed_packet = (
        _read_json(trackb_token_cohort_mma_codeword_tile_speed_packet_path)
        if trackb_token_cohort_mma_codeword_tile_speed_packet_path is not None
        and trackb_token_cohort_mma_codeword_tile_speed_packet_path.is_file()
        else None
    )
    trackb_output_stationary_codeword_tile_design = (
        _read_json(trackb_output_stationary_codeword_tile_design_path)
        if trackb_output_stationary_codeword_tile_design_path is not None
        and trackb_output_stationary_codeword_tile_design_path.is_file()
        else None
    )
    trackb_output_stationary_codeword_tile_native_parity = (
        _read_json(trackb_output_stationary_codeword_tile_native_parity_path)
        if trackb_output_stationary_codeword_tile_native_parity_path is not None
        and trackb_output_stationary_codeword_tile_native_parity_path.is_file()
        else None
    )
    trackb_output_stationary_codeword_tile_artifact_parity = (
        _read_json(trackb_output_stationary_codeword_tile_artifact_parity_path)
        if trackb_output_stationary_codeword_tile_artifact_parity_path is not None
        and trackb_output_stationary_codeword_tile_artifact_parity_path.is_file()
        else None
    )
    trackb_output_stationary_codeword_tile_speed_packet = (
        _read_json(trackb_output_stationary_codeword_tile_speed_packet_path)
        if trackb_output_stationary_codeword_tile_speed_packet_path is not None
        and trackb_output_stationary_codeword_tile_speed_packet_path.is_file()
        else None
    )
    trackb_input_stationary_codeword_tile_design = (
        _read_json(trackb_input_stationary_codeword_tile_design_path)
        if trackb_input_stationary_codeword_tile_design_path is not None
        and trackb_input_stationary_codeword_tile_design_path.is_file()
        else None
    )
    trackb_input_stationary_codeword_tile_native_parity = (
        _read_json(trackb_input_stationary_codeword_tile_native_parity_path)
        if trackb_input_stationary_codeword_tile_native_parity_path is not None
        and trackb_input_stationary_codeword_tile_native_parity_path.is_file()
        else None
    )
    trackb_input_stationary_codeword_tile_artifact_parity = (
        _read_json(trackb_input_stationary_codeword_tile_artifact_parity_path)
        if trackb_input_stationary_codeword_tile_artifact_parity_path is not None
        and trackb_input_stationary_codeword_tile_artifact_parity_path.is_file()
        else None
    )
    trackb_input_stationary_codeword_tile_speed_packet = (
        _read_json(trackb_input_stationary_codeword_tile_speed_packet_path)
        if trackb_input_stationary_codeword_tile_speed_packet_path is not None
        and trackb_input_stationary_codeword_tile_speed_packet_path.is_file()
        else None
    )
    trackb_expert_kblock_codeword_factor_reuse_design = (
        _read_json(trackb_expert_kblock_codeword_factor_reuse_design_path)
        if trackb_expert_kblock_codeword_factor_reuse_design_path is not None
        and trackb_expert_kblock_codeword_factor_reuse_design_path.is_file()
        else None
    )
    trackb_expert_kblock_codeword_factor_reuse_native_parity = (
        _read_json(trackb_expert_kblock_codeword_factor_reuse_native_parity_path)
        if trackb_expert_kblock_codeword_factor_reuse_native_parity_path is not None
        and trackb_expert_kblock_codeword_factor_reuse_native_parity_path.is_file()
        else None
    )
    trackb_expert_kblock_codeword_factor_reuse_artifact_parity = (
        _read_json(trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path)
        if trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path
        is not None
        and trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path.is_file()
        else None
    )
    trackb_expert_kblock_codeword_factor_reuse_speed_packet = (
        _read_json(trackb_expert_kblock_codeword_factor_reuse_speed_packet_path)
        if trackb_expert_kblock_codeword_factor_reuse_speed_packet_path is not None
        and trackb_expert_kblock_codeword_factor_reuse_speed_packet_path.is_file()
        else None
    )
    trackb_route_codeword_lut_accumulate_design = (
        _read_json(trackb_route_codeword_lut_accumulate_design_path)
        if trackb_route_codeword_lut_accumulate_design_path is not None
        and trackb_route_codeword_lut_accumulate_design_path.is_file()
        else None
    )
    trackb_route_codeword_lut_accumulate_native_parity = (
        _read_json(trackb_route_codeword_lut_accumulate_native_parity_path)
        if trackb_route_codeword_lut_accumulate_native_parity_path is not None
        and trackb_route_codeword_lut_accumulate_native_parity_path.is_file()
        else None
    )
    trackb_route_codeword_lut_accumulate_artifact_parity = (
        _read_json(trackb_route_codeword_lut_accumulate_artifact_parity_path)
        if trackb_route_codeword_lut_accumulate_artifact_parity_path is not None
        and trackb_route_codeword_lut_accumulate_artifact_parity_path.is_file()
        else None
    )
    trackb_route_codeword_lut_accumulate_speed_packet = (
        _read_json(trackb_route_codeword_lut_accumulate_speed_packet_path)
        if trackb_route_codeword_lut_accumulate_speed_packet_path is not None
        and trackb_route_codeword_lut_accumulate_speed_packet_path.is_file()
        else None
    )
    trackb_rowwise_codeword_tile_accumulate_design = (
        _read_json(trackb_rowwise_codeword_tile_accumulate_design_path)
        if trackb_rowwise_codeword_tile_accumulate_design_path is not None
        and trackb_rowwise_codeword_tile_accumulate_design_path.is_file()
        else None
    )
    trackb_rowwise_codeword_tile_accumulate_native_parity = (
        _read_json(trackb_rowwise_codeword_tile_accumulate_native_parity_path)
        if trackb_rowwise_codeword_tile_accumulate_native_parity_path is not None
        and trackb_rowwise_codeword_tile_accumulate_native_parity_path.is_file()
        else None
    )
    trackb_rowwise_codeword_tile_accumulate_artifact_parity = (
        _read_json(trackb_rowwise_codeword_tile_accumulate_artifact_parity_path)
        if trackb_rowwise_codeword_tile_accumulate_artifact_parity_path is not None
        and trackb_rowwise_codeword_tile_accumulate_artifact_parity_path.is_file()
        else None
    )
    trackb_rowwise_codeword_tile_accumulate_speed_packet = (
        _read_json(trackb_rowwise_codeword_tile_accumulate_speed_packet_path)
        if trackb_rowwise_codeword_tile_accumulate_speed_packet_path is not None
        and trackb_rowwise_codeword_tile_accumulate_speed_packet_path.is_file()
        else None
    )
    trackb_output_tile_local_codeword_lut_design = (
        _read_json(trackb_output_tile_local_codeword_lut_design_path)
        if trackb_output_tile_local_codeword_lut_design_path is not None
        and trackb_output_tile_local_codeword_lut_design_path.is_file()
        else None
    )
    trackb_output_tile_local_codeword_lut_native_parity = (
        _read_json(trackb_output_tile_local_codeword_lut_native_parity_path)
        if trackb_output_tile_local_codeword_lut_native_parity_path is not None
        and trackb_output_tile_local_codeword_lut_native_parity_path.is_file()
        else None
    )
    trackb_output_tile_local_codeword_lut_artifact_parity = (
        _read_json(trackb_output_tile_local_codeword_lut_artifact_parity_path)
        if trackb_output_tile_local_codeword_lut_artifact_parity_path is not None
        and trackb_output_tile_local_codeword_lut_artifact_parity_path.is_file()
        else None
    )
    trackb_output_tile_local_codeword_lut_speed_packet = (
        _read_json(trackb_output_tile_local_codeword_lut_speed_packet_path)
        if trackb_output_tile_local_codeword_lut_speed_packet_path is not None
        and trackb_output_tile_local_codeword_lut_speed_packet_path.is_file()
        else None
    )
    trackb_route_microtile_codeword_block_reduce_design = (
        _read_json(trackb_route_microtile_codeword_block_reduce_design_path)
        if trackb_route_microtile_codeword_block_reduce_design_path is not None
        and trackb_route_microtile_codeword_block_reduce_design_path.is_file()
        else None
    )
    trackb_route_microtile_codeword_block_reduce_native_parity = (
        _read_json(trackb_route_microtile_codeword_block_reduce_native_parity_path)
        if trackb_route_microtile_codeword_block_reduce_native_parity_path is not None
        and trackb_route_microtile_codeword_block_reduce_native_parity_path.is_file()
        else None
    )
    trackb_route_microtile_codeword_block_reduce_artifact_parity = (
        _read_json(trackb_route_microtile_codeword_block_reduce_artifact_parity_path)
        if trackb_route_microtile_codeword_block_reduce_artifact_parity_path is not None
        and trackb_route_microtile_codeword_block_reduce_artifact_parity_path.is_file()
        else None
    )
    trackb_route_microtile_codeword_block_reduce_speed_packet = (
        _read_json(trackb_route_microtile_codeword_block_reduce_speed_packet_path)
        if trackb_route_microtile_codeword_block_reduce_speed_packet_path is not None
        and trackb_route_microtile_codeword_block_reduce_speed_packet_path.is_file()
        else None
    )
    trackb_kblock_wavefront_codeword_scan_design = (
        _read_json(trackb_kblock_wavefront_codeword_scan_design_path)
        if trackb_kblock_wavefront_codeword_scan_design_path is not None
        and trackb_kblock_wavefront_codeword_scan_design_path.is_file()
        else None
    )
    trackb_kblock_wavefront_codeword_scan_native_parity = (
        _read_json(trackb_kblock_wavefront_codeword_scan_native_parity_path)
        if trackb_kblock_wavefront_codeword_scan_native_parity_path is not None
        and trackb_kblock_wavefront_codeword_scan_native_parity_path.is_file()
        else None
    )
    trackb_kblock_wavefront_codeword_scan_artifact_parity = (
        _read_json(trackb_kblock_wavefront_codeword_scan_artifact_parity_path)
        if trackb_kblock_wavefront_codeword_scan_artifact_parity_path is not None
        and trackb_kblock_wavefront_codeword_scan_artifact_parity_path.is_file()
        else None
    )
    trackb_kblock_wavefront_codeword_scan_speed_packet = (
        _read_json(trackb_kblock_wavefront_codeword_scan_speed_packet_path)
        if trackb_kblock_wavefront_codeword_scan_speed_packet_path is not None
        and trackb_kblock_wavefront_codeword_scan_speed_packet_path.is_file()
        else None
    )
    trackb_token_route_output_stripe_pipeline_design = (
        _read_json(trackb_token_route_output_stripe_pipeline_design_path)
        if trackb_token_route_output_stripe_pipeline_design_path is not None
        and trackb_token_route_output_stripe_pipeline_design_path.is_file()
        else None
    )
    trackb_token_route_output_stripe_pipeline_native_parity = (
        _read_json(trackb_token_route_output_stripe_pipeline_native_parity_path)
        if trackb_token_route_output_stripe_pipeline_native_parity_path is not None
        and trackb_token_route_output_stripe_pipeline_native_parity_path.is_file()
        else None
    )
    trackb_token_route_output_stripe_pipeline_artifact_parity = (
        _read_json(trackb_token_route_output_stripe_pipeline_artifact_parity_path)
        if trackb_token_route_output_stripe_pipeline_artifact_parity_path is not None
        and trackb_token_route_output_stripe_pipeline_artifact_parity_path.is_file()
        else None
    )
    trackb_token_route_output_stripe_pipeline_speed_packet = (
        _read_json(trackb_token_route_output_stripe_pipeline_speed_packet_path)
        if trackb_token_route_output_stripe_pipeline_speed_packet_path is not None
        and trackb_token_route_output_stripe_pipeline_speed_packet_path.is_file()
        else None
    )
    trackb_expert_kblock_scale_slot_stream_design = (
        _read_json(trackb_expert_kblock_scale_slot_stream_design_path)
        if trackb_expert_kblock_scale_slot_stream_design_path is not None
        and trackb_expert_kblock_scale_slot_stream_design_path.is_file()
        else None
    )
    trackb_expert_kblock_scale_slot_stream_native_parity = (
        _read_json(trackb_expert_kblock_scale_slot_stream_native_parity_path)
        if trackb_expert_kblock_scale_slot_stream_native_parity_path is not None
        and trackb_expert_kblock_scale_slot_stream_native_parity_path.is_file()
        else None
    )
    trackb_expert_kblock_scale_slot_stream_artifact_parity = (
        _read_json(trackb_expert_kblock_scale_slot_stream_artifact_parity_path)
        if trackb_expert_kblock_scale_slot_stream_artifact_parity_path is not None
        and trackb_expert_kblock_scale_slot_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_expert_kblock_scale_slot_stream_speed_packet = (
        _read_json(trackb_expert_kblock_scale_slot_stream_speed_packet_path)
        if trackb_expert_kblock_scale_slot_stream_speed_packet_path is not None
        and trackb_expert_kblock_scale_slot_stream_speed_packet_path.is_file()
        else None
    )
    trackb_scale_group_route_block_reduce_design = (
        _read_json(trackb_scale_group_route_block_reduce_design_path)
        if trackb_scale_group_route_block_reduce_design_path is not None
        and trackb_scale_group_route_block_reduce_design_path.is_file()
        else None
    )
    trackb_scale_group_route_block_reduce_native_parity = (
        _read_json(trackb_scale_group_route_block_reduce_native_parity_path)
        if trackb_scale_group_route_block_reduce_native_parity_path is not None
        and trackb_scale_group_route_block_reduce_native_parity_path.is_file()
        else None
    )
    trackb_scale_group_route_block_reduce_artifact_parity = (
        _read_json(trackb_scale_group_route_block_reduce_artifact_parity_path)
        if trackb_scale_group_route_block_reduce_artifact_parity_path is not None
        and trackb_scale_group_route_block_reduce_artifact_parity_path.is_file()
        else None
    )
    trackb_scale_group_route_block_reduce_speed_packet = (
        _read_json(trackb_scale_group_route_block_reduce_speed_packet_path)
        if trackb_scale_group_route_block_reduce_speed_packet_path is not None
        and trackb_scale_group_route_block_reduce_speed_packet_path.is_file()
        else None
    )
    trackb_route_block_output_group_stream_design = (
        _read_json(trackb_route_block_output_group_stream_design_path)
        if trackb_route_block_output_group_stream_design_path is not None
        and trackb_route_block_output_group_stream_design_path.is_file()
        else None
    )
    trackb_route_block_output_group_stream_native_parity = (
        _read_json(trackb_route_block_output_group_stream_native_parity_path)
        if trackb_route_block_output_group_stream_native_parity_path is not None
        and trackb_route_block_output_group_stream_native_parity_path.is_file()
        else None
    )
    trackb_route_block_output_group_stream_artifact_parity = (
        _read_json(trackb_route_block_output_group_stream_artifact_parity_path)
        if trackb_route_block_output_group_stream_artifact_parity_path is not None
        and trackb_route_block_output_group_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_route_block_output_group_stream_speed_packet = (
        _read_json(trackb_route_block_output_group_stream_speed_packet_path)
        if trackb_route_block_output_group_stream_speed_packet_path is not None
        and trackb_route_block_output_group_stream_speed_packet_path.is_file()
        else None
    )
    trackb_output_group_pretransposed_codeword_stream_design = (
        _read_json(trackb_output_group_pretransposed_codeword_stream_design_path)
        if trackb_output_group_pretransposed_codeword_stream_design_path is not None
        and trackb_output_group_pretransposed_codeword_stream_design_path.is_file()
        else None
    )
    trackb_output_group_pretransposed_codeword_stream_native_parity = (
        _read_json(
            trackb_output_group_pretransposed_codeword_stream_native_parity_path
        )
        if trackb_output_group_pretransposed_codeword_stream_native_parity_path
        is not None
        and trackb_output_group_pretransposed_codeword_stream_native_parity_path.is_file()
        else None
    )
    trackb_output_group_pretransposed_codeword_stream_artifact_parity = (
        _read_json(
            trackb_output_group_pretransposed_codeword_stream_artifact_parity_path
        )
        if trackb_output_group_pretransposed_codeword_stream_artifact_parity_path
        is not None
        and trackb_output_group_pretransposed_codeword_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_output_group_pretransposed_codeword_stream_speed_packet = (
        _read_json(
            trackb_output_group_pretransposed_codeword_stream_speed_packet_path
        )
        if trackb_output_group_pretransposed_codeword_stream_speed_packet_path
        is not None
        and trackb_output_group_pretransposed_codeword_stream_speed_packet_path.is_file()
        else None
    )
    trackb_kblock_output_group_route_fused_stream_design = (
        _read_json(trackb_kblock_output_group_route_fused_stream_design_path)
        if trackb_kblock_output_group_route_fused_stream_design_path is not None
        and trackb_kblock_output_group_route_fused_stream_design_path.is_file()
        else None
    )
    trackb_kblock_output_group_route_fused_stream_native_parity = (
        _read_json(trackb_kblock_output_group_route_fused_stream_native_parity_path)
        if trackb_kblock_output_group_route_fused_stream_native_parity_path
        is not None
        and trackb_kblock_output_group_route_fused_stream_native_parity_path.is_file()
        else None
    )
    trackb_kblock_output_group_route_fused_stream_artifact_parity = (
        _read_json(trackb_kblock_output_group_route_fused_stream_artifact_parity_path)
        if trackb_kblock_output_group_route_fused_stream_artifact_parity_path
        is not None
        and trackb_kblock_output_group_route_fused_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_kblock_output_group_route_fused_stream_speed_packet = (
        _read_json(trackb_kblock_output_group_route_fused_stream_speed_packet_path)
        if trackb_kblock_output_group_route_fused_stream_speed_packet_path is not None
        and trackb_kblock_output_group_route_fused_stream_speed_packet_path.is_file()
        else None
    )
    trackb_route_tile_output_swizzle_stream_design = (
        _read_json(trackb_route_tile_output_swizzle_stream_design_path)
        if trackb_route_tile_output_swizzle_stream_design_path is not None
        and trackb_route_tile_output_swizzle_stream_design_path.is_file()
        else None
    )
    trackb_route_tile_output_swizzle_stream_native_parity = (
        _read_json(trackb_route_tile_output_swizzle_stream_native_parity_path)
        if trackb_route_tile_output_swizzle_stream_native_parity_path is not None
        and trackb_route_tile_output_swizzle_stream_native_parity_path.is_file()
        else None
    )
    trackb_route_tile_output_swizzle_stream_artifact_parity = (
        _read_json(trackb_route_tile_output_swizzle_stream_artifact_parity_path)
        if trackb_route_tile_output_swizzle_stream_artifact_parity_path is not None
        and trackb_route_tile_output_swizzle_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_route_tile_output_swizzle_stream_speed_packet = (
        _read_json(trackb_route_tile_output_swizzle_stream_speed_packet_path)
        if trackb_route_tile_output_swizzle_stream_speed_packet_path is not None
        and trackb_route_tile_output_swizzle_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_topk_output_tile_stream_design = (
        _read_json(trackb_token_topk_output_tile_stream_design_path)
        if trackb_token_topk_output_tile_stream_design_path is not None
        and trackb_token_topk_output_tile_stream_design_path.is_file()
        else None
    )
    trackb_token_topk_output_tile_stream_native_parity = (
        _read_json(trackb_token_topk_output_tile_stream_native_parity_path)
        if trackb_token_topk_output_tile_stream_native_parity_path is not None
        and trackb_token_topk_output_tile_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_topk_output_tile_stream_artifact_parity = (
        _read_json(trackb_token_topk_output_tile_stream_artifact_parity_path)
        if trackb_token_topk_output_tile_stream_artifact_parity_path is not None
        and trackb_token_topk_output_tile_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_topk_output_tile_stream_speed_packet = (
        _read_json(trackb_token_topk_output_tile_stream_speed_packet_path)
        if trackb_token_topk_output_tile_stream_speed_packet_path is not None
        and trackb_token_topk_output_tile_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_block_output_group_stream_design = (
        _read_json(trackb_token_block_output_group_stream_design_path)
        if trackb_token_block_output_group_stream_design_path is not None
        and trackb_token_block_output_group_stream_design_path.is_file()
        else None
    )
    trackb_token_block_output_group_stream_native_parity = (
        _read_json(trackb_token_block_output_group_stream_native_parity_path)
        if trackb_token_block_output_group_stream_native_parity_path is not None
        and trackb_token_block_output_group_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_block_output_group_stream_artifact_parity = (
        _read_json(trackb_token_block_output_group_stream_artifact_parity_path)
        if trackb_token_block_output_group_stream_artifact_parity_path is not None
        and trackb_token_block_output_group_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_block_output_group_stream_speed_packet = (
        _read_json(trackb_token_block_output_group_stream_speed_packet_path)
        if trackb_token_block_output_group_stream_speed_packet_path is not None
        and trackb_token_block_output_group_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_output_stripe_group_stream_design = (
        _read_json(trackb_token_output_stripe_group_stream_design_path)
        if trackb_token_output_stripe_group_stream_design_path is not None
        and trackb_token_output_stripe_group_stream_design_path.is_file()
        else None
    )
    trackb_token_output_stripe_group_stream_native_parity = (
        _read_json(trackb_token_output_stripe_group_stream_native_parity_path)
        if trackb_token_output_stripe_group_stream_native_parity_path is not None
        and trackb_token_output_stripe_group_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_output_stripe_group_stream_artifact_parity = (
        _read_json(trackb_token_output_stripe_group_stream_artifact_parity_path)
        if trackb_token_output_stripe_group_stream_artifact_parity_path is not None
        and trackb_token_output_stripe_group_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_output_stripe_group_stream_speed_packet = (
        _read_json(trackb_token_output_stripe_group_stream_speed_packet_path)
        if trackb_token_output_stripe_group_stream_speed_packet_path is not None
        and trackb_token_output_stripe_group_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_expert_output_block_stream_design = (
        _read_json(trackb_token_expert_output_block_stream_design_path)
        if trackb_token_expert_output_block_stream_design_path is not None
        and trackb_token_expert_output_block_stream_design_path.is_file()
        else None
    )
    trackb_token_expert_output_block_stream_native_parity = (
        _read_json(trackb_token_expert_output_block_stream_native_parity_path)
        if trackb_token_expert_output_block_stream_native_parity_path is not None
        and trackb_token_expert_output_block_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_expert_output_block_stream_artifact_parity = (
        _read_json(trackb_token_expert_output_block_stream_artifact_parity_path)
        if trackb_token_expert_output_block_stream_artifact_parity_path is not None
        and trackb_token_expert_output_block_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_expert_output_block_stream_speed_packet = (
        _read_json(trackb_token_expert_output_block_stream_speed_packet_path)
        if trackb_token_expert_output_block_stream_speed_packet_path is not None
        and trackb_token_expert_output_block_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_kblock_accumulator_stream_design = (
        _read_json(trackb_token_pair_kblock_accumulator_stream_design_path)
        if trackb_token_pair_kblock_accumulator_stream_design_path is not None
        and trackb_token_pair_kblock_accumulator_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_kblock_accumulator_stream_native_parity = (
        _read_json(trackb_token_pair_kblock_accumulator_stream_native_parity_path)
        if trackb_token_pair_kblock_accumulator_stream_native_parity_path is not None
        and trackb_token_pair_kblock_accumulator_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_kblock_accumulator_stream_artifact_parity = (
        _read_json(trackb_token_pair_kblock_accumulator_stream_artifact_parity_path)
        if trackb_token_pair_kblock_accumulator_stream_artifact_parity_path is not None
        and trackb_token_pair_kblock_accumulator_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_kblock_accumulator_stream_speed_packet = (
        _read_json(trackb_token_pair_kblock_accumulator_stream_speed_packet_path)
        if trackb_token_pair_kblock_accumulator_stream_speed_packet_path is not None
        and trackb_token_pair_kblock_accumulator_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_output_group_stream_design = (
        _read_json(trackb_token_pair_output_group_stream_design_path)
        if trackb_token_pair_output_group_stream_design_path is not None
        and trackb_token_pair_output_group_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_output_group_stream_native_parity = (
        _read_json(trackb_token_pair_output_group_stream_native_parity_path)
        if trackb_token_pair_output_group_stream_native_parity_path is not None
        and trackb_token_pair_output_group_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_output_group_stream_artifact_parity = (
        _read_json(trackb_token_pair_output_group_stream_artifact_parity_path)
        if trackb_token_pair_output_group_stream_artifact_parity_path is not None
        and trackb_token_pair_output_group_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_output_group_stream_speed_packet = (
        _read_json(trackb_token_pair_output_group_stream_speed_packet_path)
        if trackb_token_pair_output_group_stream_speed_packet_path is not None
        and trackb_token_pair_output_group_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_group_stream_design = (
        _read_json(trackb_token_pair_slot_topk_output_group_stream_design_path)
        if trackb_token_pair_slot_topk_output_group_stream_design_path is not None
        and trackb_token_pair_slot_topk_output_group_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_group_stream_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_output_group_stream_native_parity_path
        )
        if trackb_token_pair_slot_topk_output_group_stream_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_output_group_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_group_stream_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_group_stream_speed_packet = (
        _read_json(trackb_token_pair_slot_topk_output_group_stream_speed_packet_path)
        if trackb_token_pair_slot_topk_output_group_stream_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_design = (
        _read_json(trackb_token_pair_slot_topk_codeword_group_pipeline_design_path)
        if trackb_token_pair_slot_topk_codeword_group_pipeline_design_path is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path
        )
        if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet = (
        _read_json(
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path
        )
        if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design = (
        _read_json(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path
        )
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path
        )
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet = (
        _read_json(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path
        )
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design = (
        _read_json(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path
        )
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path
        )
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet = (
        _read_json(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path
        )
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_design = (
        _read_json(trackb_token_pair_slot_topk_kblock_microtile_stream_design_path)
        if trackb_token_pair_slot_topk_kblock_microtile_stream_design_path is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path
        )
        if trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet = (
        _read_json(trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path)
        if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_design = (
        _read_json(trackb_token_pair_slot_topk_output_tile_fused_stream_design_path)
        if trackb_token_pair_slot_topk_output_tile_fused_stream_design_path
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_design_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity = (
        _read_json(
            trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path
        )
        if trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity = (
        _read_json(
            trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path
        )
        if trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path.is_file()
        else None
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet = (
        _read_json(
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path
        )
        if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path.is_file()
        else None
    )
    qwen = _read_json(qwen_gate_path)
    glm52_layer3 = _read_json(glm52_layer3_path)
    glm52_layer77 = _read_json(glm52_layer77_path)
    cache_source = _read_json(cache_source_scan_path)
    single_host_cache_attempt = (
        _read_json(single_host_cache_attempt_path)
        if single_host_cache_attempt_path is not None
        and single_host_cache_attempt_path.is_file()
        else None
    )
    layer_split_recommendation = (
        _read_json(layer_split_recommendation_path)
        if layer_split_recommendation_path is not None
        and layer_split_recommendation_path.is_file()
        else None
    )
    rdma_topology_audit = (
        _read_json(rdma_topology_audit_path)
        if rdma_topology_audit_path is not None
        and rdma_topology_audit_path.is_file()
        else None
    )
    rdma_topology = _rdma_topology_summary(rdma_topology_audit)
    transport_payloads = [workstream1, trackb, cache_source]
    if workstream1_fixed_bump is not None:
        transport_payloads.append(workstream1_fixed_bump)
    if trackb_design is not None:
        transport_payloads.append(trackb_design)
    if trackb_source_guard is not None:
        transport_payloads.append(trackb_source_guard)
    if trackb_native_parity is not None:
        transport_payloads.append(trackb_native_parity)
    if trackb_artifact_parity is not None:
        transport_payloads.append(trackb_artifact_parity)
    if trackb_speed_packet is not None:
        transport_payloads.append(trackb_speed_packet)
    if trackb_successor_design is not None:
        transport_payloads.append(trackb_successor_design)
    if trackb_successor_native_parity is not None:
        transport_payloads.append(trackb_successor_native_parity)
    if trackb_successor_artifact_parity is not None:
        transport_payloads.append(trackb_successor_artifact_parity)
    if trackb_successor_speed_packet is not None:
        transport_payloads.append(trackb_successor_speed_packet)
    if trackb_next_design is not None:
        transport_payloads.append(trackb_next_design)
    if trackb_active_route_tile_native_parity is not None:
        transport_payloads.append(trackb_active_route_tile_native_parity)
    if trackb_active_route_tile_artifact_parity is not None:
        transport_payloads.append(trackb_active_route_tile_artifact_parity)
    if trackb_active_route_tile_speed_packet is not None:
        transport_payloads.append(trackb_active_route_tile_speed_packet)
    if trackb_expert_cohort_design is not None:
        transport_payloads.append(trackb_expert_cohort_design)
    if trackb_expert_cohort_native_parity is not None:
        transport_payloads.append(trackb_expert_cohort_native_parity)
    if trackb_expert_cohort_artifact_parity is not None:
        transport_payloads.append(trackb_expert_cohort_artifact_parity)
    if trackb_expert_cohort_speed_packet is not None:
        transport_payloads.append(trackb_expert_cohort_speed_packet)
    if trackb_route_batch_segmented_design is not None:
        transport_payloads.append(trackb_route_batch_segmented_design)
    if trackb_route_batch_segmented_native_parity is not None:
        transport_payloads.append(trackb_route_batch_segmented_native_parity)
    if trackb_route_batch_segmented_artifact_parity is not None:
        transport_payloads.append(trackb_route_batch_segmented_artifact_parity)
    if trackb_route_batch_segmented_speed_packet is not None:
        transport_payloads.append(trackb_route_batch_segmented_speed_packet)
    if trackb_component_stream_partial_reduction_design is not None:
        transport_payloads.append(trackb_component_stream_partial_reduction_design)
    if trackb_component_stream_partial_reduction_native_parity is not None:
        transport_payloads.append(
            trackb_component_stream_partial_reduction_native_parity
        )
    if trackb_component_stream_tensorops_rejection is not None:
        transport_payloads.append(trackb_component_stream_tensorops_rejection)
    if trackb_next_family_gate is not None:
        transport_payloads.append(trackb_next_family_gate)
    if trackb_token_cohort_codeword_stream_design is not None:
        transport_payloads.append(trackb_token_cohort_codeword_stream_design)
    if trackb_token_cohort_codeword_stream_native_parity is not None:
        transport_payloads.append(trackb_token_cohort_codeword_stream_native_parity)
    if trackb_token_cohort_codeword_stream_artifact_parity is not None:
        transport_payloads.append(trackb_token_cohort_codeword_stream_artifact_parity)
    if trackb_token_cohort_codeword_stream_speed_packet is not None:
        transport_payloads.append(trackb_token_cohort_codeword_stream_speed_packet)
    if trackb_token_cohort_mma_codeword_tile_design is not None:
        transport_payloads.append(trackb_token_cohort_mma_codeword_tile_design)
    if trackb_token_cohort_mma_codeword_tile_native_parity is not None:
        transport_payloads.append(trackb_token_cohort_mma_codeword_tile_native_parity)
    if trackb_token_cohort_mma_codeword_tile_artifact_parity is not None:
        transport_payloads.append(trackb_token_cohort_mma_codeword_tile_artifact_parity)
    if trackb_token_cohort_mma_codeword_tile_speed_packet is not None:
        transport_payloads.append(trackb_token_cohort_mma_codeword_tile_speed_packet)
    if trackb_output_stationary_codeword_tile_design is not None:
        transport_payloads.append(trackb_output_stationary_codeword_tile_design)
    if trackb_output_stationary_codeword_tile_native_parity is not None:
        transport_payloads.append(
            trackb_output_stationary_codeword_tile_native_parity
        )
    if trackb_output_stationary_codeword_tile_artifact_parity is not None:
        transport_payloads.append(
            trackb_output_stationary_codeword_tile_artifact_parity
        )
    if trackb_output_stationary_codeword_tile_speed_packet is not None:
        transport_payloads.append(trackb_output_stationary_codeword_tile_speed_packet)
    if trackb_input_stationary_codeword_tile_design is not None:
        transport_payloads.append(trackb_input_stationary_codeword_tile_design)
    if trackb_input_stationary_codeword_tile_native_parity is not None:
        transport_payloads.append(
            trackb_input_stationary_codeword_tile_native_parity
        )
    if trackb_input_stationary_codeword_tile_artifact_parity is not None:
        transport_payloads.append(
            trackb_input_stationary_codeword_tile_artifact_parity
        )
    if trackb_input_stationary_codeword_tile_speed_packet is not None:
        transport_payloads.append(trackb_input_stationary_codeword_tile_speed_packet)
    if trackb_expert_kblock_codeword_factor_reuse_design is not None:
        transport_payloads.append(trackb_expert_kblock_codeword_factor_reuse_design)
    if trackb_expert_kblock_codeword_factor_reuse_native_parity is not None:
        transport_payloads.append(
            trackb_expert_kblock_codeword_factor_reuse_native_parity
        )
    if trackb_expert_kblock_codeword_factor_reuse_artifact_parity is not None:
        transport_payloads.append(
            trackb_expert_kblock_codeword_factor_reuse_artifact_parity
        )
    if trackb_expert_kblock_codeword_factor_reuse_speed_packet is not None:
        transport_payloads.append(
            trackb_expert_kblock_codeword_factor_reuse_speed_packet
        )
    if trackb_route_codeword_lut_accumulate_design is not None:
        transport_payloads.append(trackb_route_codeword_lut_accumulate_design)
    if trackb_route_codeword_lut_accumulate_native_parity is not None:
        transport_payloads.append(trackb_route_codeword_lut_accumulate_native_parity)
    if trackb_route_codeword_lut_accumulate_artifact_parity is not None:
        transport_payloads.append(trackb_route_codeword_lut_accumulate_artifact_parity)
    if trackb_route_codeword_lut_accumulate_speed_packet is not None:
        transport_payloads.append(trackb_route_codeword_lut_accumulate_speed_packet)
    if trackb_rowwise_codeword_tile_accumulate_design is not None:
        transport_payloads.append(trackb_rowwise_codeword_tile_accumulate_design)
    if trackb_rowwise_codeword_tile_accumulate_native_parity is not None:
        transport_payloads.append(
            trackb_rowwise_codeword_tile_accumulate_native_parity
        )
    if trackb_rowwise_codeword_tile_accumulate_artifact_parity is not None:
        transport_payloads.append(
            trackb_rowwise_codeword_tile_accumulate_artifact_parity
        )
    if trackb_rowwise_codeword_tile_accumulate_speed_packet is not None:
        transport_payloads.append(trackb_rowwise_codeword_tile_accumulate_speed_packet)
    if trackb_output_tile_local_codeword_lut_design is not None:
        transport_payloads.append(trackb_output_tile_local_codeword_lut_design)
    if trackb_output_tile_local_codeword_lut_native_parity is not None:
        transport_payloads.append(trackb_output_tile_local_codeword_lut_native_parity)
    if trackb_output_tile_local_codeword_lut_artifact_parity is not None:
        transport_payloads.append(
            trackb_output_tile_local_codeword_lut_artifact_parity
        )
    if trackb_output_tile_local_codeword_lut_speed_packet is not None:
        transport_payloads.append(trackb_output_tile_local_codeword_lut_speed_packet)
    if trackb_route_microtile_codeword_block_reduce_design is not None:
        transport_payloads.append(trackb_route_microtile_codeword_block_reduce_design)
    if trackb_route_microtile_codeword_block_reduce_native_parity is not None:
        transport_payloads.append(
            trackb_route_microtile_codeword_block_reduce_native_parity
        )
    if trackb_route_microtile_codeword_block_reduce_artifact_parity is not None:
        transport_payloads.append(
            trackb_route_microtile_codeword_block_reduce_artifact_parity
        )
    if trackb_route_microtile_codeword_block_reduce_speed_packet is not None:
        transport_payloads.append(
            trackb_route_microtile_codeword_block_reduce_speed_packet
        )
    if trackb_kblock_wavefront_codeword_scan_design is not None:
        transport_payloads.append(trackb_kblock_wavefront_codeword_scan_design)
    if trackb_kblock_wavefront_codeword_scan_native_parity is not None:
        transport_payloads.append(
            trackb_kblock_wavefront_codeword_scan_native_parity
        )
    if trackb_kblock_wavefront_codeword_scan_artifact_parity is not None:
        transport_payloads.append(
            trackb_kblock_wavefront_codeword_scan_artifact_parity
        )
    if trackb_kblock_wavefront_codeword_scan_speed_packet is not None:
        transport_payloads.append(trackb_kblock_wavefront_codeword_scan_speed_packet)
    if trackb_token_route_output_stripe_pipeline_design is not None:
        transport_payloads.append(trackb_token_route_output_stripe_pipeline_design)
    if trackb_token_route_output_stripe_pipeline_native_parity is not None:
        transport_payloads.append(
            trackb_token_route_output_stripe_pipeline_native_parity
        )
    if trackb_token_route_output_stripe_pipeline_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_route_output_stripe_pipeline_artifact_parity
        )
    if trackb_token_route_output_stripe_pipeline_speed_packet is not None:
        transport_payloads.append(
            trackb_token_route_output_stripe_pipeline_speed_packet
        )
    if trackb_expert_kblock_scale_slot_stream_design is not None:
        transport_payloads.append(trackb_expert_kblock_scale_slot_stream_design)
    if trackb_expert_kblock_scale_slot_stream_native_parity is not None:
        transport_payloads.append(
            trackb_expert_kblock_scale_slot_stream_native_parity
        )
    if trackb_expert_kblock_scale_slot_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_expert_kblock_scale_slot_stream_artifact_parity
        )
    if trackb_expert_kblock_scale_slot_stream_speed_packet is not None:
        transport_payloads.append(trackb_expert_kblock_scale_slot_stream_speed_packet)
    if trackb_scale_group_route_block_reduce_design is not None:
        transport_payloads.append(trackb_scale_group_route_block_reduce_design)
    if trackb_scale_group_route_block_reduce_native_parity is not None:
        transport_payloads.append(
            trackb_scale_group_route_block_reduce_native_parity
        )
    if trackb_scale_group_route_block_reduce_artifact_parity is not None:
        transport_payloads.append(
            trackb_scale_group_route_block_reduce_artifact_parity
        )
    if trackb_scale_group_route_block_reduce_speed_packet is not None:
        transport_payloads.append(trackb_scale_group_route_block_reduce_speed_packet)
    if trackb_route_block_output_group_stream_design is not None:
        transport_payloads.append(trackb_route_block_output_group_stream_design)
    if trackb_route_block_output_group_stream_native_parity is not None:
        transport_payloads.append(
            trackb_route_block_output_group_stream_native_parity
        )
    if trackb_route_block_output_group_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_route_block_output_group_stream_artifact_parity
        )
    if trackb_route_block_output_group_stream_speed_packet is not None:
        transport_payloads.append(trackb_route_block_output_group_stream_speed_packet)
    if trackb_output_group_pretransposed_codeword_stream_design is not None:
        transport_payloads.append(
            trackb_output_group_pretransposed_codeword_stream_design
        )
    if trackb_output_group_pretransposed_codeword_stream_native_parity is not None:
        transport_payloads.append(
            trackb_output_group_pretransposed_codeword_stream_native_parity
        )
    if trackb_output_group_pretransposed_codeword_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_output_group_pretransposed_codeword_stream_artifact_parity
        )
    if trackb_output_group_pretransposed_codeword_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_output_group_pretransposed_codeword_stream_speed_packet
        )
    if trackb_kblock_output_group_route_fused_stream_design is not None:
        transport_payloads.append(trackb_kblock_output_group_route_fused_stream_design)
    if trackb_kblock_output_group_route_fused_stream_native_parity is not None:
        transport_payloads.append(
            trackb_kblock_output_group_route_fused_stream_native_parity
        )
    if trackb_kblock_output_group_route_fused_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_kblock_output_group_route_fused_stream_artifact_parity
        )
    if trackb_kblock_output_group_route_fused_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_kblock_output_group_route_fused_stream_speed_packet
        )
    if trackb_route_tile_output_swizzle_stream_design is not None:
        transport_payloads.append(trackb_route_tile_output_swizzle_stream_design)
    if trackb_route_tile_output_swizzle_stream_native_parity is not None:
        transport_payloads.append(
            trackb_route_tile_output_swizzle_stream_native_parity
        )
    if trackb_route_tile_output_swizzle_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_route_tile_output_swizzle_stream_artifact_parity
        )
    if trackb_route_tile_output_swizzle_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_route_tile_output_swizzle_stream_speed_packet
        )
    if trackb_token_topk_output_tile_stream_design is not None:
        transport_payloads.append(trackb_token_topk_output_tile_stream_design)
    if trackb_token_topk_output_tile_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_topk_output_tile_stream_native_parity
        )
    if trackb_token_topk_output_tile_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_topk_output_tile_stream_artifact_parity
        )
    if trackb_token_topk_output_tile_stream_speed_packet is not None:
        transport_payloads.append(trackb_token_topk_output_tile_stream_speed_packet)
    if trackb_token_block_output_group_stream_design is not None:
        transport_payloads.append(trackb_token_block_output_group_stream_design)
    if trackb_token_block_output_group_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_block_output_group_stream_native_parity
        )
    if trackb_token_block_output_group_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_block_output_group_stream_artifact_parity
        )
    if trackb_token_block_output_group_stream_speed_packet is not None:
        transport_payloads.append(trackb_token_block_output_group_stream_speed_packet)
    if trackb_token_output_stripe_group_stream_design is not None:
        transport_payloads.append(trackb_token_output_stripe_group_stream_design)
    if trackb_token_output_stripe_group_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_output_stripe_group_stream_native_parity
        )
    if trackb_token_output_stripe_group_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_output_stripe_group_stream_artifact_parity
        )
    if trackb_token_output_stripe_group_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_output_stripe_group_stream_speed_packet
        )
    if trackb_token_expert_output_block_stream_design is not None:
        transport_payloads.append(trackb_token_expert_output_block_stream_design)
    if trackb_token_expert_output_block_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_expert_output_block_stream_native_parity
        )
    if trackb_token_expert_output_block_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_expert_output_block_stream_artifact_parity
        )
    if trackb_token_expert_output_block_stream_speed_packet is not None:
        transport_payloads.append(trackb_token_expert_output_block_stream_speed_packet)
    if trackb_token_pair_kblock_accumulator_stream_design is not None:
        transport_payloads.append(trackb_token_pair_kblock_accumulator_stream_design)
    if trackb_token_pair_kblock_accumulator_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_pair_kblock_accumulator_stream_native_parity
        )
    if trackb_token_pair_kblock_accumulator_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_pair_kblock_accumulator_stream_artifact_parity
        )
    if trackb_token_pair_kblock_accumulator_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_kblock_accumulator_stream_speed_packet
        )
    if trackb_token_pair_output_group_stream_design is not None:
        transport_payloads.append(trackb_token_pair_output_group_stream_design)
    if trackb_token_pair_output_group_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_pair_output_group_stream_native_parity
        )
    if trackb_token_pair_output_group_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_pair_output_group_stream_artifact_parity
        )
    if trackb_token_pair_output_group_stream_speed_packet is not None:
        transport_payloads.append(trackb_token_pair_output_group_stream_speed_packet)
    if trackb_token_pair_slot_topk_output_group_stream_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_group_stream_design
        )
    if trackb_token_pair_slot_topk_output_group_stream_native_parity is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_group_stream_native_parity
        )
    if trackb_token_pair_slot_topk_output_group_stream_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_group_stream_artifact_parity
        )
    if trackb_token_pair_slot_topk_output_group_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_group_stream_speed_packet
        )
    if trackb_token_pair_slot_topk_codeword_group_pipeline_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_codeword_group_pipeline_design
        )
    if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity
        )
    if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity
        )
    if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
        )
    if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design
        )
    if (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
        )
    if (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
        )
    if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
        )
    if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design
        )
    if (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
        )
    if (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
        )
    if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
        )
    if trackb_token_pair_slot_topk_kblock_microtile_stream_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_kblock_microtile_stream_design
        )
    if (
        trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
        )
    if (
        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
        )
    if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet
        )
    if trackb_token_pair_slot_topk_output_tile_fused_stream_design is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_tile_fused_stream_design
        )
    if (
        trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
        )
    if (
        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
        is not None
    ):
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
        )
    if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet is not None:
        transport_payloads.append(
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
        )
    if single_host_cache_attempt is not None:
        transport_payloads.append(single_host_cache_attempt)
    if rdma_topology_audit is not None:
        transport_payloads.append(rdma_topology_audit)
    transport = _transport_flags(*transport_payloads)

    qwen_pass = bool(qwen.get("family_gate_pass"))
    qwen_has_explicit_thresholds = _qwen_family_thresholds(qwen) is not None
    qwen_hardening_requirements = _qwen_hardening_requirements(qwen)
    qwen_hardened_pass = qwen_pass and not qwen_hardening_requirements
    qwen_public_status = (
        "qwen_family_gate_ready"
        if qwen_hardened_pass
        else "qwen_family_gate_scaffolding_not_public_ready"
        if qwen_pass and qwen_has_explicit_thresholds
        else qwen.get("family_gate_status")
    )
    qwen_front_status = (
        "hardened_family_gate_green"
        if qwen_hardened_pass and qwen_has_explicit_thresholds
        else "first_family_gate_scaffolding"
        if qwen_pass and qwen_has_explicit_thresholds
        else "first_family_gate_green"
        if qwen_pass
        else "needs_attention"
    )
    trackb_requires_new_family = (
        trackb.get("frontier_decision")
        == "track_b_current_frontier_requires_new_rhs_or_kernel_family"
    )
    trackb_design_ready = (
        trackb_design is not None
        and trackb_design.get("decision")
        == "route_slot_codeword_stream_ready_for_source_structure_probe"
        and bool(
            trackb_design.get("selector_verdict", {}).get("candidate_ready_for_source_probe")
        )
    )
    route_slot_guard = (
        trackb_source_guard.get("route_slot_codeword_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    route_slot_mma_guard = (
        trackb_source_guard.get("route_slot_mma_codeword_tile_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    active_route_tile_guard = (
        trackb_source_guard.get("active_route_tile_codeword_outer_product_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    expert_cohort_guard = (
        trackb_source_guard.get("expert_cohort_codeword_broadcast_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    route_batch_segmented_guard = (
        trackb_source_guard.get("route_batch_segmented_codeword_reduce_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    component_stream_partial_reduction_guard = (
        trackb_source_guard.get("component_stream_partial_reduction_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_cohort_codeword_stream_guard = (
        trackb_source_guard.get("token_cohort_codeword_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_cohort_mma_codeword_tile_guard = (
        trackb_source_guard.get("token_cohort_mma_codeword_tile_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    output_stationary_codeword_tile_guard = (
        trackb_source_guard.get("output_stationary_codeword_tile_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    input_stationary_codeword_tile_guard = (
        trackb_source_guard.get("input_stationary_codeword_tile_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    expert_kblock_codeword_factor_reuse_guard = (
        trackb_source_guard.get("expert_kblock_codeword_factor_reuse_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    route_codeword_lut_accumulate_guard = (
        trackb_source_guard.get("route_codeword_lut_accumulate_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    rowwise_codeword_tile_accumulate_guard = (
        trackb_source_guard.get("rowwise_codeword_tile_accumulate_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    output_tile_local_codeword_lut_guard = (
        trackb_source_guard.get("output_tile_local_codeword_lut_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    route_microtile_codeword_block_reduce_guard = (
        trackb_source_guard.get(
            "route_microtile_codeword_block_reduce_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    kblock_wavefront_codeword_scan_guard = (
        trackb_source_guard.get("kblock_wavefront_codeword_scan_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_route_output_stripe_pipeline_guard = (
        trackb_source_guard.get("token_route_output_stripe_pipeline_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    expert_kblock_scale_slot_stream_guard = (
        trackb_source_guard.get("expert_kblock_scale_slot_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    scale_group_route_block_reduce_guard = (
        trackb_source_guard.get("scale_group_route_block_reduce_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    route_block_output_group_stream_guard = (
        trackb_source_guard.get("route_block_output_group_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    output_group_pretransposed_codeword_stream_guard = (
        trackb_source_guard.get(
            "output_group_pretransposed_codeword_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    kblock_output_group_route_fused_stream_guard = (
        trackb_source_guard.get(
            "kblock_output_group_route_fused_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    route_tile_output_swizzle_stream_guard = (
        trackb_source_guard.get("route_tile_output_swizzle_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_topk_output_tile_stream_guard = (
        trackb_source_guard.get("token_topk_output_tile_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_block_output_group_stream_guard = (
        trackb_source_guard.get("token_block_output_group_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_output_stripe_group_stream_guard = (
        trackb_source_guard.get("token_output_stripe_group_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_expert_output_block_stream_guard = (
        trackb_source_guard.get("token_expert_output_block_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_pair_kblock_accumulator_stream_guard = (
        trackb_source_guard.get(
            "token_pair_kblock_accumulator_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_output_group_stream_guard = (
        trackb_source_guard.get("token_pair_output_group_stream_guardrail", {})
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_output_group_stream_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_output_group_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_codeword_group_pipeline_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_codeword_group_pipeline_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_scale_slot_broadcast_stream_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_scale_slot_broadcast_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_route_bucket_codeword_reduce_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_route_bucket_codeword_reduce_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_kblock_microtile_stream_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_kblock_microtile_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    token_pair_slot_topk_output_tile_fused_stream_guard = (
        trackb_source_guard.get(
            "token_pair_slot_topk_output_tile_fused_stream_guardrail", {}
        )
        if trackb_source_guard is not None
        else {}
    )
    trackb_source_guard_missing = (
        route_slot_guard.get("decision") == "missing_route_slot_codeword_stream_source"
    )
    trackb_source_guard_present = (
        route_slot_guard.get("decision")
        == "route_slot_codeword_stream_source_guardrail_present"
    )
    trackb_native_parity_pass = (
        trackb_native_parity is not None
        and trackb_native_parity.get("decision")
        == "route_slot_codeword_stream_native_parity_pass"
        and bool(trackb_native_parity.get("passes_native_parity"))
    )
    trackb_artifact_parity_pass = (
        trackb_artifact_parity is not None
        and trackb_artifact_parity.get("decision")
        == "route_slot_codeword_stream_air_artifact_parity_pass"
        and bool(trackb_artifact_parity.get("passes_artifact_parity"))
    )
    trackb_speed_rejected = (
        trackb_speed_packet is not None
        and trackb_speed_packet.get("decision")
        == "reject_route_slot_codeword_stream_speed_path"
    )
    trackb_speed_lane_s_pass = (
        trackb_speed_packet is not None
        and trackb_speed_packet.get("decision")
        == "route_slot_codeword_stream_speed_packet_lane_s_pass"
        and bool(trackb_speed_packet.get("all_lane_s_pass"))
    )
    trackb_successor_design_ready = (
        trackb_successor_design is not None
        and trackb_successor_design.get("decision")
        == "route_slot_mma_codeword_tile_ready_for_source_structure_probe"
        and bool(
            trackb_successor_design.get("selector_verdict", {}).get(
                "candidate_ready_for_source_probe"
            )
        )
    )
    trackb_successor_source_guard_missing = (
        trackb_successor_design_ready
        and route_slot_mma_guard.get("decision")
        == "missing_route_slot_mma_codeword_tile_source"
    )
    trackb_successor_source_guard_present = (
        trackb_successor_design_ready
        and route_slot_mma_guard.get("decision")
        == "route_slot_mma_codeword_tile_source_guardrail_present"
    )
    trackb_successor_native_parity_pass = (
        trackb_successor_native_parity is not None
        and trackb_successor_native_parity.get("decision")
        == "route_slot_mma_codeword_tile_native_parity_pass"
        and bool(trackb_successor_native_parity.get("passes_native_parity"))
    )
    trackb_successor_artifact_parity_pass = (
        trackb_successor_artifact_parity is not None
        and trackb_successor_artifact_parity.get("decision")
        == "route_slot_mma_codeword_tile_air_artifact_parity_pass"
        and bool(trackb_successor_artifact_parity.get("passes_artifact_parity"))
    )
    trackb_successor_speed_rejected = (
        trackb_successor_speed_packet is not None
        and trackb_successor_speed_packet.get("decision")
        == "reject_route_slot_mma_codeword_tile_speed_path"
    )
    trackb_successor_speed_lane_s_pass = (
        trackb_successor_speed_packet is not None
        and trackb_successor_speed_packet.get("decision")
        == "route_slot_mma_codeword_tile_speed_packet_lane_s_pass"
        and bool(trackb_successor_speed_packet.get("all_lane_s_pass"))
    )
    trackb_next_design_ready = (
        trackb_next_design is not None
        and trackb_next_design.get("decision")
        == "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
        and bool(
            trackb_next_design.get("selector_verdict", {}).get(
                "candidate_ready_for_source_probe"
            )
        )
    )
    trackb_next_source_guard_missing = (
        trackb_next_design_ready
        and active_route_tile_guard.get("decision")
        == "missing_active_route_tile_codeword_outer_product_source"
    )
    trackb_next_source_guard_present = (
        trackb_next_design_ready
        and active_route_tile_guard.get("decision")
        == "active_route_tile_codeword_outer_product_source_guardrail_present"
    )
    trackb_active_route_tile_native_parity_pass = (
        trackb_active_route_tile_native_parity is not None
        and trackb_active_route_tile_native_parity.get("decision")
        == "active_route_tile_codeword_outer_product_native_parity_pass"
        and bool(trackb_active_route_tile_native_parity.get("passes_native_parity"))
    )
    trackb_active_route_tile_artifact_parity_pass = (
        trackb_active_route_tile_artifact_parity is not None
        and trackb_active_route_tile_artifact_parity.get("decision")
        == "active_route_tile_codeword_outer_product_air_artifact_parity_pass"
        and bool(trackb_active_route_tile_artifact_parity.get("passes_artifact_parity"))
    )
    trackb_active_route_tile_speed_rejected = (
        trackb_active_route_tile_speed_packet is not None
        and trackb_active_route_tile_speed_packet.get("decision")
        == "reject_active_route_tile_codeword_outer_product_speed_path"
    )
    trackb_active_route_tile_speed_lane_s_pass = (
        trackb_active_route_tile_speed_packet is not None
        and trackb_active_route_tile_speed_packet.get("decision")
        == "active_route_tile_codeword_outer_product_speed_packet_lane_s_pass"
        and bool(trackb_active_route_tile_speed_packet.get("all_lane_s_pass"))
    )
    trackb_expert_cohort_design_ready = (
        trackb_expert_cohort_design is not None
        and trackb_expert_cohort_design.get("decision")
        == "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
        and bool(
            trackb_expert_cohort_design.get("selector_verdict", {}).get(
                "candidate_ready_for_source_probe"
            )
        )
    )
    trackb_expert_cohort_source_guard_missing = (
        trackb_expert_cohort_design_ready
        and expert_cohort_guard.get("decision")
        == "missing_expert_cohort_codeword_broadcast_source"
    )
    trackb_expert_cohort_source_guard_present = (
        trackb_expert_cohort_design_ready
        and expert_cohort_guard.get("decision")
        == "expert_cohort_codeword_broadcast_source_guardrail_present"
    )
    trackb_expert_cohort_native_parity_pass = (
        trackb_expert_cohort_native_parity is not None
        and trackb_expert_cohort_native_parity.get("decision")
        == "expert_cohort_codeword_broadcast_native_parity_pass"
        and bool(trackb_expert_cohort_native_parity.get("passes_native_parity"))
    )
    trackb_expert_cohort_artifact_parity_pass = (
        trackb_expert_cohort_artifact_parity is not None
        and trackb_expert_cohort_artifact_parity.get("decision")
        == "expert_cohort_codeword_broadcast_air_artifact_parity_pass"
        and bool(trackb_expert_cohort_artifact_parity.get("passes_artifact_parity"))
    )
    trackb_expert_cohort_speed_rejected = (
        trackb_expert_cohort_speed_packet is not None
        and trackb_expert_cohort_speed_packet.get("decision")
        == "reject_expert_cohort_codeword_broadcast_speed_path"
    )
    trackb_expert_cohort_speed_lane_s_pass = (
        trackb_expert_cohort_speed_packet is not None
        and trackb_expert_cohort_speed_packet.get("decision")
        == "expert_cohort_codeword_broadcast_speed_packet_lane_s_pass"
        and bool(trackb_expert_cohort_speed_packet.get("all_lane_s_pass"))
    )
    trackb_route_batch_segmented_design_ready = (
        trackb_route_batch_segmented_design is not None
        and trackb_route_batch_segmented_design.get("decision")
        == "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
        and bool(
            trackb_route_batch_segmented_design.get("selector_verdict", {}).get(
                "candidate_ready_for_source_probe"
            )
        )
    )
    trackb_route_batch_segmented_source_guard_missing = (
        trackb_route_batch_segmented_design_ready
        and route_batch_segmented_guard.get("decision")
        == "missing_route_batch_segmented_codeword_reduce_source"
    )
    trackb_route_batch_segmented_source_guard_present = (
        trackb_route_batch_segmented_design_ready
        and route_batch_segmented_guard.get("decision")
        == "route_batch_segmented_codeword_reduce_source_guardrail_present"
    )
    trackb_route_batch_segmented_native_parity_pass = (
        trackb_route_batch_segmented_native_parity is not None
        and trackb_route_batch_segmented_native_parity.get("decision")
        == "route_batch_segmented_codeword_reduce_native_parity_pass"
        and bool(trackb_route_batch_segmented_native_parity.get("passes_native_parity"))
    )
    trackb_route_batch_segmented_artifact_parity_pass = (
        trackb_route_batch_segmented_artifact_parity is not None
        and trackb_route_batch_segmented_artifact_parity.get("decision")
        == "route_batch_segmented_codeword_reduce_air_artifact_parity_pass"
        and bool(
            trackb_route_batch_segmented_artifact_parity.get("passes_artifact_parity")
        )
    )
    trackb_route_batch_segmented_speed_rejected = (
        trackb_route_batch_segmented_speed_packet is not None
        and trackb_route_batch_segmented_speed_packet.get("decision")
        == "reject_route_batch_segmented_codeword_reduce_speed_path"
    )
    trackb_route_batch_segmented_speed_lane_s_pass = (
        trackb_route_batch_segmented_speed_packet is not None
        and trackb_route_batch_segmented_speed_packet.get("decision")
        == "route_batch_segmented_codeword_reduce_speed_packet_lane_s_pass"
        and bool(trackb_route_batch_segmented_speed_packet.get("all_lane_s_pass"))
    )
    trackb_component_stream_partial_reduction_design_ready = (
        trackb_component_stream_partial_reduction_design is not None
        and trackb_component_stream_partial_reduction_design.get("decision")
        == "component_stream_partial_reduction_ready_for_source_structure_probe"
        and bool(
            trackb_component_stream_partial_reduction_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_component_stream_partial_reduction_source_guard_missing = (
        trackb_component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guard.get("decision")
        == "missing_component_stream_partial_reduction_scaffold"
    )
    trackb_component_stream_partial_reduction_source_guard_present = (
        trackb_component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guard.get("decision")
        in {
            "component_stream_partial_reduction_scaffold_scalar_body",
            "component_stream_partial_reduction_parallel_body_present",
            "component_stream_partial_reduction_tensorops_body_present",
        }
        and bool(component_stream_partial_reduction_guard.get("passes_contract"))
    )
    trackb_component_stream_partial_reduction_parallel_present = (
        trackb_component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guard.get("decision")
        == "component_stream_partial_reduction_parallel_body_present"
    )
    trackb_component_stream_partial_reduction_tensorops_present = (
        trackb_component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guard.get("decision")
        == "component_stream_partial_reduction_tensorops_body_present"
    )
    trackb_component_stream_partial_reduction_native_parity_pass = (
        trackb_component_stream_partial_reduction_native_parity is not None
        and trackb_component_stream_partial_reduction_native_parity.get("decision")
        == "component_stream_partial_reduction_native_parity_pass"
        and bool(
            trackb_component_stream_partial_reduction_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    component_stream_tensorops_guard = (
        trackb_component_stream_tensorops_rejection.get(
            "component_stream_speed_path_guardrail", {}
        )
        if trackb_component_stream_tensorops_rejection is not None
        else {}
    )
    trackb_component_stream_tensorops_speed_rejected = (
        component_stream_tensorops_guard.get("decision")
        == "reject_component_stream_tensorops_speed_path"
    )
    trackb_next_family_gate_blocks = (
        trackb_next_family_gate is not None
        and trackb_next_family_gate.get("decision")
        == "block_stale_trackb_plan_variant_requires_materially_new_family_design"
        and bool(trackb_next_family_gate.get("stale_plan_variant_blocked"))
    )
    trackb_materially_new_family_gate_active = (
        trackb_next_family_gate_blocks
        and trackb_component_stream_tensorops_speed_rejected
    )
    trackb_token_cohort_codeword_stream_design_ready = (
        trackb_token_cohort_codeword_stream_design is not None
        and trackb_token_cohort_codeword_stream_design.get("decision")
        == "token_cohort_codeword_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_cohort_codeword_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_cohort_codeword_stream_source_guard_missing = (
        trackb_token_cohort_codeword_stream_design_ready
        and token_cohort_codeword_stream_guard.get("decision")
        == "missing_token_cohort_codeword_stream_source"
    )
    trackb_token_cohort_codeword_stream_source_guard_present = (
        trackb_token_cohort_codeword_stream_design_ready
        and token_cohort_codeword_stream_guard.get("decision")
        == "token_cohort_codeword_stream_source_guardrail_present"
        and bool(token_cohort_codeword_stream_guard.get("passes_contract"))
    )
    trackb_token_cohort_codeword_stream_native_parity_pass = (
        trackb_token_cohort_codeword_stream_native_parity is not None
        and trackb_token_cohort_codeword_stream_native_parity.get("decision")
        == "token_cohort_codeword_stream_native_parity_pass"
        and bool(
            trackb_token_cohort_codeword_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_cohort_codeword_stream_artifact_parity_pass = (
        trackb_token_cohort_codeword_stream_artifact_parity is not None
        and trackb_token_cohort_codeword_stream_artifact_parity.get("decision")
        == "token_cohort_codeword_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_cohort_codeword_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_cohort_codeword_stream_speed_rejected = (
        trackb_token_cohort_codeword_stream_speed_packet is not None
        and trackb_token_cohort_codeword_stream_speed_packet.get("decision")
        == "reject_token_cohort_codeword_stream_speed_path"
    )
    trackb_token_cohort_codeword_stream_speed_lane_s_pass = (
        trackb_token_cohort_codeword_stream_speed_packet is not None
        and trackb_token_cohort_codeword_stream_speed_packet.get("decision")
        == "token_cohort_codeword_stream_speed_packet_lane_s_pass"
        and bool(trackb_token_cohort_codeword_stream_speed_packet.get("all_lane_s_pass"))
    )
    trackb_token_cohort_mma_codeword_tile_design_ready = (
        trackb_token_cohort_mma_codeword_tile_design is not None
        and trackb_token_cohort_mma_codeword_tile_design.get("decision")
        == "token_cohort_mma_codeword_tile_ready_for_source_structure_probe"
        and bool(
            trackb_token_cohort_mma_codeword_tile_design.get("selector_verdict", {}).get(
                "candidate_ready_for_source_probe"
            )
        )
    )
    trackb_token_cohort_mma_codeword_tile_source_guard_missing = (
        trackb_token_cohort_mma_codeword_tile_design_ready
        and token_cohort_mma_codeword_tile_guard.get("decision")
        == "missing_token_cohort_mma_codeword_tile_source"
    )
    trackb_token_cohort_mma_codeword_tile_source_guard_present = (
        trackb_token_cohort_mma_codeword_tile_design_ready
        and token_cohort_mma_codeword_tile_guard.get("decision")
        == "token_cohort_mma_codeword_tile_source_guardrail_present"
        and bool(token_cohort_mma_codeword_tile_guard.get("passes_contract"))
    )
    trackb_token_cohort_mma_codeword_tile_native_parity_pass = (
        trackb_token_cohort_mma_codeword_tile_native_parity is not None
        and trackb_token_cohort_mma_codeword_tile_native_parity.get("decision")
        == "token_cohort_mma_codeword_tile_native_parity_pass"
        and bool(
            trackb_token_cohort_mma_codeword_tile_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_cohort_mma_codeword_tile_artifact_parity_pass = (
        trackb_token_cohort_mma_codeword_tile_artifact_parity is not None
        and trackb_token_cohort_mma_codeword_tile_artifact_parity.get("decision")
        == "token_cohort_mma_codeword_tile_air_artifact_parity_pass"
        and bool(
            trackb_token_cohort_mma_codeword_tile_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_cohort_mma_codeword_tile_speed_rejected = (
        trackb_token_cohort_mma_codeword_tile_speed_packet is not None
        and trackb_token_cohort_mma_codeword_tile_speed_packet.get("decision")
        == "reject_token_cohort_mma_codeword_tile_speed_path"
    )
    trackb_token_cohort_mma_codeword_tile_speed_lane_s_pass = (
        trackb_token_cohort_mma_codeword_tile_speed_packet is not None
        and trackb_token_cohort_mma_codeword_tile_speed_packet.get("decision")
        == "token_cohort_mma_codeword_tile_speed_packet_lane_s_pass"
        and bool(trackb_token_cohort_mma_codeword_tile_speed_packet.get("all_lane_s_pass"))
    )
    trackb_output_stationary_codeword_tile_design_ready = (
        trackb_output_stationary_codeword_tile_design is not None
        and trackb_output_stationary_codeword_tile_design.get("decision")
        == "output_stationary_codeword_tile_ready_for_source_structure_probe"
        and bool(
            trackb_output_stationary_codeword_tile_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_output_stationary_codeword_tile_source_guard_missing = (
        trackb_output_stationary_codeword_tile_design_ready
        and output_stationary_codeword_tile_guard.get("decision")
        == "missing_output_stationary_codeword_tile_source"
    )
    trackb_output_stationary_codeword_tile_source_guard_present = (
        trackb_output_stationary_codeword_tile_design_ready
        and output_stationary_codeword_tile_guard.get("decision")
        == "output_stationary_codeword_tile_source_guardrail_present"
        and bool(output_stationary_codeword_tile_guard.get("passes_contract"))
    )
    trackb_output_stationary_codeword_tile_native_parity_pass = (
        trackb_output_stationary_codeword_tile_native_parity is not None
        and trackb_output_stationary_codeword_tile_native_parity.get("decision")
        == "output_stationary_codeword_tile_native_parity_pass"
        and bool(
            trackb_output_stationary_codeword_tile_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_output_stationary_codeword_tile_artifact_parity_pass = (
        trackb_output_stationary_codeword_tile_artifact_parity is not None
        and trackb_output_stationary_codeword_tile_artifact_parity.get("decision")
        == "output_stationary_codeword_tile_air_artifact_parity_pass"
        and bool(
            trackb_output_stationary_codeword_tile_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_output_stationary_codeword_tile_speed_rejected = (
        trackb_output_stationary_codeword_tile_speed_packet is not None
        and trackb_output_stationary_codeword_tile_speed_packet.get("decision")
        == "reject_output_stationary_codeword_tile_speed_path"
    )
    trackb_output_stationary_codeword_tile_speed_lane_s_pass = (
        trackb_output_stationary_codeword_tile_speed_packet is not None
        and trackb_output_stationary_codeword_tile_speed_packet.get("decision")
        == "output_stationary_codeword_tile_speed_packet_lane_s_pass"
        and bool(trackb_output_stationary_codeword_tile_speed_packet.get("all_lane_s_pass"))
    )
    trackb_input_stationary_codeword_tile_design_ready = (
        trackb_input_stationary_codeword_tile_design is not None
        and trackb_output_stationary_codeword_tile_speed_rejected
        and trackb_input_stationary_codeword_tile_design.get("decision")
        == "input_stationary_codeword_tile_ready_for_source_structure_probe"
        and bool(
            trackb_input_stationary_codeword_tile_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_input_stationary_codeword_tile_source_guard_missing = (
        trackb_input_stationary_codeword_tile_design_ready
        and input_stationary_codeword_tile_guard.get("decision")
        == "missing_input_stationary_codeword_tile_source"
    )
    trackb_input_stationary_codeword_tile_source_guard_present = (
        trackb_input_stationary_codeword_tile_design_ready
        and input_stationary_codeword_tile_guard.get("decision")
        == "input_stationary_codeword_tile_source_guardrail_present"
        and bool(input_stationary_codeword_tile_guard.get("passes_contract"))
    )
    trackb_input_stationary_codeword_tile_native_parity_pass = (
        trackb_input_stationary_codeword_tile_native_parity is not None
        and trackb_input_stationary_codeword_tile_native_parity.get("decision")
        == "input_stationary_codeword_tile_native_parity_pass"
        and bool(
            trackb_input_stationary_codeword_tile_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_input_stationary_codeword_tile_artifact_parity_pass = (
        trackb_input_stationary_codeword_tile_artifact_parity is not None
        and trackb_input_stationary_codeword_tile_artifact_parity.get("decision")
        == "input_stationary_codeword_tile_air_artifact_parity_pass"
        and bool(
            trackb_input_stationary_codeword_tile_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_input_stationary_codeword_tile_speed_rejected = (
        trackb_input_stationary_codeword_tile_speed_packet is not None
        and trackb_input_stationary_codeword_tile_speed_packet.get("decision")
        == "reject_input_stationary_codeword_tile_speed_path"
    )
    trackb_input_stationary_codeword_tile_speed_lane_s_pass = (
        trackb_input_stationary_codeword_tile_speed_packet is not None
        and trackb_input_stationary_codeword_tile_speed_packet.get("decision")
        == "input_stationary_codeword_tile_speed_packet_lane_s_pass"
        and bool(trackb_input_stationary_codeword_tile_speed_packet.get("all_lane_s_pass"))
    )
    trackb_expert_kblock_codeword_factor_reuse_design_ready = (
        trackb_expert_kblock_codeword_factor_reuse_design is not None
        and trackb_input_stationary_codeword_tile_speed_rejected
        and trackb_expert_kblock_codeword_factor_reuse_design.get("decision")
        == "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
        and bool(
            trackb_expert_kblock_codeword_factor_reuse_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_expert_kblock_codeword_factor_reuse_source_guard_missing = (
        trackb_expert_kblock_codeword_factor_reuse_design_ready
        and expert_kblock_codeword_factor_reuse_guard.get("decision")
        == "missing_expert_kblock_codeword_factor_reuse_source"
    )
    trackb_expert_kblock_codeword_factor_reuse_source_guard_present = (
        trackb_expert_kblock_codeword_factor_reuse_design_ready
        and expert_kblock_codeword_factor_reuse_guard.get("decision")
        == "expert_kblock_codeword_factor_reuse_source_guardrail_present"
        and bool(expert_kblock_codeword_factor_reuse_guard.get("passes_contract"))
    )
    trackb_expert_kblock_codeword_factor_reuse_native_parity_pass = (
        trackb_expert_kblock_codeword_factor_reuse_native_parity is not None
        and trackb_expert_kblock_codeword_factor_reuse_native_parity.get("decision")
        == "expert_kblock_codeword_factor_reuse_native_parity_pass"
        and bool(
            trackb_expert_kblock_codeword_factor_reuse_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_expert_kblock_codeword_factor_reuse_artifact_parity_pass = (
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity is not None
        and trackb_expert_kblock_codeword_factor_reuse_artifact_parity.get("decision")
        == "expert_kblock_codeword_factor_reuse_air_artifact_parity_pass"
        and bool(
            trackb_expert_kblock_codeword_factor_reuse_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_expert_kblock_codeword_factor_reuse_speed_rejected = (
        trackb_expert_kblock_codeword_factor_reuse_speed_packet is not None
        and trackb_expert_kblock_codeword_factor_reuse_speed_packet.get("decision")
        == "reject_expert_kblock_codeword_factor_reuse_speed_path"
    )
    trackb_expert_kblock_codeword_factor_reuse_speed_lane_s_pass = (
        trackb_expert_kblock_codeword_factor_reuse_speed_packet is not None
        and trackb_expert_kblock_codeword_factor_reuse_speed_packet.get("decision")
        == "expert_kblock_codeword_factor_reuse_speed_packet_lane_s_pass"
        and bool(
            trackb_expert_kblock_codeword_factor_reuse_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_route_codeword_lut_accumulate_design_ready = (
        trackb_route_codeword_lut_accumulate_design is not None
        and trackb_expert_kblock_codeword_factor_reuse_speed_rejected
        and trackb_route_codeword_lut_accumulate_design.get("decision")
        == "route_codeword_lut_accumulate_ready_for_source_structure_probe"
        and bool(
            trackb_route_codeword_lut_accumulate_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_route_codeword_lut_accumulate_source_guard_missing = (
        trackb_route_codeword_lut_accumulate_design_ready
        and route_codeword_lut_accumulate_guard.get("decision")
        == "missing_route_codeword_lut_accumulate_source"
    )
    trackb_route_codeword_lut_accumulate_source_guard_present = (
        trackb_route_codeword_lut_accumulate_design_ready
        and route_codeword_lut_accumulate_guard.get("decision")
        == "route_codeword_lut_accumulate_source_guardrail_present"
        and bool(route_codeword_lut_accumulate_guard.get("passes_contract"))
    )
    trackb_route_codeword_lut_accumulate_native_parity_pass = (
        trackb_route_codeword_lut_accumulate_native_parity is not None
        and trackb_route_codeword_lut_accumulate_native_parity.get("decision")
        == "route_codeword_lut_accumulate_native_parity_pass"
        and bool(
            trackb_route_codeword_lut_accumulate_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_route_codeword_lut_accumulate_artifact_parity_pass = (
        trackb_route_codeword_lut_accumulate_artifact_parity is not None
        and trackb_route_codeword_lut_accumulate_artifact_parity.get("decision")
        == "route_codeword_lut_accumulate_air_artifact_parity_pass"
        and bool(
            trackb_route_codeword_lut_accumulate_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_route_codeword_lut_accumulate_speed_rejected = (
        trackb_route_codeword_lut_accumulate_speed_packet is not None
        and trackb_route_codeword_lut_accumulate_speed_packet.get("decision")
        == "reject_route_codeword_lut_accumulate_speed_path"
        and bool(
            trackb_route_codeword_lut_accumulate_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_rowwise_codeword_tile_accumulate_design_ready = (
        trackb_rowwise_codeword_tile_accumulate_design is not None
        and trackb_route_codeword_lut_accumulate_speed_rejected
        and trackb_rowwise_codeword_tile_accumulate_design.get("decision")
        == "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
        and bool(
            trackb_rowwise_codeword_tile_accumulate_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_rowwise_codeword_tile_accumulate_source_guard_missing = (
        trackb_rowwise_codeword_tile_accumulate_design_ready
        and rowwise_codeword_tile_accumulate_guard.get("decision")
        == "missing_rowwise_codeword_tile_accumulate_source"
    )
    trackb_rowwise_codeword_tile_accumulate_source_guard_present = (
        trackb_rowwise_codeword_tile_accumulate_design_ready
        and rowwise_codeword_tile_accumulate_guard.get("decision")
        == "rowwise_codeword_tile_accumulate_source_guardrail_present"
        and bool(rowwise_codeword_tile_accumulate_guard.get("passes_contract"))
    )
    trackb_rowwise_codeword_tile_accumulate_native_parity_pass = (
        trackb_rowwise_codeword_tile_accumulate_native_parity is not None
        and trackb_rowwise_codeword_tile_accumulate_native_parity.get("decision")
        == "rowwise_codeword_tile_accumulate_native_parity_pass"
        and bool(
            trackb_rowwise_codeword_tile_accumulate_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_rowwise_codeword_tile_accumulate_artifact_parity_pass = (
        trackb_rowwise_codeword_tile_accumulate_artifact_parity is not None
        and trackb_rowwise_codeword_tile_accumulate_artifact_parity.get("decision")
        == "rowwise_codeword_tile_accumulate_air_artifact_parity_pass"
        and bool(
            trackb_rowwise_codeword_tile_accumulate_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_rowwise_codeword_tile_accumulate_speed_rejected = (
        trackb_rowwise_codeword_tile_accumulate_speed_packet is not None
        and trackb_rowwise_codeword_tile_accumulate_speed_packet.get("decision")
        == "reject_rowwise_codeword_tile_accumulate_speed_path"
        and bool(
            trackb_rowwise_codeword_tile_accumulate_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_rowwise_codeword_tile_accumulate_speed_lane_s_pass = (
        trackb_rowwise_codeword_tile_accumulate_speed_packet is not None
        and trackb_rowwise_codeword_tile_accumulate_speed_packet.get("decision")
        == "rowwise_codeword_tile_accumulate_speed_packet_lane_s_pass"
        and bool(
            trackb_rowwise_codeword_tile_accumulate_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_output_tile_local_codeword_lut_design_ready = (
        trackb_output_tile_local_codeword_lut_design is not None
        and trackb_rowwise_codeword_tile_accumulate_speed_rejected
        and trackb_output_tile_local_codeword_lut_design.get("decision")
        == "output_tile_local_codeword_lut_ready_for_source_structure_probe"
        and bool(
            trackb_output_tile_local_codeword_lut_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_output_tile_local_codeword_lut_source_guard_missing = (
        trackb_output_tile_local_codeword_lut_design_ready
        and output_tile_local_codeword_lut_guard.get("decision")
        == "missing_output_tile_local_codeword_lut_source"
    )
    trackb_output_tile_local_codeword_lut_source_guard_present = (
        trackb_output_tile_local_codeword_lut_design_ready
        and output_tile_local_codeword_lut_guard.get("decision")
        == "output_tile_local_codeword_lut_source_guardrail_present"
        and bool(output_tile_local_codeword_lut_guard.get("passes_contract"))
    )
    trackb_output_tile_local_codeword_lut_native_parity_pass = (
        trackb_output_tile_local_codeword_lut_native_parity is not None
        and trackb_output_tile_local_codeword_lut_native_parity.get("decision")
        == "output_tile_local_codeword_lut_native_parity_pass"
        and bool(
            trackb_output_tile_local_codeword_lut_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_output_tile_local_codeword_lut_artifact_parity_pass = (
        trackb_output_tile_local_codeword_lut_artifact_parity is not None
        and trackb_output_tile_local_codeword_lut_artifact_parity.get("decision")
        == "output_tile_local_codeword_lut_air_artifact_parity_pass"
        and bool(
            trackb_output_tile_local_codeword_lut_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_output_tile_local_codeword_lut_speed_rejected = (
        trackb_output_tile_local_codeword_lut_speed_packet is not None
        and trackb_output_tile_local_codeword_lut_speed_packet.get("decision")
        == "reject_output_tile_local_codeword_lut_speed_path"
        and bool(
            trackb_output_tile_local_codeword_lut_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_output_tile_local_codeword_lut_speed_lane_s_pass = (
        trackb_output_tile_local_codeword_lut_speed_packet is not None
        and trackb_output_tile_local_codeword_lut_speed_packet.get("decision")
        == "output_tile_local_codeword_lut_speed_packet_lane_s_pass"
        and bool(trackb_output_tile_local_codeword_lut_speed_packet.get("all_lane_s_pass"))
    )
    trackb_route_microtile_codeword_block_reduce_design_ready = (
        trackb_route_microtile_codeword_block_reduce_design is not None
        and trackb_output_tile_local_codeword_lut_speed_rejected
        and trackb_route_microtile_codeword_block_reduce_design.get("decision")
        == "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
        and bool(
            trackb_route_microtile_codeword_block_reduce_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_route_microtile_codeword_block_reduce_source_guard_missing = (
        trackb_route_microtile_codeword_block_reduce_design_ready
        and route_microtile_codeword_block_reduce_guard.get("decision")
        == "missing_route_microtile_codeword_block_reduce_source"
    )
    trackb_route_microtile_codeword_block_reduce_source_guard_present = (
        trackb_route_microtile_codeword_block_reduce_design_ready
        and route_microtile_codeword_block_reduce_guard.get("decision")
        == "route_microtile_codeword_block_reduce_source_guardrail_present"
        and bool(route_microtile_codeword_block_reduce_guard.get("passes_contract"))
    )
    trackb_route_microtile_codeword_block_reduce_native_parity_pass = (
        trackb_route_microtile_codeword_block_reduce_source_guard_present
        and trackb_route_microtile_codeword_block_reduce_native_parity is not None
        and trackb_route_microtile_codeword_block_reduce_native_parity.get("decision")
        == "route_microtile_codeword_block_reduce_native_parity_pass"
        and bool(
            trackb_route_microtile_codeword_block_reduce_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_route_microtile_codeword_block_reduce_artifact_parity_pass = (
        trackb_route_microtile_codeword_block_reduce_native_parity_pass
        and trackb_route_microtile_codeword_block_reduce_artifact_parity is not None
        and trackb_route_microtile_codeword_block_reduce_artifact_parity.get("decision")
        == "route_microtile_codeword_block_reduce_air_artifact_parity_pass"
        and bool(
            trackb_route_microtile_codeword_block_reduce_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_route_microtile_codeword_block_reduce_speed_rejected = (
        trackb_route_microtile_codeword_block_reduce_artifact_parity_pass
        and trackb_route_microtile_codeword_block_reduce_speed_packet is not None
        and trackb_route_microtile_codeword_block_reduce_speed_packet.get("decision")
        == "reject_route_microtile_codeword_block_reduce_speed_path"
        and bool(
            trackb_route_microtile_codeword_block_reduce_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_route_microtile_codeword_block_reduce_speed_lane_s_pass = (
        trackb_route_microtile_codeword_block_reduce_artifact_parity_pass
        and trackb_route_microtile_codeword_block_reduce_speed_packet is not None
        and trackb_route_microtile_codeword_block_reduce_speed_packet.get("decision")
        == "route_microtile_codeword_block_reduce_speed_packet_lane_s_pass"
        and bool(
            trackb_route_microtile_codeword_block_reduce_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_kblock_wavefront_codeword_scan_design_ready = (
        trackb_kblock_wavefront_codeword_scan_design is not None
        and trackb_route_microtile_codeword_block_reduce_speed_rejected
        and trackb_kblock_wavefront_codeword_scan_design.get("decision")
        == "kblock_wavefront_codeword_scan_ready_for_source_structure_probe"
        and bool(
            trackb_kblock_wavefront_codeword_scan_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_kblock_wavefront_codeword_scan_source_guard_missing = (
        trackb_kblock_wavefront_codeword_scan_design_ready
        and kblock_wavefront_codeword_scan_guard.get("decision")
        == "missing_kblock_wavefront_codeword_scan_source"
    )
    trackb_kblock_wavefront_codeword_scan_source_guard_present = (
        trackb_kblock_wavefront_codeword_scan_design_ready
        and kblock_wavefront_codeword_scan_guard.get("decision")
        == "kblock_wavefront_codeword_scan_source_guardrail_present"
        and bool(kblock_wavefront_codeword_scan_guard.get("passes_contract"))
    )
    trackb_kblock_wavefront_codeword_scan_native_parity_pass = (
        trackb_kblock_wavefront_codeword_scan_source_guard_present
        and trackb_kblock_wavefront_codeword_scan_native_parity is not None
        and trackb_kblock_wavefront_codeword_scan_native_parity.get("decision")
        == "kblock_wavefront_codeword_scan_native_parity_pass"
        and bool(
            trackb_kblock_wavefront_codeword_scan_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_kblock_wavefront_codeword_scan_artifact_parity_pass = (
        trackb_kblock_wavefront_codeword_scan_native_parity_pass
        and trackb_kblock_wavefront_codeword_scan_artifact_parity is not None
        and trackb_kblock_wavefront_codeword_scan_artifact_parity.get("decision")
        == "kblock_wavefront_codeword_scan_air_artifact_parity_pass"
        and bool(
            trackb_kblock_wavefront_codeword_scan_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_kblock_wavefront_codeword_scan_speed_rejected = (
        trackb_kblock_wavefront_codeword_scan_artifact_parity_pass
        and trackb_kblock_wavefront_codeword_scan_speed_packet is not None
        and trackb_kblock_wavefront_codeword_scan_speed_packet.get("decision")
        == "reject_kblock_wavefront_codeword_scan_speed_path"
        and bool(
            trackb_kblock_wavefront_codeword_scan_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_kblock_wavefront_codeword_scan_speed_lane_s_pass = (
        trackb_kblock_wavefront_codeword_scan_artifact_parity_pass
        and trackb_kblock_wavefront_codeword_scan_speed_packet is not None
        and trackb_kblock_wavefront_codeword_scan_speed_packet.get("decision")
        == "kblock_wavefront_codeword_scan_speed_packet_lane_s_pass"
        and bool(
            trackb_kblock_wavefront_codeword_scan_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_route_output_stripe_pipeline_design_ready = (
        trackb_token_route_output_stripe_pipeline_design is not None
        and trackb_kblock_wavefront_codeword_scan_speed_rejected
        and trackb_token_route_output_stripe_pipeline_design.get("decision")
        == "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
        and bool(
            trackb_token_route_output_stripe_pipeline_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_route_output_stripe_pipeline_source_guard_missing = (
        trackb_token_route_output_stripe_pipeline_design_ready
        and token_route_output_stripe_pipeline_guard.get("decision")
        == "missing_token_route_output_stripe_pipeline_source"
    )
    trackb_token_route_output_stripe_pipeline_source_guard_present = (
        trackb_token_route_output_stripe_pipeline_design_ready
        and token_route_output_stripe_pipeline_guard.get("decision")
        == "token_route_output_stripe_pipeline_source_guardrail_present"
        and bool(token_route_output_stripe_pipeline_guard.get("passes_contract"))
    )
    trackb_token_route_output_stripe_pipeline_native_parity_pass = (
        trackb_token_route_output_stripe_pipeline_source_guard_present
        and trackb_token_route_output_stripe_pipeline_native_parity is not None
        and trackb_token_route_output_stripe_pipeline_native_parity.get("decision")
        == "token_route_output_stripe_pipeline_native_parity_pass"
        and bool(
            trackb_token_route_output_stripe_pipeline_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_route_output_stripe_pipeline_artifact_parity_pass = (
        trackb_token_route_output_stripe_pipeline_native_parity_pass
        and trackb_token_route_output_stripe_pipeline_artifact_parity is not None
        and trackb_token_route_output_stripe_pipeline_artifact_parity.get("decision")
        == "token_route_output_stripe_pipeline_air_artifact_parity_pass"
        and bool(
            trackb_token_route_output_stripe_pipeline_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_route_output_stripe_pipeline_speed_rejected = (
        trackb_token_route_output_stripe_pipeline_artifact_parity_pass
        and trackb_token_route_output_stripe_pipeline_speed_packet is not None
        and trackb_token_route_output_stripe_pipeline_speed_packet.get("decision")
        == "reject_token_route_output_stripe_pipeline_speed_path"
        and bool(
            trackb_token_route_output_stripe_pipeline_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_route_output_stripe_pipeline_speed_lane_s_pass = (
        trackb_token_route_output_stripe_pipeline_artifact_parity_pass
        and trackb_token_route_output_stripe_pipeline_speed_packet is not None
        and trackb_token_route_output_stripe_pipeline_speed_packet.get("decision")
        == "token_route_output_stripe_pipeline_speed_packet_lane_s_pass"
        and bool(
            trackb_token_route_output_stripe_pipeline_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_route_output_stripe_pipeline_speed_parity_pass = (
        trackb_token_route_output_stripe_pipeline_artifact_parity_pass
        and trackb_token_route_output_stripe_pipeline_speed_packet is not None
        and trackb_token_route_output_stripe_pipeline_speed_packet.get("decision")
        == "token_route_output_stripe_pipeline_speed_packet_parity_pass"
        and bool(
            trackb_token_route_output_stripe_pipeline_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_expert_kblock_scale_slot_stream_design_ready = (
        trackb_expert_kblock_scale_slot_stream_design is not None
        and trackb_token_route_output_stripe_pipeline_speed_rejected
        and trackb_expert_kblock_scale_slot_stream_design.get("decision")
        == "expert_kblock_scale_slot_stream_ready_for_source_structure_probe"
        and bool(
            trackb_expert_kblock_scale_slot_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_expert_kblock_scale_slot_stream_source_guard_missing = (
        trackb_expert_kblock_scale_slot_stream_design_ready
        and expert_kblock_scale_slot_stream_guard.get("decision")
        == "missing_expert_kblock_scale_slot_stream_source"
    )
    trackb_expert_kblock_scale_slot_stream_source_guard_present = (
        trackb_expert_kblock_scale_slot_stream_design_ready
        and expert_kblock_scale_slot_stream_guard.get("decision")
        == "expert_kblock_scale_slot_stream_source_guardrail_present"
        and bool(expert_kblock_scale_slot_stream_guard.get("passes_contract"))
    )
    trackb_expert_kblock_scale_slot_stream_native_parity_pass = (
        trackb_expert_kblock_scale_slot_stream_source_guard_present
        and trackb_expert_kblock_scale_slot_stream_native_parity is not None
        and trackb_expert_kblock_scale_slot_stream_native_parity.get("decision")
        == "expert_kblock_scale_slot_stream_native_parity_pass"
        and bool(
            trackb_expert_kblock_scale_slot_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_expert_kblock_scale_slot_stream_artifact_parity_pass = (
        trackb_expert_kblock_scale_slot_stream_native_parity_pass
        and trackb_expert_kblock_scale_slot_stream_artifact_parity is not None
        and trackb_expert_kblock_scale_slot_stream_artifact_parity.get("decision")
        == "expert_kblock_scale_slot_stream_air_artifact_parity_pass"
        and bool(
            trackb_expert_kblock_scale_slot_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_expert_kblock_scale_slot_stream_speed_rejected = (
        trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
        and trackb_expert_kblock_scale_slot_stream_speed_packet is not None
        and trackb_expert_kblock_scale_slot_stream_speed_packet.get("decision")
        == "reject_expert_kblock_scale_slot_stream_speed_path"
        and bool(
            trackb_expert_kblock_scale_slot_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_expert_kblock_scale_slot_stream_speed_lane_s_pass = (
        trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
        and trackb_expert_kblock_scale_slot_stream_speed_packet is not None
        and trackb_expert_kblock_scale_slot_stream_speed_packet.get("decision")
        == "expert_kblock_scale_slot_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_expert_kblock_scale_slot_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_expert_kblock_scale_slot_stream_speed_parity_pass = (
        trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
        and trackb_expert_kblock_scale_slot_stream_speed_packet is not None
        and trackb_expert_kblock_scale_slot_stream_speed_packet.get("decision")
        == "expert_kblock_scale_slot_stream_speed_packet_parity_pass"
        and bool(
            trackb_expert_kblock_scale_slot_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_scale_group_route_block_reduce_design_ready = (
        trackb_scale_group_route_block_reduce_design is not None
        and trackb_expert_kblock_scale_slot_stream_speed_rejected
        and trackb_scale_group_route_block_reduce_design.get("decision")
        == "scale_group_route_block_reduce_ready_for_source_structure_probe"
        and bool(
            trackb_scale_group_route_block_reduce_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_scale_group_route_block_reduce_source_guard_missing = (
        trackb_scale_group_route_block_reduce_design_ready
        and scale_group_route_block_reduce_guard.get("decision")
        == "missing_scale_group_route_block_reduce_source"
    )
    trackb_scale_group_route_block_reduce_source_guard_present = (
        trackb_scale_group_route_block_reduce_design_ready
        and scale_group_route_block_reduce_guard.get("decision")
        == "scale_group_route_block_reduce_source_guardrail_present"
        and bool(scale_group_route_block_reduce_guard.get("passes_contract"))
    )
    trackb_scale_group_route_block_reduce_native_parity_pass = (
        trackb_scale_group_route_block_reduce_source_guard_present
        and trackb_scale_group_route_block_reduce_native_parity is not None
        and trackb_scale_group_route_block_reduce_native_parity.get("decision")
        == "scale_group_route_block_reduce_native_parity_pass"
        and bool(
            trackb_scale_group_route_block_reduce_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_scale_group_route_block_reduce_artifact_parity_pass = (
        trackb_scale_group_route_block_reduce_native_parity_pass
        and trackb_scale_group_route_block_reduce_artifact_parity is not None
        and trackb_scale_group_route_block_reduce_artifact_parity.get("decision")
        == "scale_group_route_block_reduce_air_artifact_parity_pass"
        and bool(
            trackb_scale_group_route_block_reduce_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_scale_group_route_block_reduce_speed_rejected = (
        trackb_scale_group_route_block_reduce_artifact_parity_pass
        and trackb_scale_group_route_block_reduce_speed_packet is not None
        and trackb_scale_group_route_block_reduce_speed_packet.get("decision")
        == "reject_scale_group_route_block_reduce_speed_path"
        and bool(
            trackb_scale_group_route_block_reduce_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_scale_group_route_block_reduce_speed_lane_s_pass = (
        trackb_scale_group_route_block_reduce_artifact_parity_pass
        and trackb_scale_group_route_block_reduce_speed_packet is not None
        and trackb_scale_group_route_block_reduce_speed_packet.get("decision")
        == "scale_group_route_block_reduce_speed_packet_lane_s_pass"
        and bool(
            trackb_scale_group_route_block_reduce_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_scale_group_route_block_reduce_speed_parity_pass = (
        trackb_scale_group_route_block_reduce_artifact_parity_pass
        and trackb_scale_group_route_block_reduce_speed_packet is not None
        and trackb_scale_group_route_block_reduce_speed_packet.get("decision")
        == "scale_group_route_block_reduce_speed_packet_parity_pass"
        and bool(
            trackb_scale_group_route_block_reduce_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_route_block_output_group_stream_design_ready = (
        trackb_route_block_output_group_stream_design is not None
        and trackb_scale_group_route_block_reduce_speed_rejected
        and trackb_route_block_output_group_stream_design.get("decision")
        == "route_block_output_group_stream_ready_for_source_structure_probe"
        and bool(
            trackb_route_block_output_group_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_route_block_output_group_stream_source_guard_missing = (
        trackb_route_block_output_group_stream_design_ready
        and route_block_output_group_stream_guard.get("decision")
        == "missing_route_block_output_group_stream_source"
    )
    trackb_route_block_output_group_stream_source_guard_present = (
        trackb_route_block_output_group_stream_design_ready
        and route_block_output_group_stream_guard.get("decision")
        == "route_block_output_group_stream_source_guardrail_present"
        and bool(route_block_output_group_stream_guard.get("passes_contract"))
    )
    trackb_route_block_output_group_stream_native_parity_pass = (
        trackb_route_block_output_group_stream_source_guard_present
        and trackb_route_block_output_group_stream_native_parity is not None
        and trackb_route_block_output_group_stream_native_parity.get("decision")
        == "route_block_output_group_stream_native_parity_pass"
        and bool(
            trackb_route_block_output_group_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_route_block_output_group_stream_artifact_parity_pass = (
        trackb_route_block_output_group_stream_native_parity_pass
        and trackb_route_block_output_group_stream_artifact_parity is not None
        and trackb_route_block_output_group_stream_artifact_parity.get("decision")
        == "route_block_output_group_stream_air_artifact_parity_pass"
        and bool(
            trackb_route_block_output_group_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_route_block_output_group_stream_speed_rejected = (
        trackb_route_block_output_group_stream_artifact_parity_pass
        and trackb_route_block_output_group_stream_speed_packet is not None
        and trackb_route_block_output_group_stream_speed_packet.get("decision")
        == "reject_route_block_output_group_stream_speed_path"
        and bool(
            trackb_route_block_output_group_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_route_block_output_group_stream_speed_lane_s_pass = (
        trackb_route_block_output_group_stream_artifact_parity_pass
        and trackb_route_block_output_group_stream_speed_packet is not None
        and trackb_route_block_output_group_stream_speed_packet.get("decision")
        == "route_block_output_group_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_route_block_output_group_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_route_block_output_group_stream_speed_parity_pass = (
        trackb_route_block_output_group_stream_artifact_parity_pass
        and trackb_route_block_output_group_stream_speed_packet is not None
        and trackb_route_block_output_group_stream_speed_packet.get("decision")
        == "route_block_output_group_stream_speed_packet_parity_pass"
        and bool(
            trackb_route_block_output_group_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_output_group_pretransposed_codeword_stream_design_ready = (
        trackb_route_block_output_group_stream_speed_rejected
        and trackb_output_group_pretransposed_codeword_stream_design is not None
        and trackb_output_group_pretransposed_codeword_stream_design.get("decision")
        == "output_group_pretransposed_codeword_stream_ready_for_source_structure_probe"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_output_group_pretransposed_codeword_stream_source_guard_missing = (
        trackb_output_group_pretransposed_codeword_stream_design_ready
        and output_group_pretransposed_codeword_stream_guard.get("decision")
        == "missing_output_group_pretransposed_codeword_stream_source"
    )
    trackb_output_group_pretransposed_codeword_stream_source_guard_present = (
        trackb_output_group_pretransposed_codeword_stream_design_ready
        and output_group_pretransposed_codeword_stream_guard.get("decision")
        == "output_group_pretransposed_codeword_stream_source_guardrail_present"
        and bool(output_group_pretransposed_codeword_stream_guard.get("passes_contract"))
    )
    trackb_output_group_pretransposed_codeword_stream_native_parity_pass = (
        trackb_output_group_pretransposed_codeword_stream_source_guard_present
        and trackb_output_group_pretransposed_codeword_stream_native_parity is not None
        and trackb_output_group_pretransposed_codeword_stream_native_parity.get(
            "decision"
        )
        == "output_group_pretransposed_codeword_stream_native_parity_pass"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass = (
        trackb_output_group_pretransposed_codeword_stream_native_parity_pass
        and trackb_output_group_pretransposed_codeword_stream_artifact_parity
        is not None
        and trackb_output_group_pretransposed_codeword_stream_artifact_parity.get(
            "decision"
        )
        == "output_group_pretransposed_codeword_stream_air_artifact_parity_pass"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_output_group_pretransposed_codeword_stream_speed_rejected = (
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
        and trackb_output_group_pretransposed_codeword_stream_speed_packet is not None
        and trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
            "decision"
        )
        == "reject_output_group_pretransposed_codeword_stream_speed_path"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_output_group_pretransposed_codeword_stream_speed_lane_s_pass = (
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
        and trackb_output_group_pretransposed_codeword_stream_speed_packet is not None
        and trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
            "decision"
        )
        == "output_group_pretransposed_codeword_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_output_group_pretransposed_codeword_stream_speed_parity_pass = (
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
        and trackb_output_group_pretransposed_codeword_stream_speed_packet is not None
        and trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
            "decision"
        )
        == "output_group_pretransposed_codeword_stream_speed_packet_parity_pass"
        and bool(
            trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_kblock_output_group_route_fused_stream_design_ready = (
        trackb_output_group_pretransposed_codeword_stream_speed_rejected
        and trackb_kblock_output_group_route_fused_stream_design is not None
        and trackb_kblock_output_group_route_fused_stream_design.get("decision")
        == "kblock_output_group_route_fused_stream_ready_for_source_structure_probe"
        and bool(
            trackb_kblock_output_group_route_fused_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_kblock_output_group_route_fused_stream_source_guard_missing = (
        trackb_kblock_output_group_route_fused_stream_design_ready
        and kblock_output_group_route_fused_stream_guard.get("decision")
        == "missing_kblock_output_group_route_fused_stream_source"
    )
    trackb_kblock_output_group_route_fused_stream_source_guard_present = (
        trackb_kblock_output_group_route_fused_stream_design_ready
        and kblock_output_group_route_fused_stream_guard.get("decision")
        == "kblock_output_group_route_fused_stream_source_guardrail_present"
        and bool(kblock_output_group_route_fused_stream_guard.get("passes_contract"))
    )
    trackb_kblock_output_group_route_fused_stream_native_parity_pass = (
        trackb_kblock_output_group_route_fused_stream_source_guard_present
        and trackb_kblock_output_group_route_fused_stream_native_parity is not None
        and trackb_kblock_output_group_route_fused_stream_native_parity.get("decision")
        == "kblock_output_group_route_fused_stream_native_parity_pass"
        and bool(
            trackb_kblock_output_group_route_fused_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_kblock_output_group_route_fused_stream_artifact_parity_pass = (
        trackb_kblock_output_group_route_fused_stream_native_parity_pass
        and trackb_kblock_output_group_route_fused_stream_artifact_parity is not None
        and trackb_kblock_output_group_route_fused_stream_artifact_parity.get(
            "decision"
        )
        == "kblock_output_group_route_fused_stream_air_artifact_parity_pass"
        and bool(
            trackb_kblock_output_group_route_fused_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_kblock_output_group_route_fused_stream_speed_rejected = (
        trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
        and trackb_kblock_output_group_route_fused_stream_speed_packet is not None
        and trackb_kblock_output_group_route_fused_stream_speed_packet.get("decision")
        == "reject_kblock_output_group_route_fused_stream_speed_path"
        and bool(
            trackb_kblock_output_group_route_fused_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_kblock_output_group_route_fused_stream_speed_lane_s_pass = (
        trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
        and trackb_kblock_output_group_route_fused_stream_speed_packet is not None
        and trackb_kblock_output_group_route_fused_stream_speed_packet.get("decision")
        == "kblock_output_group_route_fused_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_kblock_output_group_route_fused_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_kblock_output_group_route_fused_stream_speed_parity_pass = (
        trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
        and trackb_kblock_output_group_route_fused_stream_speed_packet is not None
        and trackb_kblock_output_group_route_fused_stream_speed_packet.get("decision")
        == "kblock_output_group_route_fused_stream_speed_packet_parity_pass"
        and bool(
            trackb_kblock_output_group_route_fused_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_route_tile_output_swizzle_stream_design_ready = (
        trackb_kblock_output_group_route_fused_stream_speed_rejected
        and trackb_route_tile_output_swizzle_stream_design is not None
        and trackb_route_tile_output_swizzle_stream_design.get("decision")
        == "route_tile_output_swizzle_stream_ready_for_source_structure_probe"
        and bool(
            trackb_route_tile_output_swizzle_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_route_tile_output_swizzle_stream_source_guard_missing = (
        trackb_route_tile_output_swizzle_stream_design_ready
        and route_tile_output_swizzle_stream_guard.get("decision")
        == "missing_route_tile_output_swizzle_stream_source"
    )
    trackb_route_tile_output_swizzle_stream_source_guard_present = (
        trackb_route_tile_output_swizzle_stream_design_ready
        and route_tile_output_swizzle_stream_guard.get("decision")
        == "route_tile_output_swizzle_stream_source_guardrail_present"
        and bool(route_tile_output_swizzle_stream_guard.get("passes_contract"))
    )
    trackb_route_tile_output_swizzle_stream_native_parity_pass = (
        trackb_route_tile_output_swizzle_stream_source_guard_present
        and trackb_route_tile_output_swizzle_stream_native_parity is not None
        and trackb_route_tile_output_swizzle_stream_native_parity.get("decision")
        == "route_tile_output_swizzle_stream_native_parity_pass"
        and bool(
            trackb_route_tile_output_swizzle_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_route_tile_output_swizzle_stream_artifact_parity_pass = (
        trackb_route_tile_output_swizzle_stream_native_parity_pass
        and trackb_route_tile_output_swizzle_stream_artifact_parity is not None
        and trackb_route_tile_output_swizzle_stream_artifact_parity.get("decision")
        == "route_tile_output_swizzle_stream_air_artifact_parity_pass"
        and bool(
            trackb_route_tile_output_swizzle_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_route_tile_output_swizzle_stream_speed_rejected = (
        trackb_route_tile_output_swizzle_stream_artifact_parity_pass
        and trackb_route_tile_output_swizzle_stream_speed_packet is not None
        and trackb_route_tile_output_swizzle_stream_speed_packet.get("decision")
        == "reject_route_tile_output_swizzle_stream_speed_path"
        and bool(
            trackb_route_tile_output_swizzle_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_route_tile_output_swizzle_stream_speed_lane_s_pass = (
        trackb_route_tile_output_swizzle_stream_artifact_parity_pass
        and trackb_route_tile_output_swizzle_stream_speed_packet is not None
        and trackb_route_tile_output_swizzle_stream_speed_packet.get("decision")
        == "route_tile_output_swizzle_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_route_tile_output_swizzle_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_route_tile_output_swizzle_stream_speed_parity_pass = (
        trackb_route_tile_output_swizzle_stream_artifact_parity_pass
        and trackb_route_tile_output_swizzle_stream_speed_packet is not None
        and trackb_route_tile_output_swizzle_stream_speed_packet.get("decision")
        == "route_tile_output_swizzle_stream_speed_packet_parity_pass"
        and bool(
            trackb_route_tile_output_swizzle_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_topk_output_tile_stream_design_ready = (
        trackb_route_tile_output_swizzle_stream_speed_rejected
        and trackb_token_topk_output_tile_stream_design is not None
        and trackb_token_topk_output_tile_stream_design.get("decision")
        == "token_topk_output_tile_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_topk_output_tile_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_topk_output_tile_stream_source_guard_missing = (
        trackb_token_topk_output_tile_stream_design_ready
        and token_topk_output_tile_stream_guard.get("decision")
        == "missing_token_topk_output_tile_stream_source"
    )
    trackb_token_topk_output_tile_stream_source_guard_present = (
        trackb_token_topk_output_tile_stream_design_ready
        and token_topk_output_tile_stream_guard.get("decision")
        == "token_topk_output_tile_stream_source_guardrail_present"
        and bool(token_topk_output_tile_stream_guard.get("passes_contract"))
    )
    trackb_token_topk_output_tile_stream_native_parity_pass = (
        trackb_token_topk_output_tile_stream_source_guard_present
        and trackb_token_topk_output_tile_stream_native_parity is not None
        and trackb_token_topk_output_tile_stream_native_parity.get("decision")
        == "token_topk_output_tile_stream_native_parity_pass"
        and bool(
            trackb_token_topk_output_tile_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_topk_output_tile_stream_artifact_parity_pass = (
        trackb_token_topk_output_tile_stream_native_parity_pass
        and trackb_token_topk_output_tile_stream_artifact_parity is not None
        and trackb_token_topk_output_tile_stream_artifact_parity.get("decision")
        == "token_topk_output_tile_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_topk_output_tile_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_topk_output_tile_stream_speed_rejected = (
        trackb_token_topk_output_tile_stream_artifact_parity_pass
        and trackb_token_topk_output_tile_stream_speed_packet is not None
        and trackb_token_topk_output_tile_stream_speed_packet.get("decision")
        == "reject_token_topk_output_tile_stream_speed_path"
        and bool(
            trackb_token_topk_output_tile_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_topk_output_tile_stream_speed_lane_s_pass = (
        trackb_token_topk_output_tile_stream_artifact_parity_pass
        and trackb_token_topk_output_tile_stream_speed_packet is not None
        and trackb_token_topk_output_tile_stream_speed_packet.get("decision")
        == "token_topk_output_tile_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_topk_output_tile_stream_speed_packet.get("all_lane_s_pass")
        )
    )
    trackb_token_topk_output_tile_stream_speed_parity_pass = (
        trackb_token_topk_output_tile_stream_artifact_parity_pass
        and trackb_token_topk_output_tile_stream_speed_packet is not None
        and trackb_token_topk_output_tile_stream_speed_packet.get("decision")
        == "token_topk_output_tile_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_topk_output_tile_stream_speed_packet.get("all_parity_pass")
        )
    )
    trackb_token_block_output_group_stream_design_ready = (
        trackb_token_topk_output_tile_stream_speed_rejected
        and trackb_token_block_output_group_stream_design is not None
        and trackb_token_block_output_group_stream_design.get("decision")
        == "token_block_output_group_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_block_output_group_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_block_output_group_stream_source_guard_missing = (
        trackb_token_block_output_group_stream_design_ready
        and token_block_output_group_stream_guard.get("decision")
        == "missing_token_block_output_group_stream_source"
    )
    trackb_token_block_output_group_stream_source_guard_present = (
        trackb_token_block_output_group_stream_design_ready
        and token_block_output_group_stream_guard.get("decision")
        == "token_block_output_group_stream_source_guardrail_present"
        and bool(token_block_output_group_stream_guard.get("passes_contract"))
    )
    trackb_token_block_output_group_stream_native_parity_pass = (
        trackb_token_block_output_group_stream_source_guard_present
        and trackb_token_block_output_group_stream_native_parity is not None
        and trackb_token_block_output_group_stream_native_parity.get("decision")
        == "token_block_output_group_stream_native_parity_pass"
        and bool(
            trackb_token_block_output_group_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_block_output_group_stream_artifact_parity_pass = (
        trackb_token_block_output_group_stream_native_parity_pass
        and trackb_token_block_output_group_stream_artifact_parity is not None
        and trackb_token_block_output_group_stream_artifact_parity.get("decision")
        == "token_block_output_group_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_block_output_group_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_block_output_group_stream_speed_rejected = (
        trackb_token_block_output_group_stream_artifact_parity_pass
        and trackb_token_block_output_group_stream_speed_packet is not None
        and trackb_token_block_output_group_stream_speed_packet.get("decision")
        == "reject_token_block_output_group_stream_speed_path"
        and bool(
            trackb_token_block_output_group_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_block_output_group_stream_speed_lane_s_pass = (
        trackb_token_block_output_group_stream_artifact_parity_pass
        and trackb_token_block_output_group_stream_speed_packet is not None
        and trackb_token_block_output_group_stream_speed_packet.get("decision")
        == "token_block_output_group_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_block_output_group_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_block_output_group_stream_speed_parity_pass = (
        trackb_token_block_output_group_stream_artifact_parity_pass
        and trackb_token_block_output_group_stream_speed_packet is not None
        and trackb_token_block_output_group_stream_speed_packet.get("decision")
        == "token_block_output_group_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_block_output_group_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_output_stripe_group_stream_design_ready = (
        trackb_token_block_output_group_stream_speed_rejected
        and trackb_token_output_stripe_group_stream_design is not None
        and trackb_token_output_stripe_group_stream_design.get("decision")
        == "token_output_stripe_group_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_output_stripe_group_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_output_stripe_group_stream_source_guard_missing = (
        trackb_token_output_stripe_group_stream_design_ready
        and token_output_stripe_group_stream_guard.get("decision")
        == "missing_token_output_stripe_group_stream_source"
    )
    trackb_token_output_stripe_group_stream_source_guard_present = (
        trackb_token_output_stripe_group_stream_design_ready
        and token_output_stripe_group_stream_guard.get("decision")
        == "token_output_stripe_group_stream_source_guardrail_present"
        and bool(token_output_stripe_group_stream_guard.get("passes_contract"))
    )
    trackb_token_output_stripe_group_stream_native_parity_pass = (
        trackb_token_output_stripe_group_stream_source_guard_present
        and trackb_token_output_stripe_group_stream_native_parity is not None
        and trackb_token_output_stripe_group_stream_native_parity.get("decision")
        == "token_output_stripe_group_stream_native_parity_pass"
        and bool(
            trackb_token_output_stripe_group_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_output_stripe_group_stream_artifact_parity_pass = (
        trackb_token_output_stripe_group_stream_native_parity_pass
        and trackb_token_output_stripe_group_stream_artifact_parity is not None
        and trackb_token_output_stripe_group_stream_artifact_parity.get("decision")
        == "token_output_stripe_group_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_output_stripe_group_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_output_stripe_group_stream_speed_rejected = (
        trackb_token_output_stripe_group_stream_artifact_parity_pass
        and trackb_token_output_stripe_group_stream_speed_packet is not None
        and trackb_token_output_stripe_group_stream_speed_packet.get("decision")
        == "reject_token_output_stripe_group_stream_speed_path"
        and bool(
            trackb_token_output_stripe_group_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_output_stripe_group_stream_speed_lane_s_pass = (
        trackb_token_output_stripe_group_stream_artifact_parity_pass
        and trackb_token_output_stripe_group_stream_speed_packet is not None
        and trackb_token_output_stripe_group_stream_speed_packet.get("decision")
        == "token_output_stripe_group_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_output_stripe_group_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_output_stripe_group_stream_speed_parity_pass = (
        trackb_token_output_stripe_group_stream_artifact_parity_pass
        and trackb_token_output_stripe_group_stream_speed_packet is not None
        and trackb_token_output_stripe_group_stream_speed_packet.get("decision")
        == "token_output_stripe_group_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_output_stripe_group_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_expert_output_block_stream_design_ready = (
        trackb_token_output_stripe_group_stream_speed_rejected
        and trackb_token_expert_output_block_stream_design is not None
        and trackb_token_expert_output_block_stream_design.get("decision")
        == "token_expert_output_block_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_expert_output_block_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_expert_output_block_stream_source_guard_missing = (
        trackb_token_expert_output_block_stream_design_ready
        and token_expert_output_block_stream_guard.get("decision")
        == "missing_token_expert_output_block_stream_source"
    )
    trackb_token_expert_output_block_stream_source_guard_present = (
        trackb_token_expert_output_block_stream_design_ready
        and token_expert_output_block_stream_guard.get("decision")
        == "token_expert_output_block_stream_source_guardrail_present"
        and bool(token_expert_output_block_stream_guard.get("passes_contract"))
    )
    trackb_token_expert_output_block_stream_native_parity_pass = (
        trackb_token_expert_output_block_stream_source_guard_present
        and trackb_token_expert_output_block_stream_native_parity is not None
        and trackb_token_expert_output_block_stream_native_parity.get("decision")
        == "token_expert_output_block_stream_native_parity_pass"
        and bool(
            trackb_token_expert_output_block_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_expert_output_block_stream_artifact_parity_pass = (
        trackb_token_expert_output_block_stream_native_parity_pass
        and trackb_token_expert_output_block_stream_artifact_parity is not None
        and trackb_token_expert_output_block_stream_artifact_parity.get("decision")
        == "token_expert_output_block_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_expert_output_block_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_expert_output_block_stream_speed_rejected = (
        trackb_token_expert_output_block_stream_artifact_parity_pass
        and trackb_token_expert_output_block_stream_speed_packet is not None
        and trackb_token_expert_output_block_stream_speed_packet.get("decision")
        == "reject_token_expert_output_block_stream_speed_path"
        and bool(
            trackb_token_expert_output_block_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_expert_output_block_stream_speed_lane_s_pass = (
        trackb_token_expert_output_block_stream_artifact_parity_pass
        and trackb_token_expert_output_block_stream_speed_packet is not None
        and trackb_token_expert_output_block_stream_speed_packet.get("decision")
        == "token_expert_output_block_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_expert_output_block_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_expert_output_block_stream_speed_parity_pass = (
        trackb_token_expert_output_block_stream_artifact_parity_pass
        and trackb_token_expert_output_block_stream_speed_packet is not None
        and trackb_token_expert_output_block_stream_speed_packet.get("decision")
        == "token_expert_output_block_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_expert_output_block_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_kblock_accumulator_stream_design_ready = (
        trackb_token_expert_output_block_stream_speed_rejected
        and trackb_token_pair_kblock_accumulator_stream_design is not None
        and trackb_token_pair_kblock_accumulator_stream_design.get("decision")
        == "token_pair_kblock_accumulator_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_kblock_accumulator_stream_source_guard_missing = (
        trackb_token_pair_kblock_accumulator_stream_design_ready
        and token_pair_kblock_accumulator_stream_guard.get("decision")
        == "missing_token_pair_kblock_accumulator_stream_source"
    )
    trackb_token_pair_kblock_accumulator_stream_source_guard_present = (
        trackb_token_pair_kblock_accumulator_stream_design_ready
        and token_pair_kblock_accumulator_stream_guard.get("decision")
        == "token_pair_kblock_accumulator_stream_source_guardrail_present"
    )
    trackb_token_pair_kblock_accumulator_stream_native_parity_pass = (
        trackb_token_pair_kblock_accumulator_stream_source_guard_present
        and trackb_token_pair_kblock_accumulator_stream_native_parity is not None
        and trackb_token_pair_kblock_accumulator_stream_native_parity.get("decision")
        == "token_pair_kblock_accumulator_stream_native_parity_pass"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass = (
        trackb_token_pair_kblock_accumulator_stream_native_parity_pass
        and trackb_token_pair_kblock_accumulator_stream_artifact_parity is not None
        and trackb_token_pair_kblock_accumulator_stream_artifact_parity.get("decision")
        == "token_pair_kblock_accumulator_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_kblock_accumulator_stream_speed_rejected = (
        trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
        and trackb_token_pair_kblock_accumulator_stream_speed_packet is not None
        and trackb_token_pair_kblock_accumulator_stream_speed_packet.get("decision")
        == "reject_token_pair_kblock_accumulator_stream_speed_path"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_kblock_accumulator_stream_speed_lane_s_pass = (
        trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
        and trackb_token_pair_kblock_accumulator_stream_speed_packet is not None
        and trackb_token_pair_kblock_accumulator_stream_speed_packet.get("decision")
        == "token_pair_kblock_accumulator_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_kblock_accumulator_stream_speed_parity_pass = (
        trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
        and trackb_token_pair_kblock_accumulator_stream_speed_packet is not None
        and trackb_token_pair_kblock_accumulator_stream_speed_packet.get("decision")
        == "token_pair_kblock_accumulator_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_kblock_accumulator_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_output_group_stream_design_ready = (
        trackb_token_pair_kblock_accumulator_stream_speed_rejected
        and trackb_token_pair_output_group_stream_design is not None
        and trackb_token_pair_output_group_stream_design.get("decision")
        == "token_pair_output_group_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_output_group_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_output_group_stream_source_guard_missing = (
        trackb_token_pair_output_group_stream_design_ready
        and token_pair_output_group_stream_guard.get("decision")
        == "missing_token_pair_output_group_stream_source"
    )
    trackb_token_pair_output_group_stream_source_guard_present = (
        trackb_token_pair_output_group_stream_design_ready
        and token_pair_output_group_stream_guard.get("decision")
        == "token_pair_output_group_stream_source_guardrail_present"
    )
    trackb_token_pair_output_group_stream_native_parity_pass = (
        trackb_token_pair_output_group_stream_source_guard_present
        and trackb_token_pair_output_group_stream_native_parity is not None
        and trackb_token_pair_output_group_stream_native_parity.get("decision")
        == "token_pair_output_group_stream_native_parity_pass"
        and bool(
            trackb_token_pair_output_group_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_output_group_stream_artifact_parity_pass = (
        trackb_token_pair_output_group_stream_native_parity_pass
        and trackb_token_pair_output_group_stream_artifact_parity is not None
        and trackb_token_pair_output_group_stream_artifact_parity.get("decision")
        == "token_pair_output_group_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_output_group_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_output_group_stream_speed_rejected = (
        trackb_token_pair_output_group_stream_artifact_parity_pass
        and trackb_token_pair_output_group_stream_speed_packet is not None
        and trackb_token_pair_output_group_stream_speed_packet.get("decision")
        == "reject_token_pair_output_group_stream_speed_path"
        and bool(
            trackb_token_pair_output_group_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_output_group_stream_speed_lane_s_pass = (
        trackb_token_pair_output_group_stream_artifact_parity_pass
        and trackb_token_pair_output_group_stream_speed_packet is not None
        and trackb_token_pair_output_group_stream_speed_packet.get("decision")
        == "token_pair_output_group_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_output_group_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_output_group_stream_speed_parity_pass = (
        trackb_token_pair_output_group_stream_artifact_parity_pass
        and trackb_token_pair_output_group_stream_speed_packet is not None
        and trackb_token_pair_output_group_stream_speed_packet.get("decision")
        == "token_pair_output_group_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_output_group_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_design_ready = (
        trackb_token_pair_output_group_stream_speed_rejected
        and trackb_token_pair_slot_topk_output_group_stream_design is not None
        and trackb_token_pair_slot_topk_output_group_stream_design.get("decision")
        == "token_pair_slot_topk_output_group_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_source_guard_missing = (
        trackb_token_pair_slot_topk_output_group_stream_design_ready
        and token_pair_slot_topk_output_group_stream_guard.get("decision")
        == "missing_token_pair_slot_topk_output_group_stream_source"
    )
    trackb_token_pair_slot_topk_output_group_stream_source_guard_present = (
        trackb_token_pair_slot_topk_output_group_stream_design_ready
        and token_pair_slot_topk_output_group_stream_guard.get("decision")
        == "token_pair_slot_topk_output_group_stream_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_output_group_stream_native_parity_pass = (
        trackb_token_pair_slot_topk_output_group_stream_source_guard_present
        and trackb_token_pair_slot_topk_output_group_stream_native_parity is not None
        and trackb_token_pair_slot_topk_output_group_stream_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_output_group_stream_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass = (
        trackb_token_pair_slot_topk_output_group_stream_native_parity_pass
        and trackb_token_pair_slot_topk_output_group_stream_artifact_parity is not None
        and trackb_token_pair_slot_topk_output_group_stream_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_output_group_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_speed_rejected = (
        trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_output_group_stream_speed_path"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_output_group_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_output_group_stream_speed_parity_pass = (
        trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_output_group_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready = (
        trackb_token_pair_slot_topk_output_group_stream_speed_rejected
        and trackb_token_pair_slot_topk_codeword_group_pipeline_design is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_design.get("decision")
        == "token_pair_slot_topk_codeword_group_pipeline_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_missing = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready
        and token_pair_slot_topk_codeword_group_pipeline_guard.get("decision")
        == "missing_token_pair_slot_topk_codeword_group_pipeline_source"
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_present = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready
        and token_pair_slot_topk_codeword_group_pipeline_guard.get("decision")
        == "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_pass = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_present
        and trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_codeword_group_pipeline_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_pass
        and trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_speed_rejected = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_codeword_group_pipeline_speed_path"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_codeword_group_pipeline_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_codeword_group_pipeline_speed_parity_pass = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
        is not None
        and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_codeword_group_pipeline_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready = (
        trackb_token_pair_slot_topk_codeword_group_pipeline_speed_rejected
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design.get(
            "decision"
        )
        == "token_pair_slot_topk_scale_slot_broadcast_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_missing = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
        and token_pair_slot_topk_scale_slot_broadcast_stream_guard.get("decision")
        == "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_present = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
        and token_pair_slot_topk_scale_slot_broadcast_stream_guard.get("decision")
        == "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_present
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_scale_slot_broadcast_stream_speed_path"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_parity_pass = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready = (
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design.get(
            "decision"
        )
        == "token_pair_slot_topk_route_bucket_codeword_reduce_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_missing = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
        and token_pair_slot_topk_route_bucket_codeword_reduce_guard.get("decision")
        == "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_present = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
        and token_pair_slot_topk_route_bucket_codeword_reduce_guard.get("decision")
        == "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_present
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity.get(
                "passes_native_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity.get(
                "passes_artifact_parity"
            )
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_route_bucket_codeword_reduce_speed_path"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_parity_pass = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
        is not None
        and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready = (
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected
        and trackb_token_pair_slot_topk_kblock_microtile_stream_design is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_design.get("decision")
        == "token_pair_slot_topk_kblock_microtile_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_missing = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready
        and token_pair_slot_topk_kblock_microtile_stream_guard.get("decision")
        == "missing_token_pair_slot_topk_kblock_microtile_stream_source"
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_present = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready
        and token_pair_slot_topk_kblock_microtile_stream_guard.get("decision")
        == "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_pass = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_present
        and trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
        is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_kblock_microtile_stream_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity.get(
                "native_parity_claim"
            )
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_pass
        and trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
        is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity.get(
                "artifact_parity_claim"
            )
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_speed_rejected = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_kblock_microtile_stream_speed_path"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_kblock_microtile_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_kblock_microtile_stream_speed_parity_pass = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet is not None
        and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_kblock_microtile_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready = (
        trackb_token_pair_slot_topk_kblock_microtile_stream_speed_rejected
        and trackb_token_pair_slot_topk_output_tile_fused_stream_design is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_design.get("decision")
        == "token_pair_slot_topk_output_tile_fused_stream_ready_for_source_structure_probe"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_design.get(
                "selector_verdict", {}
            ).get("candidate_ready_for_source_probe")
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_missing = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready
        and token_pair_slot_topk_output_tile_fused_stream_guard.get("decision")
        == "missing_token_pair_slot_topk_output_tile_fused_stream_source"
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_present = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready
        and token_pair_slot_topk_output_tile_fused_stream_guard.get("decision")
        == "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_pass = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_present
        and trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_output_tile_fused_stream_native_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity.get(
                "native_parity_claim"
            )
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_pass
        and trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity.get(
            "decision"
        )
        == "token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity.get(
                "artifact_parity_claim"
            )
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_speed_rejected = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
            "decision"
        )
        == "reject_token_pair_slot_topk_output_tile_fused_stream_speed_path"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
                "same_window_q2_speed_packet"
            )
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_speed_lane_s_pass = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_output_tile_fused_stream_speed_packet_lane_s_pass"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
                "all_lane_s_pass"
            )
        )
    )
    trackb_token_pair_slot_topk_output_tile_fused_stream_speed_parity_pass = (
        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
        is not None
        and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
            "decision"
        )
        == "token_pair_slot_topk_output_tile_fused_stream_speed_packet_parity_pass"
        and bool(
            trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
                "all_parity_pass"
            )
        )
    )
    cache_source_ready = bool(cache_source.get("local_export_candidate_ready"))
    cache_memory = cache_source.get("memory_observation")
    if not isinstance(cache_memory, dict):
        cache_memory = {}
    cache_source_memory_risky = bool(
        cache_memory.get("source_bytes_exceed_physical_memory")
    )
    cache_runtime_source_complete = bool(cache_source.get("complete_runtime_source"))
    cache_full_source_complete = bool(cache_source.get("complete_source"))
    single_host_cache_attempt_memory_killed = (
        single_host_cache_attempt is not None
        and single_host_cache_attempt.get("decision")
        == "single_host_cache_prefix_killed_exit137"
        and int(single_host_cache_attempt.get("exit_code") or 0) == 137
    )
    single_host_cache_attempt_guard_blocked = (
        single_host_cache_attempt is not None
        and (
            single_host_cache_attempt.get("decision")
            == "single_host_source_memory_guard_blocked"
            or single_host_cache_attempt.get("source_memory_guard_pass") is False
        )
    )
    single_host_cache_attempt_generated = bool(
        single_host_cache_attempt is not None
        and single_host_cache_attempt.get("cache_rows_generated")
    )
    single_host_cache_attempt_memory_clean = (
        single_host_cache_attempt.get("all_memory_clean")
        if single_host_cache_attempt is not None
        else None
    )
    glm52_diagnoses = {
        str(glm52_layer3_path): glm52_layer3.get("source_artifact_diagnosis"),
        str(glm52_layer77_path): glm52_layer77.get("source_artifact_diagnosis"),
    }
    glm52_gap_diagnosed = all(
        diagnosis == "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
        for diagnosis in glm52_diagnoses.values()
    )
    model_card_exists = model_card_path.is_file()
    trackb_rejected_family_count = len(trackb.get("rejected_families", []))
    trackb_circuit_broken = (
        trackb_requires_new_family
        and trackb_rejected_family_count >= 3
        and trackb_route_microtile_codeword_block_reduce_speed_rejected
        and not trackb_kblock_wavefront_codeword_scan_design_ready
    )
    trackb_circuit_breaker = (
        {
            "track": "trackb_native_speed",
            "status": "circuit_broken",
            "rejected_family_count": trackb_rejected_family_count,
            "switch_to": "workstream4_cache",
            "requires": "materially_different_rhs_or_kernel_family",
        }
        if trackb_circuit_broken
        else None
    )
    workstream4_evidence: list[str] = [str(cache_source_scan_path)]
    if (
        single_host_cache_attempt is not None
        and single_host_cache_attempt_path is not None
    ):
        workstream4_evidence.append(str(single_host_cache_attempt_path))
    if (
        layer_split_recommendation is not None
        and layer_split_recommendation_path is not None
    ):
        workstream4_evidence.append(str(layer_split_recommendation_path))
    if rdma_topology is not None and rdma_topology_audit_path is not None:
        workstream4_evidence.append(str(rdma_topology_audit_path))
    layer_split_summary = None
    if layer_split_recommendation is not None:
        layer_split_summary = {
            "recommended_layer_split": layer_split_recommendation.get(
                "recommended_layer_split"
            ),
            "candidate_count": layer_split_recommendation.get("candidate_count"),
            "optimization_target": layer_split_recommendation.get(
                "optimization_target"
            ),
            "previous_legacy_recipe_layer_split": layer_split_recommendation.get(
                "previous_legacy_recipe_layer_split"
            ),
            "updated_legacy_recipe_layer_split": layer_split_recommendation.get(
                "updated_legacy_recipe_layer_split"
            ),
        }
    router_kd = _mapping(workstream1.get("train_router_kd_1"))
    fixed_bump = (
        workstream1_fixed_bump
        if workstream1_fixed_bump is not None
        else workstream1
        if workstream1.get("record_type")
        == "glm45_air_workstream1_bump_fixed_loader_verdict"
        else None
    )
    w1_next_recommendation = workstream1.get("next_recommendation")
    if (
        workstream1.get("decision") == "reject_free_p1_levers_metric_backed"
        and fixed_bump is not None
        and fixed_bump.get("next_recommendation") is not None
    ):
        w1_next_recommendation = fixed_bump.get("next_recommendation")

    fronts = {
        "workstream1_p1_free_levers": {
            "status": _workstream1_status(workstream1),
            "evidence": str(workstream1_path),
            "fixed_loader_bump_evidence": (
                str(workstream1_fixed_bump_path)
                if workstream1_fixed_bump is not None
                and workstream1_fixed_bump_path is not None
                else None
            ),
            "decision": workstream1.get("decision"),
            "heavy_slice_rule_satisfied": bool(
                workstream1.get("heavy_slice_rule_satisfied")
            ),
            "full_split_record_count_satisfied": bool(
                workstream1.get("full_split_record_count_satisfied")
            ),
            "implementation_bug_fixed": bool(
                workstream1.get("implementation_bug_fixed")
                or _mapping(fixed_bump).get("implementation_bug_fixed")
            ),
            "non_expert_precision_exhausted": bool(
                workstream1.get("non_expert_precision_exhausted")
                or _mapping(fixed_bump).get("non_expert_precision_exhausted")
            ),
            "quality_promotable": bool(
                workstream1.get("quality_promotable")
                or _mapping(fixed_bump).get("quality_promotable")
            ),
            "residency_promotable": bool(
                workstream1.get("residency_promotable")
                or _mapping(fixed_bump).get("residency_promotable")
            ),
            "lane_s_valid": bool(
                workstream1.get("lane_s_valid")
                or _mapping(fixed_bump).get("lane_s_valid")
            ),
            "router_kd_decision": router_kd.get("decision"),
            "router_kd_compose_triggered": bool(
                router_kd.get("compose_triggered")
            ),
            "router_kd_global_top1_compose_triggered": bool(
                router_kd.get("global_top1_compose_triggered")
            ),
            "router_kd_route_top1_compose_triggered": bool(
                router_kd.get("route_top1_compose_triggered")
            ),
            "router_kd_max_global_top1_delta": router_kd.get(
                "max_global_top1_delta"
            ),
            "router_kd_max_route_domain_top1_delta": router_kd.get(
                "max_route_domain_top1_delta"
            ),
            "fixed_loader_bump_decision": _mapping(fixed_bump).get("decision"),
            "next": w1_next_recommendation,
        },
        "workstream2_trackb": {
            "status": (
                "token_pair_slot_topk_output_tile_fused_stream_speed_lane_s_pass"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_lane_s_pass
                else "token_pair_slot_topk_output_tile_fused_stream_speed_parity_pass"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_parity_pass
                else "token_pair_slot_topk_output_tile_fused_stream_speed_rejected"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_rejected
                else "token_pair_slot_topk_output_tile_fused_stream_artifact_parity_present"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
                else "token_pair_slot_topk_output_tile_fused_stream_native_parity_present"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_pass
                else "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_present
                else "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_missing"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_missing
                else "token_pair_slot_topk_output_tile_fused_stream_design_ready"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready
                else "token_pair_slot_topk_kblock_microtile_stream_speed_rejected"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_rejected
                else "token_pair_slot_topk_kblock_microtile_stream_artifact_parity_present"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
                else "token_pair_slot_topk_kblock_microtile_stream_native_parity_present"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_pass
                else "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_present
                else "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_missing"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_missing
                else "token_pair_slot_topk_kblock_microtile_stream_design_ready"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready
                else "token_pair_slot_topk_route_bucket_codeword_reduce_speed_lane_s_pass"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_lane_s_pass
                else "token_pair_slot_topk_route_bucket_codeword_reduce_speed_parity_pass"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_parity_pass
                else "token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected
                else "token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_present"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
                else "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_present"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass
                else "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_present
                else "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_missing"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_missing
                else "token_pair_slot_topk_route_bucket_codeword_reduce_design_ready"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
                else "token_pair_slot_topk_scale_slot_broadcast_stream_speed_lane_s_pass"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_lane_s_pass
                else "token_pair_slot_topk_scale_slot_broadcast_stream_speed_parity_pass"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_parity_pass
                else "token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected
                else "token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_present"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
                else "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_present"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass
                else "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_present
                else "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_missing"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_missing
                else "token_pair_slot_topk_scale_slot_broadcast_stream_design_ready"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
                else "token_pair_slot_topk_codeword_group_pipeline_speed_lane_s_pass"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_lane_s_pass
                else "token_pair_slot_topk_codeword_group_pipeline_speed_parity_pass"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_parity_pass
                else "token_pair_slot_topk_codeword_group_pipeline_speed_rejected"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_rejected
                else "token_pair_slot_topk_codeword_group_pipeline_artifact_parity_present"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
                else "token_pair_slot_topk_codeword_group_pipeline_native_parity_present"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_pass
                else "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_present
                else "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_missing"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_missing
                else "token_pair_slot_topk_codeword_group_pipeline_design_ready"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready
                else "token_pair_slot_topk_output_group_stream_speed_lane_s_pass"
                if trackb_token_pair_slot_topk_output_group_stream_speed_lane_s_pass
                else "token_pair_slot_topk_output_group_stream_speed_parity_pass"
                if trackb_token_pair_slot_topk_output_group_stream_speed_parity_pass
                else "token_pair_slot_topk_output_group_stream_speed_rejected"
                if trackb_token_pair_slot_topk_output_group_stream_speed_rejected
                else "token_pair_slot_topk_output_group_stream_artifact_parity_present"
                if trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
                else "token_pair_slot_topk_output_group_stream_native_parity_present"
                if trackb_token_pair_slot_topk_output_group_stream_native_parity_pass
                else "token_pair_slot_topk_output_group_stream_source_guardrail_present"
                if trackb_token_pair_slot_topk_output_group_stream_source_guard_present
                else "token_pair_slot_topk_output_group_stream_source_guardrail_missing"
                if trackb_token_pair_slot_topk_output_group_stream_source_guard_missing
                else "token_pair_slot_topk_output_group_stream_design_ready"
                if trackb_token_pair_slot_topk_output_group_stream_design_ready
                else "token_pair_output_group_stream_speed_lane_s_pass"
                if trackb_token_pair_output_group_stream_speed_lane_s_pass
                else "token_pair_output_group_stream_speed_parity_pass"
                if trackb_token_pair_output_group_stream_speed_parity_pass
                else "token_pair_output_group_stream_speed_rejected"
                if trackb_token_pair_output_group_stream_speed_rejected
                else "token_pair_output_group_stream_artifact_parity_present"
                if trackb_token_pair_output_group_stream_artifact_parity_pass
                else "token_pair_output_group_stream_native_parity_present"
                if trackb_token_pair_output_group_stream_native_parity_pass
                else "token_pair_output_group_stream_source_guardrail_present"
                if trackb_token_pair_output_group_stream_source_guard_present
                else "token_pair_output_group_stream_source_guardrail_missing"
                if trackb_token_pair_output_group_stream_source_guard_missing
                else "token_pair_output_group_stream_design_ready"
                if trackb_token_pair_output_group_stream_design_ready
                else "token_pair_kblock_accumulator_stream_speed_lane_s_pass"
                if trackb_token_pair_kblock_accumulator_stream_speed_lane_s_pass
                else "token_pair_kblock_accumulator_stream_speed_parity_pass"
                if trackb_token_pair_kblock_accumulator_stream_speed_parity_pass
                else "token_pair_kblock_accumulator_stream_speed_rejected"
                if trackb_token_pair_kblock_accumulator_stream_speed_rejected
                else "token_pair_kblock_accumulator_stream_artifact_parity_present"
                if trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
                else "token_pair_kblock_accumulator_stream_native_parity_present"
                if trackb_token_pair_kblock_accumulator_stream_native_parity_pass
                else "token_pair_kblock_accumulator_stream_source_guardrail_present"
                if trackb_token_pair_kblock_accumulator_stream_source_guard_present
                else "token_pair_kblock_accumulator_stream_source_guardrail_missing"
                if trackb_token_pair_kblock_accumulator_stream_source_guard_missing
                else "token_pair_kblock_accumulator_stream_design_ready"
                if trackb_token_pair_kblock_accumulator_stream_design_ready
                else "token_expert_output_block_stream_speed_lane_s_pass"
                if trackb_token_expert_output_block_stream_speed_lane_s_pass
                else "token_expert_output_block_stream_speed_parity_pass"
                if trackb_token_expert_output_block_stream_speed_parity_pass
                else "token_expert_output_block_stream_speed_rejected"
                if trackb_token_expert_output_block_stream_speed_rejected
                else "token_expert_output_block_stream_artifact_parity_present"
                if trackb_token_expert_output_block_stream_artifact_parity_pass
                else "token_expert_output_block_stream_native_parity_present"
                if trackb_token_expert_output_block_stream_native_parity_pass
                else "token_expert_output_block_stream_source_guardrail_present"
                if trackb_token_expert_output_block_stream_source_guard_present
                else "token_expert_output_block_stream_source_guardrail_missing"
                if trackb_token_expert_output_block_stream_source_guard_missing
                else "token_expert_output_block_stream_design_ready"
                if trackb_token_expert_output_block_stream_design_ready
                else "token_output_stripe_group_stream_speed_lane_s_pass"
                if trackb_token_output_stripe_group_stream_speed_lane_s_pass
                else "token_output_stripe_group_stream_speed_parity_pass"
                if trackb_token_output_stripe_group_stream_speed_parity_pass
                else "token_output_stripe_group_stream_speed_rejected"
                if trackb_token_output_stripe_group_stream_speed_rejected
                else "token_output_stripe_group_stream_artifact_parity_present"
                if trackb_token_output_stripe_group_stream_artifact_parity_pass
                else "token_output_stripe_group_stream_native_parity_present"
                if trackb_token_output_stripe_group_stream_native_parity_pass
                else "token_output_stripe_group_stream_source_guardrail_present"
                if trackb_token_output_stripe_group_stream_source_guard_present
                else "token_output_stripe_group_stream_source_guardrail_missing"
                if trackb_token_output_stripe_group_stream_source_guard_missing
                else "token_output_stripe_group_stream_design_ready"
                if trackb_token_output_stripe_group_stream_design_ready
                else "token_block_output_group_stream_speed_lane_s_pass"
                if trackb_token_block_output_group_stream_speed_lane_s_pass
                else "token_block_output_group_stream_speed_parity_pass"
                if trackb_token_block_output_group_stream_speed_parity_pass
                else "token_block_output_group_stream_speed_rejected"
                if trackb_token_block_output_group_stream_speed_rejected
                else "token_block_output_group_stream_artifact_parity_present"
                if trackb_token_block_output_group_stream_artifact_parity_pass
                else "token_block_output_group_stream_native_parity_present"
                if trackb_token_block_output_group_stream_native_parity_pass
                else "token_block_output_group_stream_source_guardrail_present"
                if trackb_token_block_output_group_stream_source_guard_present
                else "token_block_output_group_stream_source_guardrail_missing"
                if trackb_token_block_output_group_stream_source_guard_missing
                else "token_block_output_group_stream_design_ready"
                if trackb_token_block_output_group_stream_design_ready
                else "token_topk_output_tile_stream_speed_lane_s_pass"
                if trackb_token_topk_output_tile_stream_speed_lane_s_pass
                else "token_topk_output_tile_stream_speed_parity_pass"
                if trackb_token_topk_output_tile_stream_speed_parity_pass
                else "token_topk_output_tile_stream_speed_rejected"
                if trackb_token_topk_output_tile_stream_speed_rejected
                else "token_topk_output_tile_stream_artifact_parity_present"
                if trackb_token_topk_output_tile_stream_artifact_parity_pass
                else "token_topk_output_tile_stream_native_parity_present"
                if trackb_token_topk_output_tile_stream_native_parity_pass
                else "token_topk_output_tile_stream_source_guardrail_present"
                if trackb_token_topk_output_tile_stream_source_guard_present
                else "token_topk_output_tile_stream_source_guardrail_missing"
                if trackb_token_topk_output_tile_stream_source_guard_missing
                else "token_topk_output_tile_stream_design_ready"
                if trackb_token_topk_output_tile_stream_design_ready
                else "route_tile_output_swizzle_stream_speed_lane_s_pass"
                if trackb_route_tile_output_swizzle_stream_speed_lane_s_pass
                else "route_tile_output_swizzle_stream_speed_parity_pass"
                if trackb_route_tile_output_swizzle_stream_speed_parity_pass
                else "route_tile_output_swizzle_stream_speed_rejected"
                if trackb_route_tile_output_swizzle_stream_speed_rejected
                else "route_tile_output_swizzle_stream_artifact_parity_present"
                if trackb_route_tile_output_swizzle_stream_artifact_parity_pass
                else "route_tile_output_swizzle_stream_native_parity_present"
                if trackb_route_tile_output_swizzle_stream_native_parity_pass
                else "route_tile_output_swizzle_stream_source_guardrail_present"
                if trackb_route_tile_output_swizzle_stream_source_guard_present
                else "route_tile_output_swizzle_stream_source_guardrail_missing"
                if trackb_route_tile_output_swizzle_stream_source_guard_missing
                else "route_tile_output_swizzle_stream_design_ready"
                if trackb_route_tile_output_swizzle_stream_design_ready
                else "kblock_output_group_route_fused_stream_speed_lane_s_pass"
                if trackb_kblock_output_group_route_fused_stream_speed_lane_s_pass
                else "kblock_output_group_route_fused_stream_speed_parity_pass"
                if trackb_kblock_output_group_route_fused_stream_speed_parity_pass
                else "kblock_output_group_route_fused_stream_speed_rejected"
                if trackb_kblock_output_group_route_fused_stream_speed_rejected
                else "kblock_output_group_route_fused_stream_artifact_parity_present"
                if trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
                else "kblock_output_group_route_fused_stream_native_parity_present"
                if trackb_kblock_output_group_route_fused_stream_native_parity_pass
                else "kblock_output_group_route_fused_stream_source_guardrail_present"
                if trackb_kblock_output_group_route_fused_stream_source_guard_present
                else "kblock_output_group_route_fused_stream_source_guardrail_missing"
                if trackb_kblock_output_group_route_fused_stream_source_guard_missing
                else "kblock_output_group_route_fused_stream_design_ready"
                if trackb_kblock_output_group_route_fused_stream_design_ready
                else "output_group_pretransposed_codeword_stream_speed_lane_s_pass"
                if trackb_output_group_pretransposed_codeword_stream_speed_lane_s_pass
                else "output_group_pretransposed_codeword_stream_speed_parity_pass"
                if trackb_output_group_pretransposed_codeword_stream_speed_parity_pass
                else "output_group_pretransposed_codeword_stream_speed_rejected"
                if trackb_output_group_pretransposed_codeword_stream_speed_rejected
                else "output_group_pretransposed_codeword_stream_artifact_parity_present"
                if trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
                else "output_group_pretransposed_codeword_stream_native_parity_present"
                if trackb_output_group_pretransposed_codeword_stream_native_parity_pass
                else "output_group_pretransposed_codeword_stream_source_guardrail_present"
                if trackb_output_group_pretransposed_codeword_stream_source_guard_present
                else "output_group_pretransposed_codeword_stream_source_guardrail_missing"
                if trackb_output_group_pretransposed_codeword_stream_source_guard_missing
                else "output_group_pretransposed_codeword_stream_design_ready"
                if trackb_output_group_pretransposed_codeword_stream_design_ready
                else "route_block_output_group_stream_speed_lane_s_pass"
                if trackb_route_block_output_group_stream_speed_lane_s_pass
                else "route_block_output_group_stream_speed_parity_pass"
                if trackb_route_block_output_group_stream_speed_parity_pass
                else "route_block_output_group_stream_speed_rejected"
                if trackb_route_block_output_group_stream_speed_rejected
                else "route_block_output_group_stream_artifact_parity_present"
                if trackb_route_block_output_group_stream_artifact_parity_pass
                else "route_block_output_group_stream_native_parity_present"
                if trackb_route_block_output_group_stream_native_parity_pass
                else "route_block_output_group_stream_source_guardrail_present"
                if trackb_route_block_output_group_stream_source_guard_present
                else "route_block_output_group_stream_source_guardrail_missing"
                if trackb_route_block_output_group_stream_source_guard_missing
                else "route_block_output_group_stream_design_ready"
                if trackb_route_block_output_group_stream_design_ready
                else "scale_group_route_block_reduce_speed_lane_s_pass"
                if trackb_scale_group_route_block_reduce_speed_lane_s_pass
                else "scale_group_route_block_reduce_speed_parity_pass"
                if trackb_scale_group_route_block_reduce_speed_parity_pass
                else "scale_group_route_block_reduce_speed_rejected"
                if trackb_scale_group_route_block_reduce_speed_rejected
                else "scale_group_route_block_reduce_artifact_parity_present"
                if trackb_scale_group_route_block_reduce_artifact_parity_pass
                else "scale_group_route_block_reduce_native_parity_present"
                if trackb_scale_group_route_block_reduce_native_parity_pass
                else "scale_group_route_block_reduce_source_guardrail_present"
                if trackb_scale_group_route_block_reduce_source_guard_present
                else "scale_group_route_block_reduce_source_guardrail_missing"
                if trackb_scale_group_route_block_reduce_source_guard_missing
                else "scale_group_route_block_reduce_design_ready"
                if trackb_scale_group_route_block_reduce_design_ready
                else "expert_kblock_scale_slot_stream_speed_lane_s_pass"
                if trackb_expert_kblock_scale_slot_stream_speed_lane_s_pass
                else "expert_kblock_scale_slot_stream_speed_parity_pass"
                if trackb_expert_kblock_scale_slot_stream_speed_parity_pass
                else "expert_kblock_scale_slot_stream_speed_rejected"
                if trackb_expert_kblock_scale_slot_stream_speed_rejected
                else "expert_kblock_scale_slot_stream_artifact_parity_present"
                if trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
                else "expert_kblock_scale_slot_stream_native_parity_present"
                if trackb_expert_kblock_scale_slot_stream_native_parity_pass
                else "expert_kblock_scale_slot_stream_source_guardrail_present"
                if trackb_expert_kblock_scale_slot_stream_source_guard_present
                else "expert_kblock_scale_slot_stream_source_guardrail_missing"
                if trackb_expert_kblock_scale_slot_stream_source_guard_missing
                else "expert_kblock_scale_slot_stream_design_ready"
                if trackb_expert_kblock_scale_slot_stream_design_ready
                else "token_route_output_stripe_pipeline_speed_lane_s_pass"
                if trackb_token_route_output_stripe_pipeline_speed_lane_s_pass
                else "token_route_output_stripe_pipeline_speed_parity_pass"
                if trackb_token_route_output_stripe_pipeline_speed_parity_pass
                else "token_route_output_stripe_pipeline_speed_rejected"
                if trackb_token_route_output_stripe_pipeline_speed_rejected
                else "token_route_output_stripe_pipeline_artifact_parity_present"
                if trackb_token_route_output_stripe_pipeline_artifact_parity_pass
                else "token_route_output_stripe_pipeline_native_parity_present"
                if trackb_token_route_output_stripe_pipeline_native_parity_pass
                else "token_route_output_stripe_pipeline_source_guardrail_present"
                if trackb_token_route_output_stripe_pipeline_source_guard_present
                else "token_route_output_stripe_pipeline_source_guardrail_missing"
                if trackb_token_route_output_stripe_pipeline_source_guard_missing
                else "token_route_output_stripe_pipeline_design_ready"
                if trackb_token_route_output_stripe_pipeline_design_ready
                else "kblock_wavefront_codeword_scan_speed_rejected"
                if trackb_kblock_wavefront_codeword_scan_speed_rejected
                else "kblock_wavefront_codeword_scan_speed_lane_s_pass"
                if trackb_kblock_wavefront_codeword_scan_speed_lane_s_pass
                else "kblock_wavefront_codeword_scan_artifact_parity_present"
                if trackb_kblock_wavefront_codeword_scan_artifact_parity_pass
                else "kblock_wavefront_codeword_scan_native_parity_present"
                if trackb_kblock_wavefront_codeword_scan_native_parity_pass
                else "kblock_wavefront_codeword_scan_source_guardrail_present"
                if trackb_kblock_wavefront_codeword_scan_source_guard_present
                else "kblock_wavefront_codeword_scan_source_guardrail_missing"
                if trackb_kblock_wavefront_codeword_scan_source_guard_missing
                else "kblock_wavefront_codeword_scan_design_ready"
                if trackb_kblock_wavefront_codeword_scan_design_ready
                else "trackb_native_speed_circuit_broken"
                if trackb_circuit_broken
                else "route_microtile_codeword_block_reduce_speed_rejected"
                if trackb_route_microtile_codeword_block_reduce_speed_rejected
                else "route_microtile_codeword_block_reduce_speed_lane_s_pass"
                if trackb_route_microtile_codeword_block_reduce_speed_lane_s_pass
                else "route_microtile_codeword_block_reduce_artifact_parity_present"
                if trackb_route_microtile_codeword_block_reduce_artifact_parity_pass
                else "route_microtile_codeword_block_reduce_native_parity_present"
                if trackb_route_microtile_codeword_block_reduce_native_parity_pass
                else "route_microtile_codeword_block_reduce_source_guardrail_present"
                if trackb_route_microtile_codeword_block_reduce_source_guard_present
                else "route_microtile_codeword_block_reduce_source_guardrail_missing"
                if trackb_route_microtile_codeword_block_reduce_source_guard_missing
                else "route_microtile_codeword_block_reduce_design_ready"
                if trackb_route_microtile_codeword_block_reduce_design_ready
                else "output_tile_local_codeword_lut_speed_rejected"
                if trackb_output_tile_local_codeword_lut_speed_rejected
                else "output_tile_local_codeword_lut_speed_lane_s_pass"
                if trackb_output_tile_local_codeword_lut_speed_lane_s_pass
                else "output_tile_local_codeword_lut_artifact_parity_present"
                if trackb_output_tile_local_codeword_lut_artifact_parity_pass
                else "output_tile_local_codeword_lut_native_parity_present"
                if trackb_output_tile_local_codeword_lut_native_parity_pass
                else "output_tile_local_codeword_lut_source_guardrail_present"
                if trackb_output_tile_local_codeword_lut_source_guard_present
                else "output_tile_local_codeword_lut_source_guardrail_missing"
                if trackb_output_tile_local_codeword_lut_source_guard_missing
                else "output_tile_local_codeword_lut_design_ready"
                if trackb_output_tile_local_codeword_lut_design_ready
                else "rowwise_codeword_tile_accumulate_speed_rejected"
                if trackb_rowwise_codeword_tile_accumulate_speed_rejected
                else "rowwise_codeword_tile_accumulate_speed_lane_s_pass"
                if trackb_rowwise_codeword_tile_accumulate_speed_lane_s_pass
                else "rowwise_codeword_tile_accumulate_artifact_parity_present"
                if trackb_rowwise_codeword_tile_accumulate_artifact_parity_pass
                else "rowwise_codeword_tile_accumulate_native_parity_present"
                if trackb_rowwise_codeword_tile_accumulate_native_parity_pass
                else "rowwise_codeword_tile_accumulate_source_guardrail_present"
                if trackb_rowwise_codeword_tile_accumulate_source_guard_present
                else "rowwise_codeword_tile_accumulate_source_guardrail_missing"
                if trackb_rowwise_codeword_tile_accumulate_source_guard_missing
                else "rowwise_codeword_tile_accumulate_design_ready"
                if trackb_rowwise_codeword_tile_accumulate_design_ready
                else "route_codeword_lut_accumulate_speed_rejected"
                if trackb_route_codeword_lut_accumulate_speed_rejected
                else "route_codeword_lut_accumulate_artifact_parity_present"
                if trackb_route_codeword_lut_accumulate_artifact_parity_pass
                else "route_codeword_lut_accumulate_native_parity_present"
                if trackb_route_codeword_lut_accumulate_native_parity_pass
                else "route_codeword_lut_accumulate_source_guardrail_present"
                if trackb_route_codeword_lut_accumulate_source_guard_present
                else "route_codeword_lut_accumulate_source_guardrail_missing"
                if trackb_route_codeword_lut_accumulate_source_guard_missing
                else "route_codeword_lut_accumulate_design_ready"
                if trackb_route_codeword_lut_accumulate_design_ready
                else "requires_new_rhs_or_kernel_family"
                if trackb_materially_new_family_gate_active
                else "expert_kblock_codeword_factor_reuse_speed_rejected"
                if trackb_expert_kblock_codeword_factor_reuse_speed_rejected
                else "expert_kblock_codeword_factor_reuse_speed_lane_s_pass"
                if trackb_expert_kblock_codeword_factor_reuse_speed_lane_s_pass
                else "expert_kblock_codeword_factor_reuse_artifact_parity_present"
                if trackb_expert_kblock_codeword_factor_reuse_artifact_parity_pass
                else "expert_kblock_codeword_factor_reuse_native_parity_present"
                if trackb_expert_kblock_codeword_factor_reuse_native_parity_pass
                else "expert_kblock_codeword_factor_reuse_source_guardrail_present"
                if trackb_expert_kblock_codeword_factor_reuse_source_guard_present
                else "expert_kblock_codeword_factor_reuse_source_guardrail_missing"
                if trackb_expert_kblock_codeword_factor_reuse_source_guard_missing
                else "expert_kblock_codeword_factor_reuse_design_ready"
                if trackb_expert_kblock_codeword_factor_reuse_design_ready
                else "input_stationary_speed_rejected"
                if trackb_input_stationary_codeword_tile_speed_rejected
                else "input_stationary_speed_lane_s_pass"
                if trackb_input_stationary_codeword_tile_speed_lane_s_pass
                else "input_stationary_artifact_parity_present"
                if trackb_input_stationary_codeword_tile_artifact_parity_pass
                else "input_stationary_native_parity_present"
                if trackb_input_stationary_codeword_tile_native_parity_pass
                else "input_stationary_source_guardrail_present"
                if trackb_input_stationary_codeword_tile_source_guard_present
                else "input_stationary_source_guardrail_missing"
                if trackb_input_stationary_codeword_tile_source_guard_missing
                else "input_stationary_codeword_tile_design_ready"
                if trackb_input_stationary_codeword_tile_design_ready
                else "output_stationary_speed_rejected"
                if trackb_output_stationary_codeword_tile_speed_rejected
                else "output_stationary_speed_lane_s_pass"
                if trackb_output_stationary_codeword_tile_speed_lane_s_pass
                else "output_stationary_artifact_parity_present"
                if trackb_output_stationary_codeword_tile_artifact_parity_pass
                else "output_stationary_native_parity_present"
                if trackb_output_stationary_codeword_tile_native_parity_pass
                else "output_stationary_source_guardrail_present"
                if trackb_output_stationary_codeword_tile_source_guard_present
                else "output_stationary_source_guardrail_missing"
                if trackb_output_stationary_codeword_tile_source_guard_missing
                else "output_stationary_codeword_tile_design_ready"
                if trackb_output_stationary_codeword_tile_design_ready
                else "token_cohort_mma_speed_rejected"
                if trackb_token_cohort_mma_codeword_tile_speed_rejected
                else "token_cohort_mma_speed_lane_s_pass"
                if trackb_token_cohort_mma_codeword_tile_speed_lane_s_pass
                else "token_cohort_mma_artifact_parity_present"
                if trackb_token_cohort_mma_codeword_tile_artifact_parity_pass
                else "token_cohort_mma_native_parity_present"
                if trackb_token_cohort_mma_codeword_tile_native_parity_pass
                else "token_cohort_mma_source_guardrail_present"
                if trackb_token_cohort_mma_codeword_tile_source_guard_present
                else "token_cohort_mma_source_guardrail_missing"
                if trackb_token_cohort_mma_codeword_tile_source_guard_missing
                else "token_cohort_mma_design_ready"
                if trackb_token_cohort_mma_codeword_tile_design_ready
                else "token_cohort_speed_rejected"
                if trackb_token_cohort_codeword_stream_speed_rejected
                else "token_cohort_speed_lane_s_pass"
                if trackb_token_cohort_codeword_stream_speed_lane_s_pass
                else "token_cohort_artifact_parity_present"
                if trackb_token_cohort_codeword_stream_artifact_parity_pass
                else "token_cohort_native_parity_present"
                if trackb_token_cohort_codeword_stream_native_parity_pass
                else "token_cohort_source_guardrail_present"
                if trackb_token_cohort_codeword_stream_source_guard_present
                else "token_cohort_source_guardrail_missing"
                if trackb_token_cohort_codeword_stream_source_guard_missing
                else "token_cohort_codeword_stream_design_ready"
                if trackb_token_cohort_codeword_stream_design_ready
                else "requires_new_rhs_or_kernel_family"
                if trackb_next_family_gate_blocks
                else "component_stream_tensorops_speed_rejected"
                if trackb_component_stream_tensorops_speed_rejected
                else "component_stream_partial_reduction_native_parity_present"
                if trackb_component_stream_partial_reduction_native_parity_pass
                else "component_stream_partial_reduction_source_guardrail_present"
                if trackb_component_stream_partial_reduction_source_guard_present
                else "component_stream_partial_reduction_source_guardrail_missing"
                if trackb_component_stream_partial_reduction_source_guard_missing
                else "component_stream_partial_reduction_design_ready"
                if trackb_component_stream_partial_reduction_design_ready
                else "route_batch_segmented_speed_rejected"
                if trackb_route_batch_segmented_speed_rejected
                else "route_batch_segmented_lane_s_speed_present"
                if trackb_route_batch_segmented_speed_lane_s_pass
                else "route_batch_segmented_artifact_parity_present"
                if trackb_route_batch_segmented_artifact_parity_pass
                else "route_batch_segmented_native_parity_present"
                if trackb_route_batch_segmented_native_parity_pass
                else "route_batch_segmented_source_guardrail_present"
                if trackb_route_batch_segmented_source_guard_present
                else "route_batch_segmented_source_guardrail_missing"
                if trackb_route_batch_segmented_source_guard_missing
                else "route_batch_segmented_design_ready"
                if trackb_route_batch_segmented_design_ready
                else "expert_cohort_speed_rejected"
                if trackb_expert_cohort_speed_rejected
                else "expert_cohort_lane_s_speed_present"
                if trackb_expert_cohort_speed_lane_s_pass
                else "expert_cohort_artifact_parity_present"
                if trackb_expert_cohort_artifact_parity_pass
                else "expert_cohort_native_parity_present"
                if trackb_expert_cohort_native_parity_pass
                else "expert_cohort_source_guardrail_present"
                if trackb_expert_cohort_source_guard_present
                else "expert_cohort_source_guardrail_missing"
                if trackb_expert_cohort_source_guard_missing
                else "expert_cohort_design_ready"
                if trackb_expert_cohort_design_ready
                else "active_route_tile_speed_rejected"
                if trackb_active_route_tile_speed_rejected
                else "active_route_tile_lane_s_speed_present"
                if trackb_active_route_tile_speed_lane_s_pass
                else "active_route_tile_artifact_parity_present"
                if trackb_active_route_tile_artifact_parity_pass
                else "active_route_tile_native_parity_present"
                if trackb_active_route_tile_native_parity_pass
                else "active_route_tile_source_guardrail_present"
                if trackb_next_source_guard_present
                else "active_route_tile_source_guardrail_missing"
                if trackb_next_source_guard_missing
                else "active_route_tile_design_ready"
                if trackb_next_design_ready
                else "successor_speed_rejected"
                if trackb_successor_speed_rejected
                else "successor_lane_s_speed_present"
                if trackb_successor_speed_lane_s_pass
                else "successor_artifact_parity_present"
                if trackb_successor_artifact_parity_pass
                else "successor_native_parity_present"
                if trackb_successor_native_parity_pass
                else "successor_source_guardrail_present"
                if trackb_successor_source_guard_present
                else "successor_source_guardrail_missing"
                if trackb_successor_source_guard_missing
                else "successor_design_ready"
                if trackb_successor_design_ready
                else "speed_rejected"
                if trackb_speed_rejected
                else "lane_s_speed_present"
                if trackb_speed_lane_s_pass
                else "artifact_parity_present"
                if trackb_artifact_parity_pass
                else "native_parity_present"
                if trackb_native_parity_pass
                else "source_structure_guardrail_present"
                if trackb_source_guard_present
                else "source_structure_guardrail_missing"
                if trackb_source_guard_missing
                else "source_structure_probe_ready"
                if trackb_design_ready
                else "requires_new_rhs_or_kernel_family"
                if trackb_requires_new_family
                else "needs_attention"
            ),
            "evidence": (
                [
                    str(path)
                    for path in (
                        trackb_path,
                        trackb_design_path,
                        trackb_source_guard_path,
                        trackb_native_parity_path,
                        trackb_artifact_parity_path,
                        trackb_speed_packet_path,
                        trackb_successor_design_path,
                        trackb_successor_native_parity_path,
                        trackb_successor_artifact_parity_path,
                        trackb_successor_speed_packet_path,
                        trackb_next_design_path,
                        trackb_active_route_tile_native_parity_path,
                        trackb_active_route_tile_artifact_parity_path,
                        trackb_active_route_tile_speed_packet_path,
                        trackb_expert_cohort_design_path,
                        trackb_expert_cohort_native_parity_path,
                        trackb_expert_cohort_artifact_parity_path,
                        trackb_expert_cohort_speed_packet_path,
                        trackb_route_batch_segmented_design_path,
                        trackb_route_batch_segmented_native_parity_path,
                        trackb_route_batch_segmented_artifact_parity_path,
                        trackb_route_batch_segmented_speed_packet_path,
                        trackb_component_stream_partial_reduction_design_path,
                        trackb_component_stream_partial_reduction_native_parity_path,
                        trackb_component_stream_tensorops_rejection_path,
                        trackb_next_family_gate_path,
                        trackb_token_cohort_codeword_stream_design_path,
                        trackb_token_cohort_codeword_stream_native_parity_path,
                        trackb_token_cohort_codeword_stream_artifact_parity_path,
                        trackb_token_cohort_codeword_stream_speed_packet_path,
                        trackb_token_cohort_mma_codeword_tile_design_path,
                        trackb_token_cohort_mma_codeword_tile_native_parity_path,
                        trackb_token_cohort_mma_codeword_tile_artifact_parity_path,
                        trackb_token_cohort_mma_codeword_tile_speed_packet_path,
                        trackb_output_stationary_codeword_tile_design_path,
                        trackb_output_stationary_codeword_tile_native_parity_path,
                        trackb_output_stationary_codeword_tile_artifact_parity_path,
                        trackb_output_stationary_codeword_tile_speed_packet_path,
                        trackb_input_stationary_codeword_tile_design_path,
                        trackb_input_stationary_codeword_tile_native_parity_path,
                        trackb_input_stationary_codeword_tile_artifact_parity_path,
                        trackb_input_stationary_codeword_tile_speed_packet_path,
                        trackb_expert_kblock_codeword_factor_reuse_design_path,
                        trackb_expert_kblock_codeword_factor_reuse_native_parity_path,
                        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path,
                        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path,
                        trackb_route_codeword_lut_accumulate_design_path,
                        trackb_route_codeword_lut_accumulate_native_parity_path,
                        trackb_route_codeword_lut_accumulate_artifact_parity_path,
                        trackb_route_codeword_lut_accumulate_speed_packet_path,
                        trackb_rowwise_codeword_tile_accumulate_design_path,
                        trackb_rowwise_codeword_tile_accumulate_native_parity_path,
                        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path,
                        trackb_rowwise_codeword_tile_accumulate_speed_packet_path,
                        trackb_output_tile_local_codeword_lut_design_path,
                        trackb_output_tile_local_codeword_lut_native_parity_path,
                        trackb_output_tile_local_codeword_lut_artifact_parity_path,
                        trackb_output_tile_local_codeword_lut_speed_packet_path,
                        trackb_route_microtile_codeword_block_reduce_design_path,
                        trackb_route_microtile_codeword_block_reduce_native_parity_path,
                        trackb_route_microtile_codeword_block_reduce_artifact_parity_path,
                        trackb_route_microtile_codeword_block_reduce_speed_packet_path,
                        trackb_kblock_wavefront_codeword_scan_design_path,
                        trackb_kblock_wavefront_codeword_scan_native_parity_path,
                        trackb_kblock_wavefront_codeword_scan_artifact_parity_path,
                        trackb_kblock_wavefront_codeword_scan_speed_packet_path,
                        trackb_token_route_output_stripe_pipeline_design_path,
                        trackb_token_route_output_stripe_pipeline_native_parity_path,
                        trackb_token_route_output_stripe_pipeline_artifact_parity_path,
                        trackb_token_route_output_stripe_pipeline_speed_packet_path,
                        trackb_expert_kblock_scale_slot_stream_design_path,
                        trackb_expert_kblock_scale_slot_stream_native_parity_path,
                        trackb_expert_kblock_scale_slot_stream_artifact_parity_path,
                        trackb_expert_kblock_scale_slot_stream_speed_packet_path,
                        trackb_scale_group_route_block_reduce_design_path,
                        trackb_scale_group_route_block_reduce_native_parity_path,
                        trackb_scale_group_route_block_reduce_artifact_parity_path,
                        trackb_scale_group_route_block_reduce_speed_packet_path,
                        trackb_route_block_output_group_stream_design_path,
                        trackb_route_block_output_group_stream_native_parity_path,
                        trackb_route_block_output_group_stream_artifact_parity_path,
                        trackb_route_block_output_group_stream_speed_packet_path,
                        trackb_output_group_pretransposed_codeword_stream_design_path,
                        trackb_output_group_pretransposed_codeword_stream_native_parity_path,
                        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path,
                        trackb_output_group_pretransposed_codeword_stream_speed_packet_path,
                        trackb_kblock_output_group_route_fused_stream_design_path,
                        trackb_kblock_output_group_route_fused_stream_native_parity_path,
                        trackb_kblock_output_group_route_fused_stream_artifact_parity_path,
                        trackb_kblock_output_group_route_fused_stream_speed_packet_path,
                        trackb_route_tile_output_swizzle_stream_design_path,
                        trackb_route_tile_output_swizzle_stream_native_parity_path,
                        trackb_route_tile_output_swizzle_stream_artifact_parity_path,
                        trackb_route_tile_output_swizzle_stream_speed_packet_path,
                        trackb_token_topk_output_tile_stream_design_path,
                        trackb_token_topk_output_tile_stream_native_parity_path,
                        trackb_token_topk_output_tile_stream_artifact_parity_path,
                        trackb_token_topk_output_tile_stream_speed_packet_path,
                        trackb_token_block_output_group_stream_design_path,
                        trackb_token_block_output_group_stream_native_parity_path,
                        trackb_token_block_output_group_stream_artifact_parity_path,
                        trackb_token_block_output_group_stream_speed_packet_path,
                        trackb_token_output_stripe_group_stream_design_path,
                        trackb_token_output_stripe_group_stream_native_parity_path,
                        trackb_token_output_stripe_group_stream_artifact_parity_path,
                        trackb_token_output_stripe_group_stream_speed_packet_path,
                        trackb_token_expert_output_block_stream_design_path,
                        trackb_token_expert_output_block_stream_native_parity_path,
                        trackb_token_expert_output_block_stream_artifact_parity_path,
                        trackb_token_expert_output_block_stream_speed_packet_path,
                        trackb_token_pair_kblock_accumulator_stream_design_path,
                        trackb_token_pair_kblock_accumulator_stream_native_parity_path,
                        trackb_token_pair_kblock_accumulator_stream_artifact_parity_path,
                        trackb_token_pair_kblock_accumulator_stream_speed_packet_path,
                        trackb_token_pair_output_group_stream_design_path,
                        trackb_token_pair_output_group_stream_native_parity_path,
                        trackb_token_pair_output_group_stream_artifact_parity_path,
                        trackb_token_pair_output_group_stream_speed_packet_path,
                        trackb_token_pair_slot_topk_output_group_stream_design_path,
                        trackb_token_pair_slot_topk_output_group_stream_native_parity_path,
                        trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path,
                        trackb_token_pair_slot_topk_output_group_stream_speed_packet_path,
                        trackb_token_pair_slot_topk_codeword_group_pipeline_design_path,
                        trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path,
                        trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path,
                        trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path,
                        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path,
                        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path,
                        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path,
                        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path,
                        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path,
                        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path,
                        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path,
                        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path,
                        trackb_token_pair_slot_topk_kblock_microtile_stream_design_path,
                        trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path,
                        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path,
                        trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path,
                        trackb_token_pair_slot_topk_output_tile_fused_stream_design_path,
                        trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path,
                        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path,
                        trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path,
                    )
                    if path is not None
                    and (
                        path == trackb_path
                        or (path == trackb_design_path and trackb_design is not None)
                        or (
                            path == trackb_source_guard_path
                            and trackb_source_guard is not None
                        )
                        or (
                            path == trackb_native_parity_path
                            and trackb_native_parity is not None
                        )
                        or (
                            path == trackb_artifact_parity_path
                            and trackb_artifact_parity is not None
                        )
                        or (
                            path == trackb_speed_packet_path
                            and trackb_speed_packet is not None
                        )
                        or (
                            path == trackb_successor_design_path
                            and trackb_successor_design is not None
                        )
                        or (
                            path == trackb_successor_native_parity_path
                            and trackb_successor_native_parity is not None
                        )
                        or (
                            path == trackb_successor_artifact_parity_path
                            and trackb_successor_artifact_parity is not None
                        )
                        or (
                            path == trackb_successor_speed_packet_path
                            and trackb_successor_speed_packet is not None
                        )
                        or (
                            path == trackb_next_design_path
                            and trackb_next_design is not None
                        )
                        or (
                            path == trackb_active_route_tile_native_parity_path
                            and trackb_active_route_tile_native_parity is not None
                        )
                        or (
                            path == trackb_active_route_tile_artifact_parity_path
                            and trackb_active_route_tile_artifact_parity is not None
                        )
                        or (
                            path == trackb_active_route_tile_speed_packet_path
                            and trackb_active_route_tile_speed_packet is not None
                        )
                        or (
                            path == trackb_expert_cohort_design_path
                            and trackb_expert_cohort_design is not None
                        )
                        or (
                            path == trackb_expert_cohort_native_parity_path
                            and trackb_expert_cohort_native_parity is not None
                        )
                        or (
                            path == trackb_expert_cohort_artifact_parity_path
                            and trackb_expert_cohort_artifact_parity is not None
                        )
                        or (
                            path == trackb_expert_cohort_speed_packet_path
                            and trackb_expert_cohort_speed_packet is not None
                        )
                        or (
                            path == trackb_route_batch_segmented_design_path
                            and trackb_route_batch_segmented_design is not None
                        )
                        or (
                            path == trackb_route_batch_segmented_native_parity_path
                            and trackb_route_batch_segmented_native_parity is not None
                        )
                        or (
                            path == trackb_route_batch_segmented_artifact_parity_path
                            and trackb_route_batch_segmented_artifact_parity is not None
                        )
                        or (
                            path == trackb_route_batch_segmented_speed_packet_path
                            and trackb_route_batch_segmented_speed_packet is not None
                        )
                        or (
                            path
                            == trackb_component_stream_partial_reduction_design_path
                            and trackb_component_stream_partial_reduction_design
                            is not None
                        )
                        or (
                            path
                            == trackb_component_stream_partial_reduction_native_parity_path
                            and trackb_component_stream_partial_reduction_native_parity
                            is not None
                        )
                        or (
                            path == trackb_component_stream_tensorops_rejection_path
                            and trackb_component_stream_tensorops_rejection is not None
                        )
                        or (
                            path == trackb_next_family_gate_path
                            and trackb_next_family_gate is not None
                        )
                        or (
                            path == trackb_token_cohort_codeword_stream_design_path
                            and trackb_token_cohort_codeword_stream_design is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_codeword_stream_native_parity_path
                            and trackb_token_cohort_codeword_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_codeword_stream_artifact_parity_path
                            and trackb_token_cohort_codeword_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_codeword_stream_speed_packet_path
                            and trackb_token_cohort_codeword_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_mma_codeword_tile_design_path
                            and trackb_token_cohort_mma_codeword_tile_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_mma_codeword_tile_native_parity_path
                            and trackb_token_cohort_mma_codeword_tile_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_mma_codeword_tile_artifact_parity_path
                            and trackb_token_cohort_mma_codeword_tile_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_cohort_mma_codeword_tile_speed_packet_path
                            and trackb_token_cohort_mma_codeword_tile_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_output_stationary_codeword_tile_design_path
                            and trackb_output_stationary_codeword_tile_design is not None
                        )
                        or (
                            path
                            == trackb_output_stationary_codeword_tile_native_parity_path
                            and trackb_output_stationary_codeword_tile_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_stationary_codeword_tile_artifact_parity_path
                            and trackb_output_stationary_codeword_tile_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_stationary_codeword_tile_speed_packet_path
                            and trackb_output_stationary_codeword_tile_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_input_stationary_codeword_tile_design_path
                            and trackb_input_stationary_codeword_tile_design is not None
                        )
                        or (
                            path
                            == trackb_input_stationary_codeword_tile_native_parity_path
                            and trackb_input_stationary_codeword_tile_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_input_stationary_codeword_tile_artifact_parity_path
                            and trackb_input_stationary_codeword_tile_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_input_stationary_codeword_tile_speed_packet_path
                            and trackb_input_stationary_codeword_tile_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_codeword_factor_reuse_design_path
                            and trackb_expert_kblock_codeword_factor_reuse_design
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_codeword_factor_reuse_native_parity_path
                            and trackb_expert_kblock_codeword_factor_reuse_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path
                            and trackb_expert_kblock_codeword_factor_reuse_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_codeword_factor_reuse_speed_packet_path
                            and trackb_expert_kblock_codeword_factor_reuse_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_route_codeword_lut_accumulate_design_path
                            and trackb_route_codeword_lut_accumulate_design is not None
                        )
                        or (
                            path
                            == trackb_route_codeword_lut_accumulate_native_parity_path
                            and trackb_route_codeword_lut_accumulate_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_codeword_lut_accumulate_artifact_parity_path
                            and trackb_route_codeword_lut_accumulate_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_codeword_lut_accumulate_speed_packet_path
                            and trackb_route_codeword_lut_accumulate_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_rowwise_codeword_tile_accumulate_design_path
                            and trackb_rowwise_codeword_tile_accumulate_design
                            is not None
                        )
                        or (
                            path
                            == trackb_rowwise_codeword_tile_accumulate_native_parity_path
                            and trackb_rowwise_codeword_tile_accumulate_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_rowwise_codeword_tile_accumulate_artifact_parity_path
                            and trackb_rowwise_codeword_tile_accumulate_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_rowwise_codeword_tile_accumulate_speed_packet_path
                            and trackb_rowwise_codeword_tile_accumulate_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_output_tile_local_codeword_lut_design_path
                            and trackb_output_tile_local_codeword_lut_design is not None
                        )
                        or (
                            path
                            == trackb_output_tile_local_codeword_lut_native_parity_path
                            and trackb_output_tile_local_codeword_lut_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_tile_local_codeword_lut_artifact_parity_path
                            and trackb_output_tile_local_codeword_lut_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_tile_local_codeword_lut_speed_packet_path
                            and trackb_output_tile_local_codeword_lut_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_route_microtile_codeword_block_reduce_design_path
                            and trackb_route_microtile_codeword_block_reduce_design
                            is not None
                        )
                        or (
                            path
                            == trackb_route_microtile_codeword_block_reduce_native_parity_path
                            and trackb_route_microtile_codeword_block_reduce_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_microtile_codeword_block_reduce_artifact_parity_path
                            and trackb_route_microtile_codeword_block_reduce_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_microtile_codeword_block_reduce_speed_packet_path
                            and trackb_route_microtile_codeword_block_reduce_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_kblock_wavefront_codeword_scan_design_path
                            and trackb_kblock_wavefront_codeword_scan_design
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_wavefront_codeword_scan_native_parity_path
                            and trackb_kblock_wavefront_codeword_scan_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_wavefront_codeword_scan_artifact_parity_path
                            and trackb_kblock_wavefront_codeword_scan_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_wavefront_codeword_scan_speed_packet_path
                            and trackb_kblock_wavefront_codeword_scan_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_route_output_stripe_pipeline_design_path
                            and trackb_token_route_output_stripe_pipeline_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_route_output_stripe_pipeline_native_parity_path
                            and trackb_token_route_output_stripe_pipeline_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_route_output_stripe_pipeline_artifact_parity_path
                            and trackb_token_route_output_stripe_pipeline_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_route_output_stripe_pipeline_speed_packet_path
                            and trackb_token_route_output_stripe_pipeline_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_expert_kblock_scale_slot_stream_design_path
                            and trackb_expert_kblock_scale_slot_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_scale_slot_stream_native_parity_path
                            and trackb_expert_kblock_scale_slot_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_scale_slot_stream_artifact_parity_path
                            and trackb_expert_kblock_scale_slot_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_expert_kblock_scale_slot_stream_speed_packet_path
                            and trackb_expert_kblock_scale_slot_stream_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_scale_group_route_block_reduce_design_path
                            and trackb_scale_group_route_block_reduce_design
                            is not None
                        )
                        or (
                            path
                            == trackb_scale_group_route_block_reduce_native_parity_path
                            and trackb_scale_group_route_block_reduce_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_scale_group_route_block_reduce_artifact_parity_path
                            and trackb_scale_group_route_block_reduce_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_scale_group_route_block_reduce_speed_packet_path
                            and trackb_scale_group_route_block_reduce_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_route_block_output_group_stream_design_path
                            and trackb_route_block_output_group_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_route_block_output_group_stream_native_parity_path
                            and trackb_route_block_output_group_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_block_output_group_stream_artifact_parity_path
                            and trackb_route_block_output_group_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_block_output_group_stream_speed_packet_path
                            and trackb_route_block_output_group_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_output_group_pretransposed_codeword_stream_design_path
                            and trackb_output_group_pretransposed_codeword_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_output_group_pretransposed_codeword_stream_native_parity_path
                            and trackb_output_group_pretransposed_codeword_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_group_pretransposed_codeword_stream_artifact_parity_path
                            and trackb_output_group_pretransposed_codeword_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_output_group_pretransposed_codeword_stream_speed_packet_path
                            and trackb_output_group_pretransposed_codeword_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_output_group_route_fused_stream_design_path
                            and trackb_kblock_output_group_route_fused_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_output_group_route_fused_stream_native_parity_path
                            and trackb_kblock_output_group_route_fused_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_output_group_route_fused_stream_artifact_parity_path
                            and trackb_kblock_output_group_route_fused_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_kblock_output_group_route_fused_stream_speed_packet_path
                            and trackb_kblock_output_group_route_fused_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_route_tile_output_swizzle_stream_design_path
                            and trackb_route_tile_output_swizzle_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_route_tile_output_swizzle_stream_native_parity_path
                            and trackb_route_tile_output_swizzle_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_tile_output_swizzle_stream_artifact_parity_path
                            and trackb_route_tile_output_swizzle_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_route_tile_output_swizzle_stream_speed_packet_path
                            and trackb_route_tile_output_swizzle_stream_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_token_topk_output_tile_stream_design_path
                            and trackb_token_topk_output_tile_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_topk_output_tile_stream_native_parity_path
                            and trackb_token_topk_output_tile_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_topk_output_tile_stream_artifact_parity_path
                            and trackb_token_topk_output_tile_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_topk_output_tile_stream_speed_packet_path
                            and trackb_token_topk_output_tile_stream_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_token_block_output_group_stream_design_path
                            and trackb_token_block_output_group_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_block_output_group_stream_native_parity_path
                            and trackb_token_block_output_group_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_block_output_group_stream_artifact_parity_path
                            and trackb_token_block_output_group_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_block_output_group_stream_speed_packet_path
                            and trackb_token_block_output_group_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_output_stripe_group_stream_design_path
                            and trackb_token_output_stripe_group_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_output_stripe_group_stream_native_parity_path
                            and trackb_token_output_stripe_group_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_output_stripe_group_stream_artifact_parity_path
                            and trackb_token_output_stripe_group_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_output_stripe_group_stream_speed_packet_path
                            and trackb_token_output_stripe_group_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_expert_output_block_stream_design_path
                            and trackb_token_expert_output_block_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_expert_output_block_stream_native_parity_path
                            and trackb_token_expert_output_block_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_expert_output_block_stream_artifact_parity_path
                            and trackb_token_expert_output_block_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_expert_output_block_stream_speed_packet_path
                            and trackb_token_expert_output_block_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_kblock_accumulator_stream_design_path
                            and trackb_token_pair_kblock_accumulator_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_kblock_accumulator_stream_native_parity_path
                            and trackb_token_pair_kblock_accumulator_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_kblock_accumulator_stream_artifact_parity_path
                            and trackb_token_pair_kblock_accumulator_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_kblock_accumulator_stream_speed_packet_path
                            and trackb_token_pair_kblock_accumulator_stream_speed_packet
                            is not None
                        )
                        or (
                            path == trackb_token_pair_output_group_stream_design_path
                            and trackb_token_pair_output_group_stream_design is not None
                        )
                        or (
                            path
                            == trackb_token_pair_output_group_stream_native_parity_path
                            and trackb_token_pair_output_group_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_output_group_stream_artifact_parity_path
                            and trackb_token_pair_output_group_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_output_group_stream_speed_packet_path
                            and trackb_token_pair_output_group_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_group_stream_design_path
                            and trackb_token_pair_slot_topk_output_group_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_group_stream_native_parity_path
                            and trackb_token_pair_slot_topk_output_group_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path
                            and trackb_token_pair_slot_topk_output_group_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_group_stream_speed_packet_path
                            and trackb_token_pair_slot_topk_output_group_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_codeword_group_pipeline_design_path
                            and trackb_token_pair_slot_topk_codeword_group_pipeline_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path
                            and trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path
                            and trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path
                            and trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path
                            and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path
                            and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path
                            and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path
                            and trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path
                            and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path
                            and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path
                            and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path
                            and trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_kblock_microtile_stream_design_path
                            and trackb_token_pair_slot_topk_kblock_microtile_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path
                            and trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path
                            and trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path
                            and trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_tile_fused_stream_design_path
                            and trackb_token_pair_slot_topk_output_tile_fused_stream_design
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path
                            and trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path
                            and trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
                            is not None
                        )
                        or (
                            path
                            == trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path
                            and trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
                            is not None
                        )
                    )
                ]
                if (
                    trackb_design is not None
                    or trackb_source_guard is not None
                    or trackb_native_parity is not None
                    or trackb_artifact_parity is not None
                    or trackb_speed_packet is not None
                    or trackb_successor_design is not None
                    or trackb_successor_native_parity is not None
                    or trackb_successor_artifact_parity is not None
                    or trackb_successor_speed_packet is not None
                    or trackb_next_design is not None
                    or trackb_active_route_tile_native_parity is not None
                    or trackb_active_route_tile_artifact_parity is not None
                    or trackb_active_route_tile_speed_packet is not None
                    or trackb_expert_cohort_design is not None
                    or trackb_expert_cohort_native_parity is not None
                    or trackb_expert_cohort_artifact_parity is not None
                    or trackb_expert_cohort_speed_packet is not None
                    or trackb_route_batch_segmented_design is not None
                    or trackb_route_batch_segmented_native_parity is not None
                    or trackb_route_batch_segmented_artifact_parity is not None
                    or trackb_route_batch_segmented_speed_packet is not None
                    or trackb_component_stream_partial_reduction_design is not None
                    or trackb_component_stream_partial_reduction_native_parity
                    is not None
                    or trackb_component_stream_tensorops_rejection is not None
                    or trackb_next_family_gate is not None
                    or trackb_token_cohort_codeword_stream_design is not None
                    or trackb_token_cohort_codeword_stream_native_parity is not None
                    or trackb_token_cohort_codeword_stream_artifact_parity is not None
                    or trackb_token_cohort_codeword_stream_speed_packet is not None
                    or trackb_token_cohort_mma_codeword_tile_design is not None
                    or trackb_token_cohort_mma_codeword_tile_native_parity is not None
                    or trackb_token_cohort_mma_codeword_tile_artifact_parity
                    is not None
                    or trackb_token_cohort_mma_codeword_tile_speed_packet is not None
                    or trackb_output_stationary_codeword_tile_design is not None
                    or trackb_output_stationary_codeword_tile_native_parity is not None
                    or trackb_output_stationary_codeword_tile_artifact_parity
                    is not None
                    or trackb_output_stationary_codeword_tile_speed_packet is not None
                    or trackb_input_stationary_codeword_tile_design is not None
                    or trackb_input_stationary_codeword_tile_native_parity
                    is not None
                    or trackb_input_stationary_codeword_tile_artifact_parity
                    is not None
                    or trackb_input_stationary_codeword_tile_speed_packet is not None
                    or trackb_expert_kblock_codeword_factor_reuse_design is not None
                    or trackb_expert_kblock_codeword_factor_reuse_native_parity
                    is not None
                    or trackb_expert_kblock_codeword_factor_reuse_artifact_parity
                    is not None
                    or trackb_expert_kblock_codeword_factor_reuse_speed_packet
                    is not None
                    or trackb_route_codeword_lut_accumulate_design is not None
                    or trackb_route_codeword_lut_accumulate_native_parity
                    is not None
                    or trackb_route_codeword_lut_accumulate_artifact_parity
                    is not None
                    or trackb_route_codeword_lut_accumulate_speed_packet is not None
                    or trackb_rowwise_codeword_tile_accumulate_design is not None
                    or trackb_rowwise_codeword_tile_accumulate_native_parity
                    is not None
                    or trackb_rowwise_codeword_tile_accumulate_artifact_parity
                    is not None
                    or trackb_rowwise_codeword_tile_accumulate_speed_packet is not None
                    or trackb_output_tile_local_codeword_lut_design is not None
                    or trackb_output_tile_local_codeword_lut_native_parity is not None
                    or trackb_output_tile_local_codeword_lut_artifact_parity
                    is not None
                    or trackb_output_tile_local_codeword_lut_speed_packet is not None
                    or trackb_route_microtile_codeword_block_reduce_design is not None
                    or trackb_route_microtile_codeword_block_reduce_native_parity
                    is not None
                    or trackb_route_microtile_codeword_block_reduce_artifact_parity
                    is not None
                    or trackb_route_microtile_codeword_block_reduce_speed_packet
                    is not None
                    or trackb_kblock_wavefront_codeword_scan_design is not None
                    or trackb_kblock_wavefront_codeword_scan_native_parity
                    is not None
                    or trackb_kblock_wavefront_codeword_scan_artifact_parity
                    is not None
                    or trackb_kblock_wavefront_codeword_scan_speed_packet
                    is not None
                    or trackb_token_route_output_stripe_pipeline_design is not None
                    or trackb_token_route_output_stripe_pipeline_native_parity
                    is not None
                    or trackb_token_route_output_stripe_pipeline_artifact_parity
                    is not None
                    or trackb_token_route_output_stripe_pipeline_speed_packet
                    is not None
                    or trackb_expert_kblock_scale_slot_stream_design is not None
                    or trackb_expert_kblock_scale_slot_stream_native_parity
                    is not None
                    or trackb_expert_kblock_scale_slot_stream_artifact_parity
                    is not None
                    or trackb_expert_kblock_scale_slot_stream_speed_packet
                    is not None
                    or trackb_scale_group_route_block_reduce_design is not None
                    or trackb_scale_group_route_block_reduce_native_parity
                    is not None
                    or trackb_scale_group_route_block_reduce_artifact_parity
                    is not None
                    or trackb_scale_group_route_block_reduce_speed_packet
                    is not None
                    or trackb_route_block_output_group_stream_design is not None
                    or trackb_route_block_output_group_stream_native_parity
                    is not None
                    or trackb_route_block_output_group_stream_artifact_parity
                    is not None
                    or trackb_route_block_output_group_stream_speed_packet
                    is not None
                    or trackb_output_group_pretransposed_codeword_stream_design
                    is not None
                    or trackb_output_group_pretransposed_codeword_stream_native_parity
                    is not None
                    or trackb_output_group_pretransposed_codeword_stream_artifact_parity
                    is not None
                    or trackb_output_group_pretransposed_codeword_stream_speed_packet
                    is not None
                    or trackb_kblock_output_group_route_fused_stream_design
                    is not None
                    or trackb_kblock_output_group_route_fused_stream_native_parity
                    is not None
                    or trackb_kblock_output_group_route_fused_stream_artifact_parity
                    is not None
                    or trackb_kblock_output_group_route_fused_stream_speed_packet
                    is not None
                    or trackb_route_tile_output_swizzle_stream_design is not None
                    or trackb_route_tile_output_swizzle_stream_native_parity
                    is not None
                    or trackb_route_tile_output_swizzle_stream_artifact_parity
                    is not None
                    or trackb_route_tile_output_swizzle_stream_speed_packet
                    is not None
                    or trackb_token_topk_output_tile_stream_design is not None
                    or trackb_token_topk_output_tile_stream_native_parity
                    is not None
                    or trackb_token_topk_output_tile_stream_artifact_parity
                    is not None
                    or trackb_token_topk_output_tile_stream_speed_packet is not None
                    or trackb_token_block_output_group_stream_design is not None
                    or trackb_token_block_output_group_stream_native_parity
                    is not None
                    or trackb_token_block_output_group_stream_artifact_parity
                    is not None
                    or trackb_token_block_output_group_stream_speed_packet
                    is not None
                    or trackb_token_output_stripe_group_stream_design
                    is not None
                    or trackb_token_output_stripe_group_stream_native_parity
                    is not None
                    or trackb_token_output_stripe_group_stream_artifact_parity
                    is not None
                    or trackb_token_output_stripe_group_stream_speed_packet is not None
                    or trackb_token_expert_output_block_stream_design is not None
                    or trackb_token_expert_output_block_stream_native_parity
                    is not None
                    or trackb_token_expert_output_block_stream_artifact_parity
                    is not None
                    or trackb_token_expert_output_block_stream_speed_packet
                    is not None
                    or trackb_token_pair_kblock_accumulator_stream_design
                    is not None
                    or trackb_token_pair_kblock_accumulator_stream_native_parity
                    is not None
                    or trackb_token_pair_kblock_accumulator_stream_artifact_parity
                    is not None
                    or trackb_token_pair_kblock_accumulator_stream_speed_packet
                    is not None
                    or trackb_token_pair_output_group_stream_design is not None
                    or trackb_token_pair_output_group_stream_native_parity
                    is not None
                    or trackb_token_pair_output_group_stream_artifact_parity
                    is not None
                    or trackb_token_pair_output_group_stream_speed_packet is not None
                    or trackb_token_pair_slot_topk_output_group_stream_design
                    is not None
                    or trackb_token_pair_slot_topk_output_group_stream_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_output_group_stream_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_output_group_stream_speed_packet
                    is not None
                    or trackb_token_pair_slot_topk_codeword_group_pipeline_design
                    is not None
                    or trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
                    is not None
                    or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design
                    is not None
                    or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
                    is not None
                    or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design
                    is not None
                    or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
                    is not None
                    or trackb_token_pair_slot_topk_kblock_microtile_stream_design
                    is not None
                    or trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet
                    is not None
                    or trackb_token_pair_slot_topk_output_tile_fused_stream_design
                    is not None
                    or trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
                    is not None
                    or trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
                    is not None
                    or trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
                    is not None
                )
                else str(trackb_path)
            ),
            "frontier_decision": trackb.get("frontier_decision"),
            "rejected_family_count": trackb_rejected_family_count,
            "circuit_breaker": trackb_circuit_breaker,
            "ready_for_native_speed_work": bool(
                trackb.get("ready_for_native_speed_work")
            ),
            "next_required_features": trackb.get("next_required_features", []),
            "candidate_design_decision": (
                trackb_successor_design.get("decision")
                if trackb_successor_design is not None
                else trackb_design.get("decision")
                if trackb_design is not None
                else None
            ),
            "candidate_target_kernel_family": (
                trackb_successor_design.get("candidate", {}).get("target_kernel_family")
                if trackb_successor_design is not None
                else trackb_design.get("candidate", {}).get("target_kernel_family")
                if trackb_design is not None
                else None
            ),
            "successor_design_decision": (
                trackb_successor_design.get("decision")
                if trackb_successor_design is not None
                else None
            ),
            "successor_target_kernel_family": (
                trackb_successor_design.get("candidate", {}).get("target_kernel_family")
                if trackb_successor_design is not None
                else None
            ),
            "successor_source_guardrail_decision": route_slot_mma_guard.get("decision"),
            "successor_native_parity_decision": (
                trackb_successor_native_parity.get("decision")
                if trackb_successor_native_parity is not None
                else None
            ),
            "successor_artifact_parity_decision": (
                trackb_successor_artifact_parity.get("decision")
                if trackb_successor_artifact_parity is not None
                else None
            ),
            "successor_speed_packet_decision": (
                trackb_successor_speed_packet.get("decision")
                if trackb_successor_speed_packet is not None
                else None
            ),
            "next_design_decision": (
                trackb_next_design.get("decision")
                if trackb_next_design is not None
                else None
            ),
            "next_target_kernel_family": (
                trackb_next_design.get("candidate", {}).get("target_kernel_family")
                if trackb_next_design is not None
                else None
            ),
            "next_source_guardrail_decision": active_route_tile_guard.get("decision"),
            "active_route_tile_native_parity_decision": (
                trackb_active_route_tile_native_parity.get("decision")
                if trackb_active_route_tile_native_parity is not None
                else None
            ),
            "active_route_tile_artifact_parity_decision": (
                trackb_active_route_tile_artifact_parity.get("decision")
                if trackb_active_route_tile_artifact_parity is not None
                else None
            ),
            "active_route_tile_speed_packet_decision": (
                trackb_active_route_tile_speed_packet.get("decision")
                if trackb_active_route_tile_speed_packet is not None
                else None
            ),
            "expert_cohort_design_decision": (
                trackb_expert_cohort_design.get("decision")
                if trackb_expert_cohort_design is not None
                else None
            ),
            "expert_cohort_target_kernel_family": (
                trackb_expert_cohort_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_expert_cohort_design is not None
                else None
            ),
            "expert_cohort_source_guardrail_decision": expert_cohort_guard.get(
                "decision"
            ),
            "expert_cohort_native_parity_decision": (
                trackb_expert_cohort_native_parity.get("decision")
                if trackb_expert_cohort_native_parity is not None
                else None
            ),
            "expert_cohort_artifact_parity_decision": (
                trackb_expert_cohort_artifact_parity.get("decision")
                if trackb_expert_cohort_artifact_parity is not None
                else None
            ),
            "expert_cohort_speed_packet_decision": (
                trackb_expert_cohort_speed_packet.get("decision")
                if trackb_expert_cohort_speed_packet is not None
                else None
            ),
            "route_batch_segmented_design_decision": (
                trackb_route_batch_segmented_design.get("decision")
                if trackb_route_batch_segmented_design is not None
                else None
            ),
            "route_batch_segmented_target_kernel_family": (
                trackb_route_batch_segmented_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_route_batch_segmented_design is not None
                else None
            ),
            "route_batch_segmented_source_guardrail_decision": (
                route_batch_segmented_guard.get("decision")
            ),
            "route_batch_segmented_native_parity_decision": (
                trackb_route_batch_segmented_native_parity.get("decision")
                if trackb_route_batch_segmented_native_parity is not None
                else None
            ),
            "route_batch_segmented_artifact_parity_decision": (
                trackb_route_batch_segmented_artifact_parity.get("decision")
                if trackb_route_batch_segmented_artifact_parity is not None
                else None
            ),
            "route_batch_segmented_speed_packet_decision": (
                trackb_route_batch_segmented_speed_packet.get("decision")
                if trackb_route_batch_segmented_speed_packet is not None
                else None
            ),
            "component_stream_partial_reduction_design_decision": (
                trackb_component_stream_partial_reduction_design.get("decision")
                if trackb_component_stream_partial_reduction_design is not None
                else None
            ),
            "component_stream_partial_reduction_target_kernel_family": (
                trackb_component_stream_partial_reduction_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_component_stream_partial_reduction_design is not None
                else None
            ),
            "component_stream_partial_reduction_source_guardrail_decision": (
                component_stream_partial_reduction_guard.get("decision")
            ),
            "component_stream_partial_reduction_native_parity_decision": (
                trackb_component_stream_partial_reduction_native_parity.get("decision")
                if trackb_component_stream_partial_reduction_native_parity is not None
                else None
            ),
            "component_stream_tensorops_rejection_decision": (
                component_stream_tensorops_guard.get("decision")
            ),
            "next_family_gate_decision": (
                trackb_next_family_gate.get("decision")
                if trackb_next_family_gate is not None
                else None
            ),
            "token_cohort_codeword_stream_design_decision": (
                trackb_token_cohort_codeword_stream_design.get("decision")
                if trackb_token_cohort_codeword_stream_design is not None
                else None
            ),
            "token_cohort_codeword_stream_target_kernel_family": (
                trackb_token_cohort_codeword_stream_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_token_cohort_codeword_stream_design is not None
                else None
            ),
            "token_cohort_codeword_stream_source_guardrail_decision": (
                token_cohort_codeword_stream_guard.get("decision")
            ),
            "token_cohort_codeword_stream_native_parity_decision": (
                trackb_token_cohort_codeword_stream_native_parity.get("decision")
                if trackb_token_cohort_codeword_stream_native_parity is not None
                else None
            ),
            "token_cohort_codeword_stream_artifact_parity_decision": (
                trackb_token_cohort_codeword_stream_artifact_parity.get("decision")
                if trackb_token_cohort_codeword_stream_artifact_parity is not None
                else None
            ),
            "token_cohort_codeword_stream_speed_packet_decision": (
                trackb_token_cohort_codeword_stream_speed_packet.get("decision")
                if trackb_token_cohort_codeword_stream_speed_packet is not None
                else None
            ),
            "token_cohort_mma_design_decision": (
                trackb_token_cohort_mma_codeword_tile_design.get("decision")
                if trackb_token_cohort_mma_codeword_tile_design is not None
                else None
            ),
            "token_cohort_mma_target_kernel_family": (
                trackb_token_cohort_mma_codeword_tile_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_token_cohort_mma_codeword_tile_design is not None
                else None
            ),
            "token_cohort_mma_source_guardrail_decision": (
                token_cohort_mma_codeword_tile_guard.get("decision")
            ),
            "token_cohort_mma_native_parity_decision": (
                trackb_token_cohort_mma_codeword_tile_native_parity.get("decision")
                if trackb_token_cohort_mma_codeword_tile_native_parity is not None
                else None
            ),
            "token_cohort_mma_artifact_parity_decision": (
                trackb_token_cohort_mma_codeword_tile_artifact_parity.get("decision")
                if trackb_token_cohort_mma_codeword_tile_artifact_parity is not None
                else None
            ),
            "token_cohort_mma_speed_packet_decision": (
                trackb_token_cohort_mma_codeword_tile_speed_packet.get("decision")
                if trackb_token_cohort_mma_codeword_tile_speed_packet is not None
                else None
            ),
            "output_stationary_codeword_tile_design_decision": (
                trackb_output_stationary_codeword_tile_design.get("decision")
                if trackb_output_stationary_codeword_tile_design is not None
                else None
            ),
            "output_stationary_codeword_tile_target_kernel_family": (
                trackb_output_stationary_codeword_tile_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_output_stationary_codeword_tile_design is not None
                else None
            ),
            "output_stationary_codeword_tile_source_guardrail_decision": (
                output_stationary_codeword_tile_guard.get("decision")
            ),
            "output_stationary_codeword_tile_native_parity_decision": (
                trackb_output_stationary_codeword_tile_native_parity.get("decision")
                if trackb_output_stationary_codeword_tile_native_parity is not None
                else None
            ),
            "output_stationary_codeword_tile_artifact_parity_decision": (
                trackb_output_stationary_codeword_tile_artifact_parity.get("decision")
                if trackb_output_stationary_codeword_tile_artifact_parity is not None
                else None
            ),
            "output_stationary_codeword_tile_speed_packet_decision": (
                trackb_output_stationary_codeword_tile_speed_packet.get("decision")
                if trackb_output_stationary_codeword_tile_speed_packet is not None
                else None
            ),
            "input_stationary_codeword_tile_design_decision": (
                trackb_input_stationary_codeword_tile_design.get("decision")
                if trackb_input_stationary_codeword_tile_design is not None
                else None
            ),
            "input_stationary_codeword_tile_target_kernel_family": (
                trackb_input_stationary_codeword_tile_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_input_stationary_codeword_tile_design is not None
                else None
            ),
            "input_stationary_codeword_tile_source_guardrail_decision": (
                input_stationary_codeword_tile_guard.get("decision")
            ),
            "input_stationary_codeword_tile_native_parity_decision": (
                trackb_input_stationary_codeword_tile_native_parity.get("decision")
                if trackb_input_stationary_codeword_tile_native_parity is not None
                else None
            ),
            "input_stationary_codeword_tile_artifact_parity_decision": (
                trackb_input_stationary_codeword_tile_artifact_parity.get("decision")
                if trackb_input_stationary_codeword_tile_artifact_parity is not None
                else None
            ),
            "input_stationary_codeword_tile_speed_packet_decision": (
                trackb_input_stationary_codeword_tile_speed_packet.get("decision")
                if trackb_input_stationary_codeword_tile_speed_packet is not None
                else None
            ),
            "expert_kblock_codeword_factor_reuse_design_decision": (
                trackb_expert_kblock_codeword_factor_reuse_design.get("decision")
                if trackb_expert_kblock_codeword_factor_reuse_design is not None
                else None
            ),
            "expert_kblock_codeword_factor_reuse_target_kernel_family": (
                trackb_expert_kblock_codeword_factor_reuse_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_expert_kblock_codeword_factor_reuse_design is not None
                else None
            ),
            "expert_kblock_codeword_factor_reuse_source_guardrail_decision": (
                expert_kblock_codeword_factor_reuse_guard.get("decision")
            ),
            "expert_kblock_codeword_factor_reuse_native_parity_decision": (
                trackb_expert_kblock_codeword_factor_reuse_native_parity.get(
                    "decision"
                )
                if trackb_expert_kblock_codeword_factor_reuse_native_parity
                is not None
                else None
            ),
            "expert_kblock_codeword_factor_reuse_artifact_parity_decision": (
                trackb_expert_kblock_codeword_factor_reuse_artifact_parity.get(
                    "decision"
                )
                if trackb_expert_kblock_codeword_factor_reuse_artifact_parity
                is not None
                else None
            ),
            "expert_kblock_codeword_factor_reuse_speed_packet_decision": (
                trackb_expert_kblock_codeword_factor_reuse_speed_packet.get(
                    "decision"
                )
                if trackb_expert_kblock_codeword_factor_reuse_speed_packet
                is not None
                else None
            ),
            "route_codeword_lut_accumulate_design_decision": (
                trackb_route_codeword_lut_accumulate_design.get("decision")
                if trackb_route_codeword_lut_accumulate_design is not None
                else None
            ),
            "route_codeword_lut_accumulate_target_kernel_family": (
                trackb_route_codeword_lut_accumulate_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_route_codeword_lut_accumulate_design is not None
                else None
            ),
            "route_codeword_lut_accumulate_source_guardrail_decision": (
                route_codeword_lut_accumulate_guard.get("decision")
            ),
            "route_codeword_lut_accumulate_native_parity_decision": (
                trackb_route_codeword_lut_accumulate_native_parity.get("decision")
                if trackb_route_codeword_lut_accumulate_native_parity is not None
                else None
            ),
            "route_codeword_lut_accumulate_artifact_parity_decision": (
                trackb_route_codeword_lut_accumulate_artifact_parity.get("decision")
                if trackb_route_codeword_lut_accumulate_artifact_parity is not None
                else None
            ),
            "route_codeword_lut_accumulate_speed_packet_decision": (
                trackb_route_codeword_lut_accumulate_speed_packet.get("decision")
                if trackb_route_codeword_lut_accumulate_speed_packet is not None
                else None
            ),
            "rowwise_codeword_tile_accumulate_design_decision": (
                trackb_rowwise_codeword_tile_accumulate_design.get("decision")
                if trackb_rowwise_codeword_tile_accumulate_design is not None
                else None
            ),
            "rowwise_codeword_tile_accumulate_target_kernel_family": (
                trackb_rowwise_codeword_tile_accumulate_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_rowwise_codeword_tile_accumulate_design is not None
                else None
            ),
            "rowwise_codeword_tile_accumulate_source_guardrail_decision": (
                rowwise_codeword_tile_accumulate_guard.get("decision")
            ),
            "rowwise_codeword_tile_accumulate_native_parity_decision": (
                trackb_rowwise_codeword_tile_accumulate_native_parity.get("decision")
                if trackb_rowwise_codeword_tile_accumulate_native_parity is not None
                else None
            ),
            "rowwise_codeword_tile_accumulate_artifact_parity_decision": (
                trackb_rowwise_codeword_tile_accumulate_artifact_parity.get(
                    "decision"
                )
                if trackb_rowwise_codeword_tile_accumulate_artifact_parity
                is not None
                else None
            ),
            "rowwise_codeword_tile_accumulate_speed_packet_decision": (
                trackb_rowwise_codeword_tile_accumulate_speed_packet.get("decision")
                if trackb_rowwise_codeword_tile_accumulate_speed_packet is not None
                else None
            ),
            "output_tile_local_codeword_lut_design_decision": (
                trackb_output_tile_local_codeword_lut_design.get("decision")
                if trackb_output_tile_local_codeword_lut_design is not None
                else None
            ),
            "output_tile_local_codeword_lut_target_kernel_family": (
                trackb_output_tile_local_codeword_lut_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_output_tile_local_codeword_lut_design is not None
                else None
            ),
            "output_tile_local_codeword_lut_source_guardrail_decision": (
                output_tile_local_codeword_lut_guard.get("decision")
            ),
            "output_tile_local_codeword_lut_native_parity_decision": (
                trackb_output_tile_local_codeword_lut_native_parity.get("decision")
                if trackb_output_tile_local_codeword_lut_native_parity is not None
                else None
            ),
            "output_tile_local_codeword_lut_artifact_parity_decision": (
                trackb_output_tile_local_codeword_lut_artifact_parity.get("decision")
                if trackb_output_tile_local_codeword_lut_artifact_parity is not None
                else None
            ),
            "output_tile_local_codeword_lut_speed_packet_decision": (
                trackb_output_tile_local_codeword_lut_speed_packet.get("decision")
                if trackb_output_tile_local_codeword_lut_speed_packet is not None
                else None
            ),
            "route_microtile_codeword_block_reduce_design_decision": (
                trackb_route_microtile_codeword_block_reduce_design.get("decision")
                if trackb_route_microtile_codeword_block_reduce_design is not None
                else None
            ),
            "route_microtile_codeword_block_reduce_target_kernel_family": (
                trackb_route_microtile_codeword_block_reduce_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_route_microtile_codeword_block_reduce_design is not None
                else None
            ),
            "route_microtile_codeword_block_reduce_source_guardrail_decision": (
                route_microtile_codeword_block_reduce_guard.get("decision")
            ),
            "route_microtile_codeword_block_reduce_native_parity_decision": (
                trackb_route_microtile_codeword_block_reduce_native_parity.get(
                    "decision"
                )
                if trackb_route_microtile_codeword_block_reduce_native_parity
                is not None
                else None
            ),
            "route_microtile_codeword_block_reduce_artifact_parity_decision": (
                trackb_route_microtile_codeword_block_reduce_artifact_parity.get(
                    "decision"
                )
                if trackb_route_microtile_codeword_block_reduce_artifact_parity
                is not None
                else None
            ),
            "route_microtile_codeword_block_reduce_speed_packet_decision": (
                trackb_route_microtile_codeword_block_reduce_speed_packet.get(
                    "decision"
                )
                if trackb_route_microtile_codeword_block_reduce_speed_packet
                is not None
                else None
            ),
            "kblock_wavefront_codeword_scan_design_decision": (
                trackb_kblock_wavefront_codeword_scan_design.get("decision")
                if trackb_kblock_wavefront_codeword_scan_design is not None
                else None
            ),
            "kblock_wavefront_codeword_scan_target_kernel_family": (
                trackb_kblock_wavefront_codeword_scan_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_kblock_wavefront_codeword_scan_design is not None
                else None
            ),
            "kblock_wavefront_codeword_scan_source_guardrail_decision": (
                kblock_wavefront_codeword_scan_guard.get("decision")
            ),
            "kblock_wavefront_codeword_scan_native_parity_decision": (
                trackb_kblock_wavefront_codeword_scan_native_parity.get("decision")
                if trackb_kblock_wavefront_codeword_scan_native_parity is not None
                else None
            ),
            "kblock_wavefront_codeword_scan_artifact_parity_decision": (
                trackb_kblock_wavefront_codeword_scan_artifact_parity.get("decision")
                if trackb_kblock_wavefront_codeword_scan_artifact_parity is not None
                else None
            ),
            "kblock_wavefront_codeword_scan_speed_packet_decision": (
                trackb_kblock_wavefront_codeword_scan_speed_packet.get("decision")
                if trackb_kblock_wavefront_codeword_scan_speed_packet is not None
                else None
            ),
            "token_route_output_stripe_pipeline_design_decision": (
                trackb_token_route_output_stripe_pipeline_design.get("decision")
                if trackb_token_route_output_stripe_pipeline_design is not None
                else None
            ),
            "token_route_output_stripe_pipeline_target_kernel_family": (
                trackb_token_route_output_stripe_pipeline_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_route_output_stripe_pipeline_design is not None
                else None
            ),
            "token_route_output_stripe_pipeline_source_guardrail_decision": (
                token_route_output_stripe_pipeline_guard.get("decision")
            ),
            "token_route_output_stripe_pipeline_native_parity_decision": (
                trackb_token_route_output_stripe_pipeline_native_parity.get(
                    "decision"
                )
                if trackb_token_route_output_stripe_pipeline_native_parity
                is not None
                else None
            ),
            "token_route_output_stripe_pipeline_artifact_parity_decision": (
                trackb_token_route_output_stripe_pipeline_artifact_parity.get(
                    "decision"
                )
                if trackb_token_route_output_stripe_pipeline_artifact_parity
                is not None
                else None
            ),
            "token_route_output_stripe_pipeline_speed_packet_decision": (
                trackb_token_route_output_stripe_pipeline_speed_packet.get("decision")
                if trackb_token_route_output_stripe_pipeline_speed_packet is not None
                else None
            ),
            "expert_kblock_scale_slot_stream_design_decision": (
                trackb_expert_kblock_scale_slot_stream_design.get("decision")
                if trackb_expert_kblock_scale_slot_stream_design is not None
                else None
            ),
            "expert_kblock_scale_slot_stream_target_kernel_family": (
                trackb_expert_kblock_scale_slot_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_expert_kblock_scale_slot_stream_design is not None
                else None
            ),
            "expert_kblock_scale_slot_stream_source_guardrail_decision": (
                expert_kblock_scale_slot_stream_guard.get("decision")
            ),
            "expert_kblock_scale_slot_stream_native_parity_decision": (
                trackb_expert_kblock_scale_slot_stream_native_parity.get("decision")
                if trackb_expert_kblock_scale_slot_stream_native_parity is not None
                else None
            ),
            "expert_kblock_scale_slot_stream_artifact_parity_decision": (
                trackb_expert_kblock_scale_slot_stream_artifact_parity.get("decision")
                if trackb_expert_kblock_scale_slot_stream_artifact_parity is not None
                else None
            ),
            "expert_kblock_scale_slot_stream_speed_packet_decision": (
                trackb_expert_kblock_scale_slot_stream_speed_packet.get("decision")
                if trackb_expert_kblock_scale_slot_stream_speed_packet is not None
                else None
            ),
            "scale_group_route_block_reduce_design_decision": (
                trackb_scale_group_route_block_reduce_design.get("decision")
                if trackb_scale_group_route_block_reduce_design is not None
                else None
            ),
            "scale_group_route_block_reduce_target_kernel_family": (
                trackb_scale_group_route_block_reduce_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_scale_group_route_block_reduce_design is not None
                else None
            ),
            "scale_group_route_block_reduce_source_guardrail_decision": (
                scale_group_route_block_reduce_guard.get("decision")
            ),
            "scale_group_route_block_reduce_native_parity_decision": (
                trackb_scale_group_route_block_reduce_native_parity.get("decision")
                if trackb_scale_group_route_block_reduce_native_parity is not None
                else None
            ),
            "scale_group_route_block_reduce_artifact_parity_decision": (
                trackb_scale_group_route_block_reduce_artifact_parity.get("decision")
                if trackb_scale_group_route_block_reduce_artifact_parity is not None
                else None
            ),
            "scale_group_route_block_reduce_speed_packet_decision": (
                trackb_scale_group_route_block_reduce_speed_packet.get("decision")
                if trackb_scale_group_route_block_reduce_speed_packet is not None
                else None
            ),
            "route_block_output_group_stream_design_decision": (
                trackb_route_block_output_group_stream_design.get("decision")
                if trackb_route_block_output_group_stream_design is not None
                else None
            ),
            "route_block_output_group_stream_target_kernel_family": (
                trackb_route_block_output_group_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_route_block_output_group_stream_design is not None
                else None
            ),
            "route_block_output_group_stream_source_guardrail_decision": (
                route_block_output_group_stream_guard.get("decision")
            ),
            "route_block_output_group_stream_native_parity_decision": (
                trackb_route_block_output_group_stream_native_parity.get("decision")
                if trackb_route_block_output_group_stream_native_parity is not None
                else None
            ),
            "route_block_output_group_stream_artifact_parity_decision": (
                trackb_route_block_output_group_stream_artifact_parity.get("decision")
                if trackb_route_block_output_group_stream_artifact_parity is not None
                else None
            ),
            "route_block_output_group_stream_speed_packet_decision": (
                trackb_route_block_output_group_stream_speed_packet.get("decision")
                if trackb_route_block_output_group_stream_speed_packet is not None
                else None
            ),
            "output_group_pretransposed_codeword_stream_design_decision": (
                trackb_output_group_pretransposed_codeword_stream_design.get(
                    "decision"
                )
                if trackb_output_group_pretransposed_codeword_stream_design is not None
                else None
            ),
            "output_group_pretransposed_codeword_stream_target_kernel_family": (
                trackb_output_group_pretransposed_codeword_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_output_group_pretransposed_codeword_stream_design is not None
                else None
            ),
            "output_group_pretransposed_codeword_stream_source_guardrail_decision": (
                output_group_pretransposed_codeword_stream_guard.get("decision")
            ),
            "output_group_pretransposed_codeword_stream_native_parity_decision": (
                trackb_output_group_pretransposed_codeword_stream_native_parity.get(
                    "decision"
                )
                if trackb_output_group_pretransposed_codeword_stream_native_parity
                is not None
                else None
            ),
            "output_group_pretransposed_codeword_stream_artifact_parity_decision": (
                trackb_output_group_pretransposed_codeword_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_output_group_pretransposed_codeword_stream_artifact_parity
                is not None
                else None
            ),
            "output_group_pretransposed_codeword_stream_speed_packet_decision": (
                trackb_output_group_pretransposed_codeword_stream_speed_packet.get(
                    "decision"
                )
                if trackb_output_group_pretransposed_codeword_stream_speed_packet
                is not None
                else None
            ),
            "kblock_output_group_route_fused_stream_design_decision": (
                trackb_kblock_output_group_route_fused_stream_design.get("decision")
                if trackb_kblock_output_group_route_fused_stream_design is not None
                else None
            ),
            "kblock_output_group_route_fused_stream_target_kernel_family": (
                trackb_kblock_output_group_route_fused_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_kblock_output_group_route_fused_stream_design is not None
                else None
            ),
            "kblock_output_group_route_fused_stream_source_guardrail_decision": (
                kblock_output_group_route_fused_stream_guard.get("decision")
            ),
            "kblock_output_group_route_fused_stream_native_parity_decision": (
                trackb_kblock_output_group_route_fused_stream_native_parity.get(
                    "decision"
                )
                if trackb_kblock_output_group_route_fused_stream_native_parity
                is not None
                else None
            ),
            "kblock_output_group_route_fused_stream_artifact_parity_decision": (
                trackb_kblock_output_group_route_fused_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_kblock_output_group_route_fused_stream_artifact_parity
                is not None
                else None
            ),
            "kblock_output_group_route_fused_stream_speed_packet_decision": (
                trackb_kblock_output_group_route_fused_stream_speed_packet.get(
                    "decision"
                )
                if trackb_kblock_output_group_route_fused_stream_speed_packet
                is not None
                else None
            ),
            "route_tile_output_swizzle_stream_design_decision": (
                trackb_route_tile_output_swizzle_stream_design.get("decision")
                if trackb_route_tile_output_swizzle_stream_design is not None
                else None
            ),
            "route_tile_output_swizzle_stream_target_kernel_family": (
                trackb_route_tile_output_swizzle_stream_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_route_tile_output_swizzle_stream_design is not None
                else None
            ),
            "route_tile_output_swizzle_stream_source_guardrail_decision": (
                route_tile_output_swizzle_stream_guard.get("decision")
            ),
            "route_tile_output_swizzle_stream_native_parity_decision": (
                trackb_route_tile_output_swizzle_stream_native_parity.get(
                    "decision"
                )
                if trackb_route_tile_output_swizzle_stream_native_parity is not None
                else None
            ),
            "route_tile_output_swizzle_stream_artifact_parity_decision": (
                trackb_route_tile_output_swizzle_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_route_tile_output_swizzle_stream_artifact_parity is not None
                else None
            ),
            "route_tile_output_swizzle_stream_speed_packet_decision": (
                trackb_route_tile_output_swizzle_stream_speed_packet.get(
                    "decision"
                )
                if trackb_route_tile_output_swizzle_stream_speed_packet is not None
                else None
            ),
            "token_topk_output_tile_stream_design_decision": (
                trackb_token_topk_output_tile_stream_design.get("decision")
                if trackb_token_topk_output_tile_stream_design is not None
                else None
            ),
            "token_topk_output_tile_stream_target_kernel_family": (
                trackb_token_topk_output_tile_stream_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_token_topk_output_tile_stream_design is not None
                else None
            ),
            "token_topk_output_tile_stream_source_guardrail_decision": (
                token_topk_output_tile_stream_guard.get("decision")
            ),
            "token_topk_output_tile_stream_native_parity_decision": (
                trackb_token_topk_output_tile_stream_native_parity.get("decision")
                if trackb_token_topk_output_tile_stream_native_parity is not None
                else None
            ),
            "token_topk_output_tile_stream_artifact_parity_decision": (
                trackb_token_topk_output_tile_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_topk_output_tile_stream_artifact_parity is not None
                else None
            ),
            "token_topk_output_tile_stream_speed_packet_decision": (
                trackb_token_topk_output_tile_stream_speed_packet.get("decision")
                if trackb_token_topk_output_tile_stream_speed_packet is not None
                else None
            ),
            "token_block_output_group_stream_design_decision": (
                trackb_token_block_output_group_stream_design.get("decision")
                if trackb_token_block_output_group_stream_design is not None
                else None
            ),
            "token_block_output_group_stream_target_kernel_family": (
                trackb_token_block_output_group_stream_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_token_block_output_group_stream_design is not None
                else None
            ),
            "token_block_output_group_stream_source_guardrail_decision": (
                token_block_output_group_stream_guard.get("decision")
            ),
            "token_block_output_group_stream_native_parity_decision": (
                trackb_token_block_output_group_stream_native_parity.get("decision")
                if trackb_token_block_output_group_stream_native_parity is not None
                else None
            ),
            "token_block_output_group_stream_artifact_parity_decision": (
                trackb_token_block_output_group_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_block_output_group_stream_artifact_parity is not None
                else None
            ),
            "token_block_output_group_stream_speed_packet_decision": (
                trackb_token_block_output_group_stream_speed_packet.get("decision")
                if trackb_token_block_output_group_stream_speed_packet is not None
                else None
            ),
            "token_output_stripe_group_stream_design_decision": (
                trackb_token_output_stripe_group_stream_design.get("decision")
                if trackb_token_output_stripe_group_stream_design is not None
                else None
            ),
            "token_output_stripe_group_stream_target_kernel_family": (
                trackb_token_output_stripe_group_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_output_stripe_group_stream_design is not None
                else None
            ),
            "token_output_stripe_group_stream_source_guardrail_decision": (
                token_output_stripe_group_stream_guard.get("decision")
            ),
            "token_output_stripe_group_stream_native_parity_decision": (
                trackb_token_output_stripe_group_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_output_stripe_group_stream_native_parity is not None
                else None
            ),
            "token_output_stripe_group_stream_artifact_parity_decision": (
                trackb_token_output_stripe_group_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_output_stripe_group_stream_artifact_parity is not None
                else None
            ),
            "token_output_stripe_group_stream_speed_packet_decision": (
                trackb_token_output_stripe_group_stream_speed_packet.get("decision")
                if trackb_token_output_stripe_group_stream_speed_packet is not None
                else None
            ),
            "token_expert_output_block_stream_design_decision": (
                trackb_token_expert_output_block_stream_design.get("decision")
                if trackb_token_expert_output_block_stream_design is not None
                else None
            ),
            "token_expert_output_block_stream_target_kernel_family": (
                trackb_token_expert_output_block_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_expert_output_block_stream_design is not None
                else None
            ),
            "token_expert_output_block_stream_source_guardrail_decision": (
                token_expert_output_block_stream_guard.get("decision")
            ),
            "token_expert_output_block_stream_native_parity_decision": (
                trackb_token_expert_output_block_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_expert_output_block_stream_native_parity is not None
                else None
            ),
            "token_expert_output_block_stream_artifact_parity_decision": (
                trackb_token_expert_output_block_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_expert_output_block_stream_artifact_parity
                is not None
                else None
            ),
            "token_expert_output_block_stream_speed_packet_decision": (
                trackb_token_expert_output_block_stream_speed_packet.get("decision")
                if trackb_token_expert_output_block_stream_speed_packet is not None
                else None
            ),
            "token_pair_kblock_accumulator_stream_design_decision": (
                trackb_token_pair_kblock_accumulator_stream_design.get("decision")
                if trackb_token_pair_kblock_accumulator_stream_design is not None
                else None
            ),
            "token_pair_kblock_accumulator_stream_target_kernel_family": (
                trackb_token_pair_kblock_accumulator_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_kblock_accumulator_stream_design is not None
                else None
            ),
            "token_pair_kblock_accumulator_stream_source_guardrail_decision": (
                token_pair_kblock_accumulator_stream_guard.get("decision")
            ),
            "token_pair_kblock_accumulator_stream_native_parity_decision": (
                trackb_token_pair_kblock_accumulator_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_kblock_accumulator_stream_native_parity
                is not None
                else None
            ),
            "token_pair_kblock_accumulator_stream_artifact_parity_decision": (
                trackb_token_pair_kblock_accumulator_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_kblock_accumulator_stream_artifact_parity
                is not None
                else None
            ),
            "token_pair_kblock_accumulator_stream_speed_packet_decision": (
                trackb_token_pair_kblock_accumulator_stream_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_kblock_accumulator_stream_speed_packet
                is not None
                else None
            ),
            "token_pair_output_group_stream_design_decision": (
                trackb_token_pair_output_group_stream_design.get("decision")
                if trackb_token_pair_output_group_stream_design is not None
                else None
            ),
            "token_pair_output_group_stream_target_kernel_family": (
                trackb_token_pair_output_group_stream_design.get("candidate", {}).get(
                    "target_kernel_family"
                )
                if trackb_token_pair_output_group_stream_design is not None
                else None
            ),
            "token_pair_output_group_stream_source_guardrail_decision": (
                token_pair_output_group_stream_guard.get("decision")
            ),
            "token_pair_output_group_stream_native_parity_decision": (
                trackb_token_pair_output_group_stream_native_parity.get("decision")
                if trackb_token_pair_output_group_stream_native_parity
                else None
            ),
            "token_pair_output_group_stream_artifact_parity_decision": (
                trackb_token_pair_output_group_stream_artifact_parity.get("decision")
                if trackb_token_pair_output_group_stream_artifact_parity
                else None
            ),
            "token_pair_output_group_stream_speed_packet_decision": (
                trackb_token_pair_output_group_stream_speed_packet.get("decision")
                if trackb_token_pair_output_group_stream_speed_packet
                else None
            ),
            "token_pair_slot_topk_output_group_stream_design_decision": (
                trackb_token_pair_slot_topk_output_group_stream_design.get("decision")
                if trackb_token_pair_slot_topk_output_group_stream_design is not None
                else None
            ),
            "token_pair_slot_topk_output_group_stream_target_kernel_family": (
                trackb_token_pair_slot_topk_output_group_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_output_group_stream_design is not None
                else None
            ),
            "token_pair_slot_topk_output_group_stream_source_guardrail_decision": (
                token_pair_slot_topk_output_group_stream_guard.get("decision")
            ),
            "token_pair_slot_topk_output_group_stream_native_parity_decision": (
                trackb_token_pair_slot_topk_output_group_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_group_stream_native_parity
                else None
            ),
            "token_pair_slot_topk_output_group_stream_artifact_parity_decision": (
                trackb_token_pair_slot_topk_output_group_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_group_stream_artifact_parity
                else None
            ),
            "token_pair_slot_topk_output_group_stream_speed_packet_decision": (
                trackb_token_pair_slot_topk_output_group_stream_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_group_stream_speed_packet
                else None
            ),
            "token_pair_slot_topk_codeword_group_pipeline_design_decision": (
                trackb_token_pair_slot_topk_codeword_group_pipeline_design.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_codeword_group_pipeline_design
                is not None
                else None
            ),
            "token_pair_slot_topk_codeword_group_pipeline_target_kernel_family": (
                trackb_token_pair_slot_topk_codeword_group_pipeline_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_codeword_group_pipeline_design
                is not None
                else None
            ),
            "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_decision": (
                token_pair_slot_topk_codeword_group_pipeline_guard.get("decision")
            ),
            "token_pair_slot_topk_codeword_group_pipeline_native_parity_decision": (
                trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity
                else None
            ),
            "token_pair_slot_topk_codeword_group_pipeline_artifact_parity_decision": (
                trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity
                else None
            ),
            "token_pair_slot_topk_codeword_group_pipeline_speed_packet_decision": (
                trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet
                else None
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_design_decision": (
                trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_target_kernel_family": (
                trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_decision": (
                token_pair_slot_topk_scale_slot_broadcast_stream_guard.get("decision")
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_decision": (
                trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity
                else None
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_decision": (
                trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity
                else None
            ),
            "token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_decision": (
                trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet
                else None
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_design_decision": (
                trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design
                is not None
                else None
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_target_kernel_family": (
                trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design
                is not None
                else None
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_decision": (
                token_pair_slot_topk_route_bucket_codeword_reduce_guard.get("decision")
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_decision": (
                trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity
                else None
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_decision": (
                trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity
                else None
            ),
            "token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_decision": (
                trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet
                else None
            ),
            "token_pair_slot_topk_kblock_microtile_stream_design_decision": (
                trackb_token_pair_slot_topk_kblock_microtile_stream_design.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_kblock_microtile_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_kblock_microtile_stream_target_kernel_family": (
                trackb_token_pair_slot_topk_kblock_microtile_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_kblock_microtile_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_decision": (
                token_pair_slot_topk_kblock_microtile_stream_guard.get("decision")
            ),
            "token_pair_slot_topk_kblock_microtile_stream_native_parity_decision": (
                trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity
                is not None
                else None
            ),
            "token_pair_slot_topk_kblock_microtile_stream_artifact_parity_decision": (
                trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity
                is not None
                else None
            ),
            "token_pair_slot_topk_kblock_microtile_stream_speed_packet_decision": (
                trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet
                is not None
                else None
            ),
            "token_pair_slot_topk_output_tile_fused_stream_design_decision": (
                trackb_token_pair_slot_topk_output_tile_fused_stream_design.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_tile_fused_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_output_tile_fused_stream_target_kernel_family": (
                trackb_token_pair_slot_topk_output_tile_fused_stream_design.get(
                    "candidate", {}
                ).get("target_kernel_family")
                if trackb_token_pair_slot_topk_output_tile_fused_stream_design
                is not None
                else None
            ),
            "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_decision": (
                token_pair_slot_topk_output_tile_fused_stream_guard.get("decision")
            ),
            "token_pair_slot_topk_output_tile_fused_stream_native_parity_decision": (
                trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity
                is not None
                else None
            ),
            "token_pair_slot_topk_output_tile_fused_stream_artifact_parity_decision": (
                trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity
                is not None
                else None
            ),
            "token_pair_slot_topk_output_tile_fused_stream_speed_packet_decision": (
                trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet.get(
                    "decision"
                )
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet
                is not None
                else None
            ),
            "source_guardrail_decision": route_slot_guard.get("decision"),
            "native_parity_decision": (
                trackb_native_parity.get("decision")
                if trackb_native_parity is not None
                else None
            ),
            "artifact_parity_decision": (
                trackb_artifact_parity.get("decision")
                if trackb_artifact_parity is not None
                else None
            ),
            "speed_packet_decision": (
                trackb_speed_packet.get("decision")
                if trackb_speed_packet is not None
                else None
            ),
            "next": (
                "route_resident_auto_after_token_pair_slot_topk_output_tile_fused_stream_lane_s_gate"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_output_tile_fused_stream_speed_shapes"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_parity_pass
                else "change_token_pair_slot_topk_output_tile_fused_stream_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_rejected
                else "run_token_pair_slot_topk_output_tile_fused_stream_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
                else "prove_token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_pass
                else "prove_token_pair_slot_topk_output_tile_fused_stream_native_parity"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_output_tile_fused_stream"
                if trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready
                else "route_resident_auto_after_token_pair_slot_topk_kblock_microtile_stream_lane_s_gate"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_kblock_microtile_stream_speed_shapes"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_parity_pass
                else "change_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_rejected
                else "run_token_pair_slot_topk_kblock_microtile_stream_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
                else "prove_token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_pass
                else "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity"
                if trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_kblock_microtile_stream"
                if (
                    trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_missing
                    or trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready
                )
                else "route_resident_auto_after_token_pair_slot_topk_route_bucket_codeword_reduce_lane_s_gate"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_route_bucket_codeword_reduce_speed_shapes"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_parity_pass
                else "change_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected
                else "run_token_pair_slot_topk_route_bucket_codeword_reduce_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
                else "prove_token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass
                else "prove_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
                if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_route_bucket_codeword_reduce"
                if (
                    trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_missing
                    or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
                )
                else "route_resident_auto_after_token_pair_slot_topk_scale_slot_broadcast_stream_lane_s_gate"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_scale_slot_broadcast_stream_speed_shapes"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_parity_pass
                else "change_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected
                else "run_token_pair_slot_topk_scale_slot_broadcast_stream_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
                else "prove_token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass
                else "prove_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
                if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_scale_slot_broadcast_stream"
                if (
                    trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_missing
                    or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
                )
                else "route_resident_auto_after_token_pair_slot_topk_codeword_group_pipeline_lane_s_gate"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_codeword_group_pipeline_speed_shapes"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_parity_pass
                else "change_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_rejected
                else "run_token_pair_slot_topk_codeword_group_pipeline_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
                else "prove_token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_pass
                else "prove_token_pair_slot_topk_codeword_group_pipeline_native_parity"
                if trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_codeword_group_pipeline"
                if (
                    trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_missing
                    or trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready
                )
                else "route_resident_auto_after_token_pair_slot_topk_output_group_stream_lane_s_gate"
                if trackb_token_pair_slot_topk_output_group_stream_speed_lane_s_pass
                else "run_missing_token_pair_slot_topk_output_group_stream_speed_shapes"
                if trackb_token_pair_slot_topk_output_group_stream_speed_parity_pass
                else "change_token_pair_slot_topk_output_group_stream_layout_or_kernel_family"
                if trackb_token_pair_slot_topk_output_group_stream_speed_rejected
                else "run_token_pair_slot_topk_output_group_stream_same_window_q2_speed_packet"
                if trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
                else "prove_token_pair_slot_topk_output_group_stream_air_artifact_parity"
                if trackb_token_pair_slot_topk_output_group_stream_native_parity_pass
                else "prove_token_pair_slot_topk_output_group_stream_native_parity"
                if trackb_token_pair_slot_topk_output_group_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_slot_topk_output_group_stream"
                if (
                    trackb_token_pair_slot_topk_output_group_stream_source_guard_missing
                    or trackb_token_pair_slot_topk_output_group_stream_design_ready
                )
                else "route_resident_auto_after_token_pair_output_group_stream_lane_s_gate"
                if trackb_token_pair_output_group_stream_speed_lane_s_pass
                else "run_missing_token_pair_output_group_stream_speed_shapes"
                if trackb_token_pair_output_group_stream_speed_parity_pass
                else "change_token_pair_output_group_stream_layout_or_kernel_family"
                if trackb_token_pair_output_group_stream_speed_rejected
                else "run_token_pair_output_group_stream_same_window_q2_speed_packet"
                if trackb_token_pair_output_group_stream_artifact_parity_pass
                else "prove_token_pair_output_group_stream_air_artifact_parity"
                if trackb_token_pair_output_group_stream_native_parity_pass
                else "prove_token_pair_output_group_stream_native_parity"
                if trackb_token_pair_output_group_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_output_group_stream"
                if (
                    trackb_token_pair_output_group_stream_source_guard_missing
                    or trackb_token_pair_output_group_stream_design_ready
                )
                else "route_resident_auto_after_token_pair_kblock_accumulator_stream_lane_s_gate"
                if trackb_token_pair_kblock_accumulator_stream_speed_lane_s_pass
                else "run_missing_token_pair_kblock_accumulator_stream_speed_shapes"
                if trackb_token_pair_kblock_accumulator_stream_speed_parity_pass
                else "change_token_pair_kblock_accumulator_stream_layout_or_kernel_family"
                if trackb_token_pair_kblock_accumulator_stream_speed_rejected
                else "run_token_pair_kblock_accumulator_stream_same_window_q2_speed_packet"
                if trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
                else "prove_token_pair_kblock_accumulator_stream_air_artifact_parity"
                if trackb_token_pair_kblock_accumulator_stream_native_parity_pass
                else "prove_token_pair_kblock_accumulator_stream_native_parity"
                if trackb_token_pair_kblock_accumulator_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_pair_kblock_accumulator_stream"
                if (
                    trackb_token_pair_kblock_accumulator_stream_source_guard_missing
                    or trackb_token_pair_kblock_accumulator_stream_design_ready
                )
                else "route_resident_auto_after_token_expert_output_block_stream_lane_s_gate"
                if trackb_token_expert_output_block_stream_speed_lane_s_pass
                else "run_missing_token_expert_output_block_stream_speed_shapes"
                if trackb_token_expert_output_block_stream_speed_parity_pass
                else "change_token_expert_output_block_stream_layout_or_kernel_family"
                if trackb_token_expert_output_block_stream_speed_rejected
                else "run_token_expert_output_block_stream_same_window_q2_speed_packet"
                if trackb_token_expert_output_block_stream_artifact_parity_pass
                else "prove_token_expert_output_block_stream_air_artifact_parity"
                if trackb_token_expert_output_block_stream_native_parity_pass
                else "prove_token_expert_output_block_stream_native_parity"
                if trackb_token_expert_output_block_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_expert_output_block_stream"
                if (
                    trackb_token_expert_output_block_stream_source_guard_missing
                    or trackb_token_expert_output_block_stream_design_ready
                )
                else "route_resident_auto_after_token_output_stripe_group_stream_lane_s_gate"
                if trackb_token_output_stripe_group_stream_speed_lane_s_pass
                else "run_missing_token_output_stripe_group_stream_speed_shapes"
                if trackb_token_output_stripe_group_stream_speed_parity_pass
                else "change_token_output_stripe_group_stream_layout_or_kernel_family"
                if trackb_token_output_stripe_group_stream_speed_rejected
                else "run_token_output_stripe_group_stream_same_window_q2_speed_packet"
                if trackb_token_output_stripe_group_stream_artifact_parity_pass
                else "prove_token_output_stripe_group_stream_air_artifact_parity"
                if trackb_token_output_stripe_group_stream_native_parity_pass
                else "prove_token_output_stripe_group_stream_native_parity"
                if trackb_token_output_stripe_group_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_output_stripe_group_stream"
                if (
                    trackb_token_output_stripe_group_stream_source_guard_missing
                    or trackb_token_output_stripe_group_stream_design_ready
                )
                else "route_resident_auto_after_token_block_output_group_stream_lane_s_gate"
                if trackb_token_block_output_group_stream_speed_lane_s_pass
                else "run_missing_token_block_output_group_stream_speed_shapes"
                if trackb_token_block_output_group_stream_speed_parity_pass
                else "change_token_block_output_group_stream_layout_or_kernel_family"
                if trackb_token_block_output_group_stream_speed_rejected
                else "run_token_block_output_group_stream_same_window_q2_speed_packet"
                if trackb_token_block_output_group_stream_artifact_parity_pass
                else "prove_token_block_output_group_stream_air_artifact_parity"
                if trackb_token_block_output_group_stream_native_parity_pass
                else "prove_token_block_output_group_stream_native_parity"
                if trackb_token_block_output_group_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_block_output_group_stream"
                if (
                    trackb_token_block_output_group_stream_source_guard_missing
                    or trackb_token_block_output_group_stream_design_ready
                )
                else "route_resident_auto_after_token_topk_output_tile_stream_lane_s_gate"
                if trackb_token_topk_output_tile_stream_speed_lane_s_pass
                else "run_missing_token_topk_output_tile_stream_speed_shapes"
                if trackb_token_topk_output_tile_stream_speed_parity_pass
                else "change_token_topk_output_tile_stream_layout_or_kernel_family"
                if trackb_token_topk_output_tile_stream_speed_rejected
                else "run_token_topk_output_tile_stream_same_window_q2_speed_packet"
                if trackb_token_topk_output_tile_stream_artifact_parity_pass
                else "prove_token_topk_output_tile_stream_air_artifact_parity"
                if trackb_token_topk_output_tile_stream_native_parity_pass
                else "prove_token_topk_output_tile_stream_native_parity"
                if trackb_token_topk_output_tile_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_topk_output_tile_stream"
                if (
                    trackb_token_topk_output_tile_stream_source_guard_missing
                    or trackb_token_topk_output_tile_stream_design_ready
                )
                else "route_resident_auto_after_route_tile_output_swizzle_stream_lane_s_gate"
                if trackb_route_tile_output_swizzle_stream_speed_lane_s_pass
                else "run_missing_route_tile_output_swizzle_stream_speed_shapes"
                if trackb_route_tile_output_swizzle_stream_speed_parity_pass
                else "change_route_tile_output_swizzle_stream_layout_or_kernel_family"
                if trackb_route_tile_output_swizzle_stream_speed_rejected
                else "run_route_tile_output_swizzle_stream_same_window_q2_speed_packet"
                if trackb_route_tile_output_swizzle_stream_artifact_parity_pass
                else "prove_route_tile_output_swizzle_stream_air_artifact_parity"
                if trackb_route_tile_output_swizzle_stream_native_parity_pass
                else "prove_route_tile_output_swizzle_stream_native_parity"
                if trackb_route_tile_output_swizzle_stream_source_guard_present
                else "implement_source_structure_guardrail_for_route_tile_output_swizzle_stream"
                if (
                    trackb_route_tile_output_swizzle_stream_source_guard_missing
                    or trackb_route_tile_output_swizzle_stream_design_ready
                )
                else "route_resident_auto_after_kblock_output_group_route_fused_stream_lane_s_gate"
                if trackb_kblock_output_group_route_fused_stream_speed_lane_s_pass
                else "run_missing_kblock_output_group_route_fused_stream_speed_shapes"
                if trackb_kblock_output_group_route_fused_stream_speed_parity_pass
                else "change_kblock_output_group_route_fused_stream_layout_or_kernel_family"
                if trackb_kblock_output_group_route_fused_stream_speed_rejected
                else "run_kblock_output_group_route_fused_stream_same_window_q2_speed_packet"
                if trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
                else "prove_kblock_output_group_route_fused_stream_air_artifact_parity"
                if trackb_kblock_output_group_route_fused_stream_native_parity_pass
                else "prove_kblock_output_group_route_fused_stream_native_parity"
                if trackb_kblock_output_group_route_fused_stream_source_guard_present
                else "implement_source_structure_guardrail_for_kblock_output_group_route_fused_stream"
                if trackb_kblock_output_group_route_fused_stream_design_ready
                else "route_resident_auto_after_output_group_pretransposed_codeword_stream_lane_s_gate"
                if trackb_output_group_pretransposed_codeword_stream_speed_lane_s_pass
                else "run_missing_output_group_pretransposed_codeword_stream_speed_shapes"
                if trackb_output_group_pretransposed_codeword_stream_speed_parity_pass
                else "change_output_group_pretransposed_codeword_stream_layout_or_kernel_family"
                if trackb_output_group_pretransposed_codeword_stream_speed_rejected
                else "run_output_group_pretransposed_codeword_stream_same_window_q2_speed_packet"
                if trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
                else "prove_output_group_pretransposed_codeword_stream_air_artifact_parity"
                if trackb_output_group_pretransposed_codeword_stream_native_parity_pass
                else "prove_output_group_pretransposed_codeword_stream_native_parity"
                if trackb_output_group_pretransposed_codeword_stream_source_guard_present
                else "implement_source_structure_guardrail_for_output_group_pretransposed_codeword_stream"
                if (
                    trackb_output_group_pretransposed_codeword_stream_source_guard_missing
                    or trackb_output_group_pretransposed_codeword_stream_design_ready
                )
                else "route_resident_auto_after_route_block_output_group_stream_lane_s_gate"
                if trackb_route_block_output_group_stream_speed_lane_s_pass
                else "run_missing_route_block_output_group_stream_speed_shapes"
                if trackb_route_block_output_group_stream_speed_parity_pass
                else "change_route_block_output_group_stream_layout_or_kernel_family"
                if trackb_route_block_output_group_stream_speed_rejected
                else "run_route_block_output_group_stream_same_window_q2_speed_packet"
                if trackb_route_block_output_group_stream_artifact_parity_pass
                else "prove_route_block_output_group_stream_air_artifact_parity"
                if trackb_route_block_output_group_stream_native_parity_pass
                else "prove_route_block_output_group_stream_native_parity"
                if trackb_route_block_output_group_stream_source_guard_present
                else "implement_source_structure_guardrail_for_route_block_output_group_stream"
                if (
                    trackb_route_block_output_group_stream_source_guard_missing
                    or trackb_route_block_output_group_stream_design_ready
                )
                else "route_resident_auto_after_scale_group_route_block_reduce_lane_s_gate"
                if trackb_scale_group_route_block_reduce_speed_lane_s_pass
                else "run_missing_scale_group_route_block_reduce_speed_shapes"
                if trackb_scale_group_route_block_reduce_speed_parity_pass
                else "change_scale_group_route_block_reduce_layout_or_kernel_family"
                if trackb_scale_group_route_block_reduce_speed_rejected
                else "run_scale_group_route_block_reduce_same_window_q2_speed_packet"
                if trackb_scale_group_route_block_reduce_artifact_parity_pass
                else "prove_scale_group_route_block_reduce_air_artifact_parity"
                if trackb_scale_group_route_block_reduce_native_parity_pass
                else "prove_scale_group_route_block_reduce_native_parity"
                if trackb_scale_group_route_block_reduce_source_guard_present
                else "implement_source_structure_guardrail_for_scale_group_route_block_reduce"
                if (
                    trackb_scale_group_route_block_reduce_source_guard_missing
                    or trackb_scale_group_route_block_reduce_design_ready
                )
                else "route_resident_auto_after_expert_kblock_scale_slot_stream_lane_s_gate"
                if trackb_expert_kblock_scale_slot_stream_speed_lane_s_pass
                else "run_missing_expert_kblock_scale_slot_stream_speed_shapes"
                if trackb_expert_kblock_scale_slot_stream_speed_parity_pass
                else "change_expert_kblock_scale_slot_stream_layout_or_kernel_family"
                if trackb_expert_kblock_scale_slot_stream_speed_rejected
                else "run_expert_kblock_scale_slot_stream_same_window_q2_speed_packet"
                if trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
                else "prove_expert_kblock_scale_slot_stream_air_artifact_parity"
                if trackb_expert_kblock_scale_slot_stream_native_parity_pass
                else "prove_expert_kblock_scale_slot_stream_native_parity"
                if trackb_expert_kblock_scale_slot_stream_source_guard_present
                else "implement_source_structure_guardrail_for_expert_kblock_scale_slot_stream"
                if (
                    trackb_expert_kblock_scale_slot_stream_source_guard_missing
                    or trackb_expert_kblock_scale_slot_stream_design_ready
                )
                else "route_resident_auto_after_token_route_output_stripe_pipeline_lane_s_gate"
                if trackb_token_route_output_stripe_pipeline_speed_lane_s_pass
                else "run_missing_token_route_output_stripe_pipeline_speed_shapes"
                if trackb_token_route_output_stripe_pipeline_speed_parity_pass
                else "change_token_route_output_stripe_pipeline_layout_or_kernel_family"
                if trackb_token_route_output_stripe_pipeline_speed_rejected
                else "run_token_route_output_stripe_pipeline_same_window_q2_speed_packet"
                if trackb_token_route_output_stripe_pipeline_artifact_parity_pass
                else "prove_token_route_output_stripe_pipeline_air_artifact_parity"
                if trackb_token_route_output_stripe_pipeline_native_parity_pass
                else "prove_token_route_output_stripe_pipeline_native_parity"
                if trackb_token_route_output_stripe_pipeline_source_guard_present
                else "implement_source_structure_guardrail_for_token_route_output_stripe_pipeline"
                if (
                    trackb_token_route_output_stripe_pipeline_source_guard_missing
                    or trackb_token_route_output_stripe_pipeline_design_ready
                )
                else "change_kblock_wavefront_codeword_scan_layout_or_kernel_family"
                if trackb_kblock_wavefront_codeword_scan_speed_rejected
                else "route_resident_auto_after_kblock_wavefront_codeword_scan_lane_s_gate"
                if trackb_kblock_wavefront_codeword_scan_speed_lane_s_pass
                else "run_kblock_wavefront_codeword_scan_same_window_q2_speed_packet"
                if trackb_kblock_wavefront_codeword_scan_artifact_parity_pass
                else "prove_kblock_wavefront_codeword_scan_air_artifact_parity"
                if trackb_kblock_wavefront_codeword_scan_native_parity_pass
                else "prove_kblock_wavefront_codeword_scan_native_parity"
                if trackb_kblock_wavefront_codeword_scan_source_guard_present
                else "implement_source_structure_guardrail_for_kblock_wavefront_codeword_scan"
                if (
                    trackb_kblock_wavefront_codeword_scan_source_guard_missing
                    or trackb_kblock_wavefront_codeword_scan_design_ready
                )
                else "trackb_circuit_broken_switch_to_w4_cache_cleanup_until_new_family_exists"
                if trackb_circuit_broken
                else "change_route_microtile_codeword_block_reduce_layout_or_kernel_family"
                if trackb_route_microtile_codeword_block_reduce_speed_rejected
                else "route_resident_auto_after_route_microtile_codeword_block_reduce_lane_s_gate"
                if trackb_route_microtile_codeword_block_reduce_speed_lane_s_pass
                else "run_route_microtile_codeword_block_reduce_same_window_q2_speed_packet"
                if trackb_route_microtile_codeword_block_reduce_artifact_parity_pass
                else "prove_route_microtile_codeword_block_reduce_air_artifact_parity"
                if trackb_route_microtile_codeword_block_reduce_native_parity_pass
                else "prove_route_microtile_codeword_block_reduce_native_parity"
                if trackb_route_microtile_codeword_block_reduce_source_guard_present
                else "implement_source_structure_guardrail_for_route_microtile_codeword_block_reduce"
                if (
                    trackb_route_microtile_codeword_block_reduce_source_guard_missing
                    or trackb_route_microtile_codeword_block_reduce_design_ready
                )
                else "change_output_tile_local_codeword_lut_layout_or_kernel_family"
                if trackb_output_tile_local_codeword_lut_speed_rejected
                else "route_resident_auto_after_output_tile_local_codeword_lut_lane_s_gate"
                if trackb_output_tile_local_codeword_lut_speed_lane_s_pass
                else "run_output_tile_local_codeword_lut_same_window_q2_speed_packet"
                if trackb_output_tile_local_codeword_lut_artifact_parity_pass
                else "prove_output_tile_local_codeword_lut_air_artifact_parity"
                if trackb_output_tile_local_codeword_lut_native_parity_pass
                else "prove_output_tile_local_codeword_lut_native_parity"
                if trackb_output_tile_local_codeword_lut_source_guard_present
                else "implement_source_structure_guardrail_for_output_tile_local_codeword_lut"
                if trackb_output_tile_local_codeword_lut_design_ready
                else "change_rowwise_codeword_tile_accumulate_layout_or_kernel_family"
                if trackb_rowwise_codeword_tile_accumulate_speed_rejected
                else "route_resident_auto_after_rowwise_codeword_tile_accumulate_lane_s_gate"
                if trackb_rowwise_codeword_tile_accumulate_speed_lane_s_pass
                else "run_rowwise_codeword_tile_accumulate_same_window_q2_speed_packet"
                if trackb_rowwise_codeword_tile_accumulate_artifact_parity_pass
                else "prove_rowwise_codeword_tile_accumulate_air_artifact_parity"
                if trackb_rowwise_codeword_tile_accumulate_native_parity_pass
                else "prove_rowwise_codeword_tile_accumulate_native_parity"
                if trackb_rowwise_codeword_tile_accumulate_source_guard_present
                else (
                    "implement_source_structure_guardrail_for_rowwise_codeword_tile_accumulate"
                )
                if trackb_rowwise_codeword_tile_accumulate_source_guard_missing
                else trackb_rowwise_codeword_tile_accumulate_design.get(
                    "next_track_b_action"
                )
                if trackb_rowwise_codeword_tile_accumulate_design_ready
                else "change_route_codeword_lut_accumulate_layout_or_kernel_family"
                if trackb_route_codeword_lut_accumulate_speed_rejected
                else "run_route_codeword_lut_accumulate_same_window_q2_speed_packet"
                if trackb_route_codeword_lut_accumulate_artifact_parity_pass
                else "prove_route_codeword_lut_accumulate_air_artifact_parity"
                if trackb_route_codeword_lut_accumulate_native_parity_pass
                else "prove_route_codeword_lut_accumulate_native_parity"
                if trackb_route_codeword_lut_accumulate_source_guard_present
                else (
                    "implement_source_structure_guardrail_for_route_codeword_lut_accumulate"
                )
                if trackb_route_codeword_lut_accumulate_source_guard_missing
                else trackb_route_codeword_lut_accumulate_design.get("next_track_b_action")
                if trackb_route_codeword_lut_accumulate_design_ready
                else trackb_next_family_gate.get("next_track_b_action")
                if trackb_materially_new_family_gate_active
                else "change_expert_kblock_codeword_factor_reuse_layout_or_kernel_family"
                if trackb_expert_kblock_codeword_factor_reuse_speed_rejected
                else "route_resident_auto_after_expert_kblock_lane_s_gate"
                if trackb_expert_kblock_codeword_factor_reuse_speed_lane_s_pass
                else "run_expert_kblock_codeword_factor_reuse_same_window_q2_speed_packet"
                if trackb_expert_kblock_codeword_factor_reuse_artifact_parity_pass
                else "prove_expert_kblock_codeword_factor_reuse_air_artifact_parity"
                if trackb_expert_kblock_codeword_factor_reuse_native_parity_pass
                else "prove_expert_kblock_codeword_factor_reuse_native_parity"
                if trackb_expert_kblock_codeword_factor_reuse_source_guard_present
                else (
                    "implement_source_structure_guardrail_for_expert_kblock_codeword_factor_reuse"
                )
                if trackb_expert_kblock_codeword_factor_reuse_source_guard_missing
                else trackb_expert_kblock_codeword_factor_reuse_design.get(
                    "next_track_b_action"
                )
                if trackb_expert_kblock_codeword_factor_reuse_design_ready
                else "change_input_stationary_codeword_tile_layout_or_kernel_family"
                if trackb_input_stationary_codeword_tile_speed_rejected
                else "route_resident_auto_after_input_stationary_lane_s_gate"
                if trackb_input_stationary_codeword_tile_speed_lane_s_pass
                else "run_input_stationary_codeword_tile_same_window_q2_speed_packet"
                if trackb_input_stationary_codeword_tile_artifact_parity_pass
                else "prove_input_stationary_codeword_tile_air_artifact_parity"
                if trackb_input_stationary_codeword_tile_native_parity_pass
                else "prove_input_stationary_codeword_tile_native_parity"
                if trackb_input_stationary_codeword_tile_source_guard_present
                else (
                    "implement_source_structure_guardrail_for_input_stationary_codeword_tile"
                )
                if trackb_input_stationary_codeword_tile_source_guard_missing
                else trackb_input_stationary_codeword_tile_design.get(
                    "next_track_b_action"
                )
                if trackb_input_stationary_codeword_tile_design_ready
                else "change_output_stationary_codeword_tile_layout_or_kernel_family"
                if trackb_output_stationary_codeword_tile_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_output_stationary_codeword_tile_speed_lane_s_pass
                else "run_output_stationary_codeword_tile_same_window_q2_speed_packet"
                if trackb_output_stationary_codeword_tile_artifact_parity_pass
                else "prove_output_stationary_codeword_tile_air_artifact_parity"
                if trackb_output_stationary_codeword_tile_native_parity_pass
                else "prove_output_stationary_codeword_tile_native_parity"
                if trackb_output_stationary_codeword_tile_source_guard_present
                else (
                    "implement_source_structure_guardrail_for_output_stationary_codeword_tile"
                )
                if trackb_output_stationary_codeword_tile_source_guard_missing
                else trackb_output_stationary_codeword_tile_design.get("next_track_b_action")
                if trackb_output_stationary_codeword_tile_design_ready
                else "change_token_cohort_mma_codeword_tile_layout_or_kernel_family"
                if trackb_token_cohort_mma_codeword_tile_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_token_cohort_mma_codeword_tile_speed_lane_s_pass
                else "run_token_cohort_mma_codeword_tile_same_window_q2_speed_packet"
                if trackb_token_cohort_mma_codeword_tile_artifact_parity_pass
                else "prove_token_cohort_mma_codeword_tile_air_artifact_parity"
                if trackb_token_cohort_mma_codeword_tile_native_parity_pass
                else "prove_token_cohort_mma_codeword_tile_native_parity"
                if trackb_token_cohort_mma_codeword_tile_source_guard_present
                else "implement_source_structure_guardrail_for_token_cohort_mma_codeword_tile"
                if trackb_token_cohort_mma_codeword_tile_source_guard_missing
                else trackb_token_cohort_mma_codeword_tile_design.get("next_track_b_action")
                if trackb_token_cohort_mma_codeword_tile_design_ready
                else "change_token_cohort_codeword_stream_layout_or_kernel_family"
                if trackb_token_cohort_codeword_stream_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_token_cohort_codeword_stream_speed_lane_s_pass
                else "run_token_cohort_codeword_stream_same_window_q2_speed_packet"
                if trackb_token_cohort_codeword_stream_artifact_parity_pass
                else "prove_token_cohort_codeword_stream_air_artifact_parity"
                if trackb_token_cohort_codeword_stream_native_parity_pass
                else "prove_token_cohort_codeword_stream_native_parity"
                if trackb_token_cohort_codeword_stream_source_guard_present
                else "implement_source_structure_guardrail_for_token_cohort_codeword_stream"
                if trackb_token_cohort_codeword_stream_design_ready
                else trackb_next_family_gate.get("next_track_b_action")
                if trackb_next_family_gate_blocks
                else "change_component_stream_tensorops_layout_or_kernel_family"
                if trackb_component_stream_tensorops_speed_rejected
                else "prove_component_stream_partial_reduction_air_artifact_parity"
                if trackb_component_stream_partial_reduction_native_parity_pass
                else "prove_component_stream_tensorops_partial_parity_before_speed"
                if trackb_component_stream_partial_reduction_tensorops_present
                else "prove_component_stream_partial_reduction_native_parity"
                if trackb_component_stream_partial_reduction_parallel_present
                else "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule"
                if trackb_component_stream_partial_reduction_source_guard_present
                else "implement_source_structure_guardrail_for_component_stream_partial_reduction"
                if trackb_component_stream_partial_reduction_source_guard_missing
                else "implement_source_structure_guardrail_for_component_stream_partial_reduction"
                if trackb_component_stream_partial_reduction_design_ready
                else "change_route_batch_segmented_codeword_reduce_layout_or_kernel_family"
                if trackb_route_batch_segmented_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_route_batch_segmented_speed_lane_s_pass
                else "run_route_batch_segmented_codeword_reduce_same_window_q2_speed_packet"
                if trackb_route_batch_segmented_artifact_parity_pass
                else "prove_route_batch_segmented_codeword_reduce_air_artifact_parity"
                if trackb_route_batch_segmented_native_parity_pass
                else "prove_route_batch_segmented_codeword_reduce_native_parity"
                if trackb_route_batch_segmented_source_guard_present
                else "implement_source_structure_guardrail_for_route_batch_segmented_codeword_reduce"
                if (
                    trackb_route_batch_segmented_source_guard_missing
                    or trackb_route_batch_segmented_design_ready
                )
                else "change_expert_cohort_codeword_broadcast_layout_or_kernel_family"
                if trackb_expert_cohort_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_expert_cohort_speed_lane_s_pass
                else "run_expert_cohort_codeword_broadcast_same_window_q2_speed_packet"
                if trackb_expert_cohort_artifact_parity_pass
                else "prove_expert_cohort_codeword_broadcast_air_artifact_parity"
                if trackb_expert_cohort_native_parity_pass
                else "prove_expert_cohort_codeword_broadcast_native_parity"
                if trackb_expert_cohort_source_guard_present
                else "implement_source_structure_guardrail_for_expert_cohort_codeword_broadcast"
                if trackb_expert_cohort_design_ready
                else "change_active_route_tile_codeword_outer_product_layout_or_kernel_family"
                if trackb_active_route_tile_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_active_route_tile_speed_lane_s_pass
                else "run_active_route_tile_codeword_outer_product_same_window_q2_speed_packet"
                if trackb_active_route_tile_artifact_parity_pass
                else "prove_active_route_tile_codeword_outer_product_air_artifact_parity"
                if trackb_active_route_tile_native_parity_pass
                else "prove_active_route_tile_codeword_outer_product_native_parity"
                if trackb_next_source_guard_present
                else trackb_next_design.get("next_track_b_action")
                if trackb_next_design_ready
                else "change_route_slot_mma_codeword_tile_layout_or_kernel_family"
                if trackb_successor_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_successor_speed_lane_s_pass
                else "run_route_slot_mma_codeword_tile_same_window_q2_speed_packet"
                if trackb_successor_artifact_parity_pass
                else "prove_route_slot_mma_codeword_tile_air_artifact_parity"
                if trackb_successor_native_parity_pass
                else "prove_route_slot_mma_codeword_tile_native_parity"
                if trackb_successor_source_guard_present
                else "implement_source_structure_guardrail_for_route_slot_mma_codeword_tile"
                if trackb_successor_source_guard_missing
                else trackb_successor_design.get("next_track_b_action")
                if trackb_successor_design_ready
                else "change_route_slot_codeword_stream_layout_or_kernel_family"
                if trackb_speed_rejected
                else "route_resident_auto_after_lane_s_gate"
                if trackb_speed_lane_s_pass
                else "run_route_slot_codeword_stream_same_window_q2_speed_packet"
                if trackb_artifact_parity_pass
                else "prove_route_slot_codeword_stream_air_artifact_parity"
                if trackb_native_parity_pass
                else "prove_route_slot_codeword_stream_native_parity"
                if trackb_source_guard_present
                else "implement_route_slot_codeword_stream_source_guardrail"
                if trackb_source_guard_missing
                else
                trackb_design.get("next_track_b_action")
                if trackb_design_ready
                else "design_materially_new_trackb_rhs_or_kernel_family"
            ),
        },
        "workstream3_qwen": {
            "status": qwen_front_status,
            "evidence": str(qwen_gate_path),
            "family_gate_pass": qwen_pass,
            "raw_family_gate_status": qwen.get("family_gate_status"),
            "family_gate_status": qwen_public_status,
            "hardened_family_gate_pass": qwen_hardened_pass,
            "public_family_gate_ready": qwen_hardened_pass,
            "hardening_requirements": qwen_hardening_requirements,
            "qwen_eval_required_row_count": _qwen_eval_required_row_count(qwen),
            "qwen_benchmark_reference": _qwen_benchmark_reference(qwen),
            "missing_requirements": qwen.get("missing_requirements", []),
        },
        "workstream3_glm52": {
            "status": (
                "source_artifact_gap_diagnosed"
                if glm52_gap_diagnosed
                else "needs_attention"
            ),
            "evidence": [str(glm52_layer3_path), str(glm52_layer77_path)],
            "source_artifact_diagnoses": glm52_diagnoses,
        },
        "workstream4_cache": {
            "status": (
                "single_host_cache_prefix_memory_killed"
                if single_host_cache_attempt_memory_killed
                else "single_host_source_memory_guard_blocked"
                if single_host_cache_attempt_guard_blocked
                else "single_host_cache_prefix_generated_dirty_memory"
                if single_host_cache_attempt_generated
                and single_host_cache_attempt_memory_clean is False
                else "single_host_cache_prefix_generated"
                if single_host_cache_attempt_generated
                else "local_runtime_source_ready_memory_risky"
                if cache_source_ready and cache_source_memory_risky
                else "local_single_host_source_ready"
                if cache_source_ready
                else "blocked_on_local_high_bit_source"
            ),
            "evidence": workstream4_evidence,
            "decision": cache_source.get("decision"),
            "rdma_topology": rdma_topology,
            "rank0_logits_layer_split_recommendation": layer_split_summary,
            "single_host_cache_attempt_decision": (
                single_host_cache_attempt.get("decision")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_attempt_exit_code": (
                single_host_cache_attempt.get("exit_code")
                if single_host_cache_attempt is not None
                else None
            ),
            "source_memory_guard_pass": (
                single_host_cache_attempt.get("source_memory_guard_pass")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_validation_ok": (
                single_host_cache_attempt.get("validation_ok")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_all_memory_clean": single_host_cache_attempt_memory_clean,
            "single_host_cache_lm_head_chunk_rows": (
                single_host_cache_attempt.get("lm_head_chunk_rows")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_stage_processes": (
                single_host_cache_attempt.get("local_sequential_stage_processes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_head_process": (
                single_host_cache_attempt.get("local_sequential_head_process")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_stage_view_plan_json": (
                single_host_cache_attempt.get("local_sequential_stage_view_plan_json")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_stage_view_roots_json": (
                single_host_cache_attempt.get("local_sequential_stage_view_roots_json")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_stage_view_roots": (
                single_host_cache_attempt.get("local_sequential_stage_view_roots")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_stage_view_source_visibility": (
                single_host_cache_attempt.get("stage_view_source_visibility")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_split_layer": (
                single_host_cache_attempt.get("local_sequential_lower_split_layer")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_split_layer": (
                single_host_cache_attempt.get("local_sequential_upper_split_layer")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_split_layers": (
                single_host_cache_attempt.get("local_sequential_lower_split_layers")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_split_layers": (
                single_host_cache_attempt.get("local_sequential_upper_split_layers")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_window_count": (
                single_host_cache_attempt.get("local_sequential_lower_window_count")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_window_count": (
                single_host_cache_attempt.get("local_sequential_upper_window_count")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_pre_hidden_host_spill": (
                single_host_cache_attempt.get("local_sequential_lower_pre_hidden_host_spill")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_pre_hidden_host_dtype": (
                single_host_cache_attempt.get("local_sequential_lower_pre_hidden_host_dtype")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_pre_hidden_host_shape": (
                single_host_cache_attempt.get("local_sequential_lower_pre_hidden_host_shape")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_pre_hidden_host_nbytes": (
                single_host_cache_attempt.get("local_sequential_lower_pre_hidden_host_nbytes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_hidden_host_spill": (
                single_host_cache_attempt.get("local_sequential_lower_hidden_host_spill")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_hidden_host_dtype": (
                single_host_cache_attempt.get("local_sequential_lower_hidden_host_dtype")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_hidden_host_shape": (
                single_host_cache_attempt.get("local_sequential_lower_hidden_host_shape")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_lower_hidden_host_nbytes": (
                single_host_cache_attempt.get("local_sequential_lower_hidden_host_nbytes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_pre_hidden_host_spill": (
                single_host_cache_attempt.get("local_sequential_upper_pre_hidden_host_spill")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_pre_hidden_host_dtype": (
                single_host_cache_attempt.get("local_sequential_upper_pre_hidden_host_dtype")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_pre_hidden_host_shape": (
                single_host_cache_attempt.get("local_sequential_upper_pre_hidden_host_shape")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_pre_hidden_host_nbytes": (
                single_host_cache_attempt.get("local_sequential_upper_pre_hidden_host_nbytes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_hidden_host_spill": (
                single_host_cache_attempt.get("local_sequential_upper_hidden_host_spill")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_hidden_host_dtype": (
                single_host_cache_attempt.get("local_sequential_upper_hidden_host_dtype")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_hidden_host_shape": (
                single_host_cache_attempt.get("local_sequential_upper_hidden_host_shape")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_upper_hidden_host_nbytes": (
                single_host_cache_attempt.get("local_sequential_upper_hidden_host_nbytes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_layer_split": (
                single_host_cache_attempt.get("layer_split")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_mlx_peak_bytes": (
                single_host_cache_attempt.get("mlx_peak_bytes")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_pageouts_delta": (
                single_host_cache_attempt.get("pageouts_delta")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_swapouts_delta": (
                single_host_cache_attempt.get("swapouts_delta")
                if single_host_cache_attempt is not None
                else None
            ),
            "single_host_cache_memory_delta_vs_previous_prefix": (
                single_host_cache_attempt.get("memory_delta_vs_previous_prefix")
                if single_host_cache_attempt is not None
                else None
            ),
            "source_ready_for_single_host_export": bool(
                cache_source.get("source_ready_for_single_host_export")
            ),
            "complete_source": cache_full_source_complete,
            "complete_runtime_source": cache_runtime_source_complete,
            "local_export_candidate_ready": cache_source_ready,
            "recommended_model_path": cache_source.get("recommended_model_path"),
            "missing_shards": cache_source.get("missing_shards", []),
            "missing_required_runtime_shards": cache_source.get(
                "missing_required_runtime_shards", []
            ),
            "missing_extra_shards": cache_source.get("missing_extra_shards", []),
            "source_bytes_exceed_physical_memory": cache_source_memory_risky,
            "expected_source_to_physical_memory_ratio": cache_memory.get(
                "expected_source_to_physical_memory_ratio"
            ),
            "cache_rows_generated": bool(
                cache_source.get("cache_rows_generated")
                or (
                    single_host_cache_attempt is not None
                    and single_host_cache_attempt.get("cache_rows_generated")
                )
            ),
        },
        "workstream5_hygiene": {
            "status": (
                "publication_packet_draft_present"
                if model_card_exists
                else "publication_packet_missing"
            ),
            "evidence": str(model_card_path),
            "draft_only": True,
        },
    }

    trackb_next_slice = (
        "route_trackb_resident_auto_after_token_pair_slot_topk_output_tile_fused_stream_lane_s_gate"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_lane_s_pass
        else "run_trackb_missing_token_pair_slot_topk_output_tile_fused_stream_speed_shapes"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_output_tile_fused_stream_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_speed_rejected
        else "run_trackb_token_pair_slot_topk_output_tile_fused_stream_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_output_tile_fused_stream_air_artifact_parity"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_source_guard_present
        else "implement_trackb_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
        if trackb_token_pair_slot_topk_output_tile_fused_stream_design_ready
        else "route_trackb_resident_auto_after_token_pair_slot_topk_kblock_microtile_stream_lane_s_gate"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_lane_s_pass
        else "run_trackb_missing_token_pair_slot_topk_kblock_microtile_stream_speed_shapes"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_kblock_microtile_stream_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_speed_rejected
        else "run_trackb_token_pair_slot_topk_kblock_microtile_stream_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_kblock_microtile_stream_air_artifact_parity"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity"
        if trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_present
        else "implement_trackb_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
        if (
            trackb_token_pair_slot_topk_kblock_microtile_stream_source_guard_missing
            or trackb_token_pair_slot_topk_kblock_microtile_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_slot_topk_route_bucket_codeword_reduce_lane_s_gate"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_lane_s_pass
        else "run_missing_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_shapes"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_rejected
        else "run_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_air_artifact_parity"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
        if trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_present
        else "implement_trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
        if (
            trackb_token_pair_slot_topk_route_bucket_codeword_reduce_source_guard_missing
            or trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_slot_topk_scale_slot_broadcast_stream_lane_s_gate"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_lane_s_pass
        else "run_missing_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_shapes"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_rejected
        else "run_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_air_artifact_parity"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
        if trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_present
        else "implement_trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
        if (
            trackb_token_pair_slot_topk_scale_slot_broadcast_stream_source_guard_missing
            or trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_slot_topk_codeword_group_pipeline_lane_s_gate"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_lane_s_pass
        else "run_missing_trackb_token_pair_slot_topk_codeword_group_pipeline_speed_shapes"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_codeword_group_pipeline_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_speed_rejected
        else "run_trackb_token_pair_slot_topk_codeword_group_pipeline_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_codeword_group_pipeline_air_artifact_parity"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity"
        if trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_present
        else "implement_trackb_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
        if (
            trackb_token_pair_slot_topk_codeword_group_pipeline_source_guard_missing
            or trackb_token_pair_slot_topk_codeword_group_pipeline_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_slot_topk_output_group_stream_lane_s_gate"
        if trackb_token_pair_slot_topk_output_group_stream_speed_lane_s_pass
        else "run_missing_trackb_token_pair_slot_topk_output_group_stream_speed_shapes"
        if trackb_token_pair_slot_topk_output_group_stream_speed_parity_pass
        else "change_trackb_token_pair_slot_topk_output_group_stream_layout_or_kernel_family"
        if trackb_token_pair_slot_topk_output_group_stream_speed_rejected
        else "run_trackb_token_pair_slot_topk_output_group_stream_same_window_q2_speed_packet"
        if trackb_token_pair_slot_topk_output_group_stream_artifact_parity_pass
        else "prove_trackb_token_pair_slot_topk_output_group_stream_air_artifact_parity"
        if trackb_token_pair_slot_topk_output_group_stream_native_parity_pass
        else "prove_trackb_token_pair_slot_topk_output_group_stream_native_parity"
        if trackb_token_pair_slot_topk_output_group_stream_source_guard_present
        else "implement_trackb_token_pair_slot_topk_output_group_stream_source_guardrail"
        if (
            trackb_token_pair_slot_topk_output_group_stream_source_guard_missing
            or trackb_token_pair_slot_topk_output_group_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_output_group_stream_lane_s_gate"
        if trackb_token_pair_output_group_stream_speed_lane_s_pass
        else "run_missing_trackb_token_pair_output_group_stream_speed_shapes"
        if trackb_token_pair_output_group_stream_speed_parity_pass
        else "change_trackb_token_pair_output_group_stream_layout_or_kernel_family"
        if trackb_token_pair_output_group_stream_speed_rejected
        else "run_trackb_token_pair_output_group_stream_same_window_q2_speed_packet"
        if trackb_token_pair_output_group_stream_artifact_parity_pass
        else "prove_trackb_token_pair_output_group_stream_air_artifact_parity"
        if trackb_token_pair_output_group_stream_native_parity_pass
        else "prove_trackb_token_pair_output_group_stream_native_parity"
        if trackb_token_pair_output_group_stream_source_guard_present
        else "implement_trackb_token_pair_output_group_stream_source_guardrail"
        if (
            trackb_token_pair_output_group_stream_source_guard_missing
            or trackb_token_pair_output_group_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_pair_kblock_accumulator_stream_lane_s_gate"
        if trackb_token_pair_kblock_accumulator_stream_speed_lane_s_pass
        else "run_missing_trackb_token_pair_kblock_accumulator_stream_speed_shapes"
        if trackb_token_pair_kblock_accumulator_stream_speed_parity_pass
        else "change_trackb_token_pair_kblock_accumulator_stream_layout_or_kernel_family"
        if trackb_token_pair_kblock_accumulator_stream_speed_rejected
        else "run_trackb_token_pair_kblock_accumulator_stream_same_window_q2_speed_packet"
        if trackb_token_pair_kblock_accumulator_stream_artifact_parity_pass
        else "prove_trackb_token_pair_kblock_accumulator_stream_air_artifact_parity"
        if trackb_token_pair_kblock_accumulator_stream_native_parity_pass
        else "prove_trackb_token_pair_kblock_accumulator_stream_native_parity"
        if trackb_token_pair_kblock_accumulator_stream_source_guard_present
        else "implement_trackb_token_pair_kblock_accumulator_stream_source_guardrail"
        if (
            trackb_token_pair_kblock_accumulator_stream_source_guard_missing
            or trackb_token_pair_kblock_accumulator_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_expert_output_block_stream_lane_s_gate"
        if trackb_token_expert_output_block_stream_speed_lane_s_pass
        else "run_missing_trackb_token_expert_output_block_stream_speed_shapes"
        if trackb_token_expert_output_block_stream_speed_parity_pass
        else "change_trackb_token_expert_output_block_stream_layout_or_kernel_family"
        if trackb_token_expert_output_block_stream_speed_rejected
        else "run_trackb_token_expert_output_block_stream_same_window_q2_speed_packet"
        if trackb_token_expert_output_block_stream_artifact_parity_pass
        else "prove_trackb_token_expert_output_block_stream_air_artifact_parity"
        if trackb_token_expert_output_block_stream_native_parity_pass
        else "prove_trackb_token_expert_output_block_stream_native_parity"
        if trackb_token_expert_output_block_stream_source_guard_present
        else "implement_trackb_token_expert_output_block_stream_source_guardrail"
        if (
            trackb_token_expert_output_block_stream_source_guard_missing
            or trackb_token_expert_output_block_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_output_stripe_group_stream_lane_s_gate"
        if trackb_token_output_stripe_group_stream_speed_lane_s_pass
        else "run_missing_trackb_token_output_stripe_group_stream_speed_shapes"
        if trackb_token_output_stripe_group_stream_speed_parity_pass
        else "change_trackb_token_output_stripe_group_stream_layout_or_kernel_family"
        if trackb_token_output_stripe_group_stream_speed_rejected
        else "run_trackb_token_output_stripe_group_stream_same_window_q2_speed_packet"
        if trackb_token_output_stripe_group_stream_artifact_parity_pass
        else "prove_trackb_token_output_stripe_group_stream_air_artifact_parity"
        if trackb_token_output_stripe_group_stream_native_parity_pass
        else "prove_trackb_token_output_stripe_group_stream_native_parity"
        if trackb_token_output_stripe_group_stream_source_guard_present
        else "implement_trackb_token_output_stripe_group_stream_source_guardrail"
        if (
            trackb_token_output_stripe_group_stream_source_guard_missing
            or trackb_token_output_stripe_group_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_block_output_group_stream_lane_s_gate"
        if trackb_token_block_output_group_stream_speed_lane_s_pass
        else "run_missing_trackb_token_block_output_group_stream_speed_shapes"
        if trackb_token_block_output_group_stream_speed_parity_pass
        else "change_trackb_token_block_output_group_stream_layout_or_kernel_family"
        if trackb_token_block_output_group_stream_speed_rejected
        else "run_trackb_token_block_output_group_stream_same_window_q2_speed_packet"
        if trackb_token_block_output_group_stream_artifact_parity_pass
        else "prove_trackb_token_block_output_group_stream_air_artifact_parity"
        if trackb_token_block_output_group_stream_native_parity_pass
        else "prove_trackb_token_block_output_group_stream_native_parity"
        if trackb_token_block_output_group_stream_source_guard_present
        else "implement_trackb_token_block_output_group_stream_source_guardrail"
        if (
            trackb_token_block_output_group_stream_source_guard_missing
            or trackb_token_block_output_group_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_topk_output_tile_stream_lane_s_gate"
        if trackb_token_topk_output_tile_stream_speed_lane_s_pass
        else "run_missing_trackb_token_topk_output_tile_stream_speed_shapes"
        if trackb_token_topk_output_tile_stream_speed_parity_pass
        else "change_trackb_token_topk_output_tile_stream_layout_or_kernel_family"
        if trackb_token_topk_output_tile_stream_speed_rejected
        else "run_trackb_token_topk_output_tile_stream_same_window_q2_speed_packet"
        if trackb_token_topk_output_tile_stream_artifact_parity_pass
        else "prove_trackb_token_topk_output_tile_stream_air_artifact_parity"
        if trackb_token_topk_output_tile_stream_native_parity_pass
        else "prove_trackb_token_topk_output_tile_stream_native_parity"
        if trackb_token_topk_output_tile_stream_source_guard_present
        else "implement_trackb_token_topk_output_tile_stream_source_guardrail"
        if (
            trackb_token_topk_output_tile_stream_source_guard_missing
            or trackb_token_topk_output_tile_stream_design_ready
        )
        else "route_trackb_resident_auto_after_route_tile_output_swizzle_stream_lane_s_gate"
        if trackb_route_tile_output_swizzle_stream_speed_lane_s_pass
        else "run_missing_trackb_route_tile_output_swizzle_stream_speed_shapes"
        if trackb_route_tile_output_swizzle_stream_speed_parity_pass
        else "change_trackb_route_tile_output_swizzle_stream_layout_or_kernel_family"
        if trackb_route_tile_output_swizzle_stream_speed_rejected
        else "run_trackb_route_tile_output_swizzle_stream_same_window_q2_speed_packet"
        if trackb_route_tile_output_swizzle_stream_artifact_parity_pass
        else "prove_trackb_route_tile_output_swizzle_stream_air_artifact_parity"
        if trackb_route_tile_output_swizzle_stream_native_parity_pass
        else "prove_trackb_route_tile_output_swizzle_stream_native_parity"
        if trackb_route_tile_output_swizzle_stream_source_guard_present
        else "implement_trackb_route_tile_output_swizzle_stream_source_guardrail"
        if (
            trackb_route_tile_output_swizzle_stream_source_guard_missing
            or trackb_route_tile_output_swizzle_stream_design_ready
        )
        else "route_trackb_resident_auto_after_kblock_output_group_route_fused_stream_lane_s_gate"
        if trackb_kblock_output_group_route_fused_stream_speed_lane_s_pass
        else "run_missing_trackb_kblock_output_group_route_fused_stream_speed_shapes"
        if trackb_kblock_output_group_route_fused_stream_speed_parity_pass
        else "change_trackb_kblock_output_group_route_fused_stream_layout_or_kernel_family"
        if trackb_kblock_output_group_route_fused_stream_speed_rejected
        else "run_trackb_kblock_output_group_route_fused_stream_same_window_q2_speed_packet"
        if trackb_kblock_output_group_route_fused_stream_artifact_parity_pass
        else "prove_trackb_kblock_output_group_route_fused_stream_air_artifact_parity"
        if trackb_kblock_output_group_route_fused_stream_native_parity_pass
        else "prove_trackb_kblock_output_group_route_fused_stream_native_parity"
        if trackb_kblock_output_group_route_fused_stream_source_guard_present
        else "implement_trackb_kblock_output_group_route_fused_stream_source_guardrail"
        if trackb_kblock_output_group_route_fused_stream_design_ready
        else "route_trackb_resident_auto_after_output_group_pretransposed_codeword_stream_lane_s_gate"
        if trackb_output_group_pretransposed_codeword_stream_speed_lane_s_pass
        else "run_missing_trackb_output_group_pretransposed_codeword_stream_speed_shapes"
        if trackb_output_group_pretransposed_codeword_stream_speed_parity_pass
        else "change_trackb_output_group_pretransposed_codeword_stream_layout_or_kernel_family"
        if trackb_output_group_pretransposed_codeword_stream_speed_rejected
        else "run_trackb_output_group_pretransposed_codeword_stream_same_window_q2_speed_packet"
        if trackb_output_group_pretransposed_codeword_stream_artifact_parity_pass
        else "prove_trackb_output_group_pretransposed_codeword_stream_air_artifact_parity"
        if trackb_output_group_pretransposed_codeword_stream_native_parity_pass
        else "prove_trackb_output_group_pretransposed_codeword_stream_native_parity"
        if trackb_output_group_pretransposed_codeword_stream_source_guard_present
        else "implement_trackb_output_group_pretransposed_codeword_stream_source_guardrail"
        if (
            trackb_output_group_pretransposed_codeword_stream_source_guard_missing
            or trackb_output_group_pretransposed_codeword_stream_design_ready
        )
        else "route_trackb_resident_auto_after_route_block_output_group_stream_lane_s_gate"
        if trackb_route_block_output_group_stream_speed_lane_s_pass
        else "run_missing_trackb_route_block_output_group_stream_speed_shapes"
        if trackb_route_block_output_group_stream_speed_parity_pass
        else "change_trackb_route_block_output_group_stream_layout_or_kernel_family"
        if trackb_route_block_output_group_stream_speed_rejected
        else "run_trackb_route_block_output_group_stream_same_window_q2_speed_packet"
        if trackb_route_block_output_group_stream_artifact_parity_pass
        else "prove_trackb_route_block_output_group_stream_air_artifact_parity"
        if trackb_route_block_output_group_stream_native_parity_pass
        else "prove_trackb_route_block_output_group_stream_native_parity"
        if trackb_route_block_output_group_stream_source_guard_present
        else "implement_trackb_route_block_output_group_stream_source_guardrail"
        if (
            trackb_route_block_output_group_stream_source_guard_missing
            or trackb_route_block_output_group_stream_design_ready
        )
        else "route_trackb_resident_auto_after_scale_group_route_block_reduce_lane_s_gate"
        if trackb_scale_group_route_block_reduce_speed_lane_s_pass
        else "run_missing_trackb_scale_group_route_block_reduce_speed_shapes"
        if trackb_scale_group_route_block_reduce_speed_parity_pass
        else "change_trackb_scale_group_route_block_reduce_layout_or_kernel_family"
        if trackb_scale_group_route_block_reduce_speed_rejected
        else "run_trackb_scale_group_route_block_reduce_same_window_q2_speed_packet"
        if trackb_scale_group_route_block_reduce_artifact_parity_pass
        else "prove_trackb_scale_group_route_block_reduce_air_artifact_parity"
        if trackb_scale_group_route_block_reduce_native_parity_pass
        else "prove_trackb_scale_group_route_block_reduce_native_parity"
        if trackb_scale_group_route_block_reduce_source_guard_present
        else "implement_trackb_scale_group_route_block_reduce_source_guardrail"
        if (
            trackb_scale_group_route_block_reduce_source_guard_missing
            or trackb_scale_group_route_block_reduce_design_ready
        )
        else "route_trackb_resident_auto_after_expert_kblock_scale_slot_stream_lane_s_gate"
        if trackb_expert_kblock_scale_slot_stream_speed_lane_s_pass
        else "run_missing_trackb_expert_kblock_scale_slot_stream_speed_shapes"
        if trackb_expert_kblock_scale_slot_stream_speed_parity_pass
        else "change_trackb_expert_kblock_scale_slot_stream_layout_or_kernel_family"
        if trackb_expert_kblock_scale_slot_stream_speed_rejected
        else "run_trackb_expert_kblock_scale_slot_stream_same_window_q2_speed_packet"
        if trackb_expert_kblock_scale_slot_stream_artifact_parity_pass
        else "prove_trackb_expert_kblock_scale_slot_stream_air_artifact_parity"
        if trackb_expert_kblock_scale_slot_stream_native_parity_pass
        else "prove_trackb_expert_kblock_scale_slot_stream_native_parity"
        if trackb_expert_kblock_scale_slot_stream_source_guard_present
        else "implement_trackb_expert_kblock_scale_slot_stream_source_guardrail"
        if (
            trackb_expert_kblock_scale_slot_stream_source_guard_missing
            or trackb_expert_kblock_scale_slot_stream_design_ready
        )
        else "route_trackb_resident_auto_after_token_route_output_stripe_pipeline_lane_s_gate"
        if trackb_token_route_output_stripe_pipeline_speed_lane_s_pass
        else "run_trackb_missing_token_route_output_stripe_pipeline_speed_shapes"
        if trackb_token_route_output_stripe_pipeline_speed_parity_pass
        else "change_trackb_token_route_output_stripe_pipeline_layout_or_kernel_family"
        if trackb_token_route_output_stripe_pipeline_speed_rejected
        else "run_trackb_token_route_output_stripe_pipeline_same_window_q2_speed_packet"
        if trackb_token_route_output_stripe_pipeline_artifact_parity_pass
        else "prove_trackb_token_route_output_stripe_pipeline_air_artifact_parity"
        if trackb_token_route_output_stripe_pipeline_native_parity_pass
        else "prove_trackb_token_route_output_stripe_pipeline_native_parity"
        if trackb_token_route_output_stripe_pipeline_source_guard_present
        else "implement_trackb_token_route_output_stripe_pipeline_source_guardrail"
        if (
            trackb_token_route_output_stripe_pipeline_source_guard_missing
            or trackb_token_route_output_stripe_pipeline_design_ready
        )
        else "change_trackb_kblock_wavefront_codeword_scan_layout_or_kernel_family"
        if trackb_kblock_wavefront_codeword_scan_speed_rejected
        else "route_trackb_resident_auto_after_kblock_wavefront_codeword_scan_lane_s_gate"
        if trackb_kblock_wavefront_codeword_scan_speed_lane_s_pass
        else "run_trackb_kblock_wavefront_codeword_scan_same_window_q2_speed_packet"
        if trackb_kblock_wavefront_codeword_scan_artifact_parity_pass
        else "prove_trackb_kblock_wavefront_codeword_scan_air_artifact_parity"
        if trackb_kblock_wavefront_codeword_scan_native_parity_pass
        else "prove_trackb_kblock_wavefront_codeword_scan_native_parity"
        if trackb_kblock_wavefront_codeword_scan_source_guard_present
        else "implement_trackb_kblock_wavefront_codeword_scan_source_guardrail"
        if (
            trackb_kblock_wavefront_codeword_scan_source_guard_missing
            or trackb_kblock_wavefront_codeword_scan_design_ready
        )
        else "trackb_circuit_broken_wait_for_materially_new_rhs_or_kernel_family"
        if trackb_circuit_broken
        else "change_trackb_route_microtile_codeword_block_reduce_layout_or_kernel_family"
        if trackb_route_microtile_codeword_block_reduce_speed_rejected
        else "route_trackb_resident_auto_after_route_microtile_codeword_block_reduce_lane_s_gate"
        if trackb_route_microtile_codeword_block_reduce_speed_lane_s_pass
        else "run_trackb_route_microtile_codeword_block_reduce_same_window_q2_speed_packet"
        if trackb_route_microtile_codeword_block_reduce_artifact_parity_pass
        else "prove_trackb_route_microtile_codeword_block_reduce_air_artifact_parity"
        if trackb_route_microtile_codeword_block_reduce_native_parity_pass
        else "prove_trackb_route_microtile_codeword_block_reduce_native_parity"
        if trackb_route_microtile_codeword_block_reduce_source_guard_present
        else "implement_trackb_route_microtile_codeword_block_reduce_source_guardrail"
        if (
            trackb_route_microtile_codeword_block_reduce_source_guard_missing
            or trackb_route_microtile_codeword_block_reduce_design_ready
        )
        else "change_trackb_output_tile_local_codeword_lut_layout_or_kernel_family"
        if trackb_output_tile_local_codeword_lut_speed_rejected
        else "route_trackb_resident_auto_after_output_tile_local_codeword_lut_lane_s_gate"
        if trackb_output_tile_local_codeword_lut_speed_lane_s_pass
        else "run_trackb_output_tile_local_codeword_lut_same_window_q2_speed_packet"
        if trackb_output_tile_local_codeword_lut_artifact_parity_pass
        else "prove_trackb_output_tile_local_codeword_lut_air_artifact_parity"
        if trackb_output_tile_local_codeword_lut_native_parity_pass
        else "prove_trackb_output_tile_local_codeword_lut_native_parity"
        if trackb_output_tile_local_codeword_lut_source_guard_present
        else "implement_trackb_output_tile_local_codeword_lut_source_guardrail"
        if trackb_output_tile_local_codeword_lut_design_ready
        else "change_trackb_rowwise_codeword_tile_accumulate_layout_or_kernel_family"
        if trackb_rowwise_codeword_tile_accumulate_speed_rejected
        else "route_trackb_resident_auto_after_rowwise_codeword_tile_accumulate_lane_s_gate"
        if trackb_rowwise_codeword_tile_accumulate_speed_lane_s_pass
        else "run_trackb_rowwise_codeword_tile_accumulate_same_window_q2_speed_packet"
        if trackb_rowwise_codeword_tile_accumulate_artifact_parity_pass
        else "prove_trackb_rowwise_codeword_tile_accumulate_air_artifact_parity"
        if trackb_rowwise_codeword_tile_accumulate_native_parity_pass
        else "prove_trackb_rowwise_codeword_tile_accumulate_native_parity"
        if trackb_rowwise_codeword_tile_accumulate_source_guard_present
        else "implement_trackb_rowwise_codeword_tile_accumulate_source_guardrail"
        if (
            trackb_rowwise_codeword_tile_accumulate_source_guard_missing
            or trackb_rowwise_codeword_tile_accumulate_design_ready
        )
        else "change_trackb_route_codeword_lut_accumulate_layout_or_kernel_family"
        if trackb_route_codeword_lut_accumulate_speed_rejected
        else "run_trackb_route_codeword_lut_accumulate_same_window_q2_speed_packet"
        if trackb_route_codeword_lut_accumulate_artifact_parity_pass
        else "prove_trackb_route_codeword_lut_accumulate_air_artifact_parity"
        if trackb_route_codeword_lut_accumulate_native_parity_pass
        else "prove_trackb_route_codeword_lut_accumulate_native_parity"
        if trackb_route_codeword_lut_accumulate_source_guard_present
        else "implement_trackb_route_codeword_lut_accumulate_source_guardrail"
        if (
            trackb_route_codeword_lut_accumulate_source_guard_missing
            or trackb_route_codeword_lut_accumulate_design_ready
        )
        else "write_materially_new_trackb_rhs_or_kernel_family_design"
        if trackb_materially_new_family_gate_active
        else "change_trackb_expert_kblock_codeword_factor_reuse_layout_or_kernel_family"
        if trackb_expert_kblock_codeword_factor_reuse_speed_rejected
        else "route_trackb_expert_kblock_resident_auto_after_lane_s_gate"
        if trackb_expert_kblock_codeword_factor_reuse_speed_lane_s_pass
        else "run_trackb_expert_kblock_codeword_factor_reuse_same_window_q2_speed_packet"
        if trackb_expert_kblock_codeword_factor_reuse_artifact_parity_pass
        else "prove_trackb_expert_kblock_codeword_factor_reuse_air_artifact_parity"
        if trackb_expert_kblock_codeword_factor_reuse_native_parity_pass
        else "prove_trackb_expert_kblock_codeword_factor_reuse_native_parity"
        if trackb_expert_kblock_codeword_factor_reuse_source_guard_present
        else "implement_trackb_expert_kblock_codeword_factor_reuse_source_guardrail"
        if trackb_expert_kblock_codeword_factor_reuse_design_ready
        else "change_trackb_input_stationary_codeword_tile_layout_or_kernel_family"
        if trackb_input_stationary_codeword_tile_speed_rejected
        else "route_trackb_input_stationary_codeword_tile_resident_auto_after_lane_s_gate"
        if trackb_input_stationary_codeword_tile_speed_lane_s_pass
        else "run_trackb_input_stationary_codeword_tile_same_window_q2_speed_packet"
        if trackb_input_stationary_codeword_tile_artifact_parity_pass
        else "prove_trackb_input_stationary_codeword_tile_air_artifact_parity"
        if trackb_input_stationary_codeword_tile_native_parity_pass
        else "prove_trackb_input_stationary_codeword_tile_native_parity"
        if trackb_input_stationary_codeword_tile_source_guard_present
        else "implement_trackb_input_stationary_codeword_tile_source_guardrail"
        if trackb_input_stationary_codeword_tile_source_guard_missing
        else "implement_trackb_input_stationary_codeword_tile_source_guardrail"
        if trackb_input_stationary_codeword_tile_design_ready
        else "change_trackb_output_stationary_codeword_tile_layout_or_kernel_family"
        if trackb_output_stationary_codeword_tile_speed_rejected
        else "route_trackb_output_stationary_resident_auto_after_lane_s_gate"
        if trackb_output_stationary_codeword_tile_speed_lane_s_pass
        else "run_trackb_output_stationary_codeword_tile_same_window_q2_speed_packet"
        if trackb_output_stationary_codeword_tile_artifact_parity_pass
        else "prove_trackb_output_stationary_codeword_tile_air_artifact_parity"
        if trackb_output_stationary_codeword_tile_native_parity_pass
        else "prove_trackb_output_stationary_codeword_tile_native_parity"
        if trackb_output_stationary_codeword_tile_source_guard_present
        else "implement_trackb_output_stationary_codeword_tile_source_guardrail"
        if trackb_output_stationary_codeword_tile_design_ready
        else "change_trackb_token_cohort_mma_codeword_tile_layout_or_kernel_family"
        if trackb_token_cohort_mma_codeword_tile_speed_rejected
        else "route_trackb_token_cohort_mma_resident_auto_after_lane_s_gate"
        if trackb_token_cohort_mma_codeword_tile_speed_lane_s_pass
        else "run_trackb_token_cohort_mma_codeword_tile_same_window_q2_speed_packet"
        if trackb_token_cohort_mma_codeword_tile_artifact_parity_pass
        else "prove_trackb_token_cohort_mma_codeword_tile_air_artifact_parity"
        if trackb_token_cohort_mma_codeword_tile_native_parity_pass
        else "prove_trackb_token_cohort_mma_codeword_tile_native_parity"
        if trackb_token_cohort_mma_codeword_tile_source_guard_present
        else "implement_trackb_token_cohort_mma_codeword_tile_source_guardrail"
        if trackb_token_cohort_mma_codeword_tile_design_ready
        else "change_trackb_token_cohort_codeword_stream_layout_or_kernel_family"
        if trackb_token_cohort_codeword_stream_speed_rejected
        else "route_trackb_token_cohort_resident_auto_after_lane_s_gate"
        if trackb_token_cohort_codeword_stream_speed_lane_s_pass
        else "run_trackb_token_cohort_codeword_stream_same_window_q2_speed_packet"
        if trackb_token_cohort_codeword_stream_artifact_parity_pass
        else "prove_trackb_token_cohort_codeword_stream_air_artifact_parity"
        if trackb_token_cohort_codeword_stream_native_parity_pass
        else "prove_trackb_token_cohort_codeword_stream_native_parity"
        if trackb_token_cohort_codeword_stream_source_guard_present
        else "implement_trackb_token_cohort_codeword_stream_source_guardrail"
        if trackb_token_cohort_codeword_stream_design_ready
        else "write_materially_new_trackb_rhs_or_kernel_family_design"
        if trackb_next_family_gate_blocks
        else "change_trackb_component_stream_tensorops_layout_or_kernel_family"
        if trackb_component_stream_tensorops_speed_rejected
        else "prove_trackb_component_stream_partial_reduction_air_artifact_parity"
        if trackb_component_stream_partial_reduction_native_parity_pass
        else "prove_trackb_component_stream_tensorops_partial_parity_before_speed"
        if trackb_component_stream_partial_reduction_tensorops_present
        else "prove_trackb_component_stream_partial_reduction_native_parity"
        if trackb_component_stream_partial_reduction_parallel_present
        else "replace_trackb_component_stream_partial_reduction_scalar_body"
        if trackb_component_stream_partial_reduction_source_guard_present
        else "implement_trackb_component_stream_partial_reduction_source_guardrail"
        if trackb_component_stream_partial_reduction_source_guard_missing
        else "implement_trackb_component_stream_partial_reduction_source_guardrail"
        if trackb_component_stream_partial_reduction_design_ready
        else "change_trackb_route_batch_segmented_codeword_reduce_layout_or_kernel_family"
        if trackb_route_batch_segmented_speed_rejected
        else "route_trackb_route_batch_segmented_codeword_reduce_resident_auto_after_lane_s_gate"
        if trackb_route_batch_segmented_speed_lane_s_pass
        else "run_trackb_route_batch_segmented_codeword_reduce_same_window_q2_speed_packet"
        if trackb_route_batch_segmented_artifact_parity_pass
        else "prove_trackb_route_batch_segmented_codeword_reduce_air_artifact_parity"
        if trackb_route_batch_segmented_native_parity_pass
        else "prove_trackb_route_batch_segmented_codeword_reduce_native_parity"
        if trackb_route_batch_segmented_source_guard_present
        else "implement_trackb_route_batch_segmented_codeword_reduce_source_guardrail"
        if (
            trackb_route_batch_segmented_source_guard_missing
            or trackb_route_batch_segmented_design_ready
        )
        else "change_trackb_expert_cohort_codeword_broadcast_layout_or_kernel_family"
        if trackb_expert_cohort_speed_rejected
        else "route_trackb_expert_cohort_codeword_broadcast_resident_auto_after_lane_s_gate"
        if trackb_expert_cohort_speed_lane_s_pass
        else "run_trackb_expert_cohort_codeword_broadcast_same_window_q2_speed_packet"
        if trackb_expert_cohort_artifact_parity_pass
        else "prove_trackb_expert_cohort_codeword_broadcast_air_artifact_parity"
        if trackb_expert_cohort_native_parity_pass
        else "prove_trackb_expert_cohort_codeword_broadcast_native_parity"
        if trackb_expert_cohort_source_guard_present
        else "implement_trackb_expert_cohort_codeword_broadcast_source_guardrail"
        if trackb_expert_cohort_design_ready
        else "change_trackb_active_route_tile_codeword_outer_product_layout_or_kernel_family"
        if trackb_active_route_tile_speed_rejected
        else "route_trackb_active_route_tile_codeword_outer_product_resident_auto_after_lane_s_gate"
        if trackb_active_route_tile_speed_lane_s_pass
        else "run_trackb_active_route_tile_codeword_outer_product_same_window_q2_speed_packet"
        if trackb_active_route_tile_artifact_parity_pass
        else "prove_trackb_active_route_tile_codeword_outer_product_air_artifact_parity"
        if trackb_active_route_tile_native_parity_pass
        else "prove_trackb_active_route_tile_codeword_outer_product_native_parity"
        if trackb_next_source_guard_present
        else "implement_trackb_active_route_tile_codeword_outer_product_source_guardrail"
        if trackb_next_design_ready
        else "change_trackb_route_slot_mma_codeword_tile_layout_or_kernel_family"
        if trackb_successor_speed_rejected
        else "route_trackb_route_slot_mma_codeword_tile_resident_auto_after_lane_s_gate"
        if trackb_successor_speed_lane_s_pass
        else "run_trackb_route_slot_mma_codeword_tile_same_window_q2_speed_packet"
        if trackb_successor_artifact_parity_pass
        else "prove_trackb_route_slot_mma_codeword_tile_air_artifact_parity"
        if trackb_successor_native_parity_pass
        else "prove_trackb_route_slot_mma_codeword_tile_native_parity"
        if trackb_successor_source_guard_present
        else "implement_trackb_route_slot_mma_codeword_tile_source_guardrail"
        if trackb_successor_design_ready
        else "change_trackb_route_slot_codeword_stream_layout_or_kernel_family"
        if trackb_speed_rejected
        else "route_trackb_route_slot_codeword_stream_resident_auto_after_lane_s_gate"
        if trackb_speed_lane_s_pass
        else "run_trackb_route_slot_codeword_stream_same_window_q2_speed_packet"
        if trackb_artifact_parity_pass
        else "prove_trackb_route_slot_codeword_stream_air_artifact_parity"
        if trackb_native_parity_pass
        else "prove_trackb_route_slot_codeword_stream_native_parity"
        if trackb_source_guard_present
        else "implement_trackb_route_slot_codeword_stream_source_guardrail"
        if trackb_source_guard_missing
        else
        "implement_trackb_route_slot_codeword_stream_source_guardrail"
        if trackb_design_ready
        else "design_materially_new_trackb_rhs_or_kernel_family"
    )
    cache_next_slice = (
        "reduce_single_host_cache_loader_peak_or_restore_distributed_cache_path"
        if single_host_cache_attempt_memory_killed
        else "restore_distributed_cache_path_or_implement_streaming_single_host_source_loader"
        if single_host_cache_attempt_guard_blocked
        else "clean_single_host_cache_prefix_memory_or_restore_distributed_cache_path"
        if single_host_cache_attempt_generated
        and single_host_cache_attempt_memory_clean is False
        else "extend_single_host_cache_prefix_or_restore_distributed_cache_path"
        if single_host_cache_attempt_generated
        else "run_bounded_single_host_cache_prefix_with_lazy_skip_consume"
        if cache_source_ready and cache_source_memory_risky
        else "run_single_host_cache_prefix"
        if cache_source_ready
        else "repair_or_supply_complete_local_high_bit_teacher_source_for_single_host_cache"
    )
    cache_should_preempt_trackb = (
        single_host_cache_attempt_memory_killed
        or single_host_cache_attempt_guard_blocked
        or (
            cache_source_ready
            and cache_source_memory_risky
            and not single_host_cache_attempt_generated
        )
    )
    next_local_slices = (
        [
            cache_next_slice,
            "refresh_publication_draft_with_latest_trackb_and_local_source_scan",
            "design_materially_new_trackb_rhs_or_kernel_family",
        ]
        if trackb_circuit_broken
        else (
        [
            cache_next_slice,
            trackb_next_slice,
            "refresh_publication_draft_with_latest_trackb_and_local_source_scan",
        ]
        if cache_should_preempt_trackb
        else [
            trackb_next_slice,
            cache_next_slice,
            "refresh_publication_draft_with_latest_trackb_and_local_source_scan",
        ]
        )
    )
    if qwen_pass and qwen_has_explicit_thresholds and not qwen_hardened_pass:
        next_local_slices.insert(
            0,
            "harden_qwen_family_gate_64row_source_teacher_eval_and_reference_benchmark",
        )
    elif not qwen_pass:
        next_local_slices.insert(0, "repair_qwen_first_family_gate")
    if not glm52_gap_diagnosed:
        next_local_slices.insert(0, "finish_glm52_source_artifact_root_cause")

    cache_completion_blocker = (
        "single_host_teacher_cache_prefix_memory_killed"
        if single_host_cache_attempt_memory_killed
        else "single_host_teacher_cache_source_memory_guard_blocked"
        if single_host_cache_attempt_guard_blocked
        else "single_host_teacher_cache_prefix_memory_dirty"
        if single_host_cache_attempt_generated
        and single_host_cache_attempt_memory_clean is False
        else None
        if single_host_cache_attempt_generated
        else "distributed_or_single_host_teacher_cache_not_generated"
    )
    completion_blockers = [
        "p1_quality_not_promotable",
        "trackb_native_speed_path_missing",
        "publication_packet_draft_only",
    ]
    if cache_completion_blocker is not None:
        completion_blockers.insert(2, cache_completion_blocker)
    transport_boundary = (
        "peer2_read_only_topology_checked_direct_path_ready"
        if rdma_topology is not None and rdma_topology.get("direct_path_ready")
        else "peer2_read_only_topology_checked_direct_path_down"
        if rdma_topology is not None
        else "peer2_approved_not_used_in_this_slice"
    )

    return {
        "schema_version": 1,
        "record_type": "keep_five_front_current_status",
        "peer2_used": transport["peer2_used"],
        "rdma_jaccl_touched": transport["rdma_jaccl_touched"],
        "transport_boundary": transport_boundary,
        "mission_complete": False,
        "mission_status": "active_not_complete",
        "fronts": fronts,
        "next_local_slices": next_local_slices,
        "completion_blockers": completion_blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize current KEEP Five-Front Push status from local evidence."
    )
    parser.add_argument("--workstream1-json", type=Path, default=DEFAULT_WORKSTREAM1)
    parser.add_argument(
        "--workstream1-fixed-bump-json",
        type=Path,
        default=DEFAULT_WORKSTREAM1_FIXED_BUMP,
    )
    parser.add_argument("--trackb-json", type=Path, default=DEFAULT_TRACKB)
    parser.add_argument("--trackb-design-json", type=Path, default=DEFAULT_TRACKB_DESIGN)
    parser.add_argument(
        "--trackb-source-guard-json",
        type=Path,
        default=DEFAULT_TRACKB_SOURCE_GUARD,
    )
    parser.add_argument(
        "--trackb-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-successor-design-json",
        type=Path,
        default=DEFAULT_TRACKB_SUCCESSOR_DESIGN,
    )
    parser.add_argument(
        "--trackb-successor-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_SUCCESSOR_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-successor-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_SUCCESSOR_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-successor-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_SUCCESSOR_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-next-design-json",
        type=Path,
        default=DEFAULT_TRACKB_NEXT_DESIGN,
    )
    parser.add_argument(
        "--trackb-active-route-tile-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-active-route-tile-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-active-route-tile-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ACTIVE_ROUTE_TILE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-expert-cohort-design-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_COHORT_DESIGN,
    )
    parser.add_argument(
        "--trackb-expert-cohort-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_COHORT_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-cohort-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_COHORT_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-cohort-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_COHORT_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-route-batch-segmented-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_DESIGN,
    )
    parser.add_argument(
        "--trackb-route-batch-segmented-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-route-batch-segmented-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-route-batch-segmented-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BATCH_SEGMENTED_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-component-stream-partial-reduction-design-json",
        type=Path,
        default=DEFAULT_TRACKB_COMPONENT_STREAM_PARTIAL_REDUCTION_DESIGN,
    )
    parser.add_argument(
        "--trackb-component-stream-partial-reduction-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_COMPONENT_STREAM_PARTIAL_REDUCTION_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-component-stream-tensorops-rejection-json",
        type=Path,
        default=DEFAULT_TRACKB_COMPONENT_STREAM_TENSOROPS_REJECTION,
    )
    parser.add_argument(
        "--trackb-next-family-gate-json",
        type=Path,
        default=DEFAULT_TRACKB_NEXT_FAMILY_GATE,
    )
    parser.add_argument(
        "--trackb-token-cohort-codeword-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-cohort-codeword-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-cohort-codeword-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-cohort-codeword-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_CODEWORD_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-cohort-mma-codeword-tile-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-cohort-mma-codeword-tile-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-cohort-mma-codeword-tile-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-cohort-mma-codeword-tile-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_COHORT_MMA_CODEWORD_TILE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-output-stationary-codeword-tile-design-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_DESIGN,
    )
    parser.add_argument(
        "--trackb-output-stationary-codeword-tile-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-output-stationary-codeword-tile-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-output-stationary-codeword-tile-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_STATIONARY_CODEWORD_TILE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-input-stationary-codeword-tile-design-json",
        type=Path,
        default=DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_DESIGN,
    )
    parser.add_argument(
        "--trackb-input-stationary-codeword-tile-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-input-stationary-codeword-tile-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-input-stationary-codeword-tile-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_INPUT_STATIONARY_CODEWORD_TILE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-expert-kblock-codeword-factor-reuse-design-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_DESIGN,
    )
    parser.add_argument(
        "--trackb-expert-kblock-codeword-factor-reuse-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-kblock-codeword-factor-reuse-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-kblock-codeword-factor-reuse-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-route-codeword-lut-accumulate-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_DESIGN,
    )
    parser.add_argument(
        "--trackb-route-codeword-lut-accumulate-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-route-codeword-lut-accumulate-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-route-codeword-lut-accumulate-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_CODEWORD_LUT_ACCUMULATE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-rowwise-codeword-tile-accumulate-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_DESIGN,
    )
    parser.add_argument(
        "--trackb-rowwise-codeword-tile-accumulate-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-rowwise-codeword-tile-accumulate-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-rowwise-codeword-tile-accumulate-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROWWISE_CODEWORD_TILE_ACCUMULATE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-output-tile-local-codeword-lut-design-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_DESIGN,
    )
    parser.add_argument(
        "--trackb-output-tile-local-codeword-lut-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-output-tile-local-codeword-lut-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-output-tile-local-codeword-lut-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_TILE_LOCAL_CODEWORD_LUT_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-route-microtile-codeword-block-reduce-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_DESIGN,
    )
    parser.add_argument(
        "--trackb-route-microtile-codeword-block-reduce-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-route-microtile-codeword-block-reduce-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-route-microtile-codeword-block-reduce-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-kblock-wavefront-codeword-scan-design-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_DESIGN,
    )
    parser.add_argument(
        "--trackb-kblock-wavefront-codeword-scan-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-kblock-wavefront-codeword-scan-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-kblock-wavefront-codeword-scan-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_WAVEFRONT_CODEWORD_SCAN_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-route-output-stripe-pipeline-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-route-output-stripe-pipeline-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-route-output-stripe-pipeline-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-route-output-stripe-pipeline-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-expert-kblock-scale-slot-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-expert-kblock-scale-slot-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-kblock-scale-slot-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-expert-kblock-scale-slot-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_EXPERT_KBLOCK_SCALE_SLOT_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-scale-group-route-block-reduce-design-json",
        type=Path,
        default=DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_DESIGN,
    )
    parser.add_argument(
        "--trackb-scale-group-route-block-reduce-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-scale-group-route-block-reduce-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-scale-group-route-block-reduce-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_SCALE_GROUP_ROUTE_BLOCK_REDUCE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-route-block-output-group-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-route-block-output-group-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-route-block-output-group-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-route-block-output-group-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-output-group-pretransposed-codeword-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-output-group-pretransposed-codeword-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-output-group-pretransposed-codeword-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-output-group-pretransposed-codeword-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-kblock-output-group-route-fused-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-kblock-output-group-route-fused-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-kblock-output-group-route-fused-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-kblock-output-group-route-fused-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-route-tile-output-swizzle-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-route-tile-output-swizzle-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-route-tile-output-swizzle-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-route-tile-output-swizzle-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-topk-output-tile-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-topk-output-tile-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-topk-output-tile-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-topk-output-tile-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_TOPK_OUTPUT_TILE_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-block-output-group-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-block-output-group-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-block-output-group-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-block-output-group-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-output-stripe-group-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-output-stripe-group-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-output-stripe-group-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-output-stripe-group-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-expert-output-block-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-expert-output-block-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-expert-output-block-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-expert-output-block-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-kblock-accumulator-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-kblock-accumulator-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-kblock-accumulator-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-kblock-accumulator-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-output-group-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-output-group-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-output-group-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-output-group-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-group-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-group-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-group-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-group-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-codeword-group-pipeline-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-codeword-group-pipeline-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-codeword-group-pipeline-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-codeword-group-pipeline-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-scale-slot-broadcast-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-scale-slot-broadcast-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-scale-slot-broadcast-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-scale-slot-broadcast-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-route-bucket-codeword-reduce-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-route-bucket-codeword-reduce-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-route-bucket-codeword-reduce-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-route-bucket-codeword-reduce-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-kblock-microtile-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-kblock-microtile-stream-native-parity-json",
        type=Path,
        default=(
            DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_NATIVE_PARITY
        ),
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-kblock-microtile-stream-artifact-parity-json",
        type=Path,
        default=(
            DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_ARTIFACT_PARITY
        ),
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-kblock-microtile-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_SPEED_PACKET,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-tile-fused-stream-design-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_DESIGN,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-tile-fused-stream-native-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_NATIVE_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-tile-fused-stream-artifact-parity-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_ARTIFACT_PARITY,
    )
    parser.add_argument(
        "--trackb-token-pair-slot-topk-output-tile-fused-stream-speed-packet-json",
        type=Path,
        default=DEFAULT_TRACKB_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_SPEED_PACKET,
    )
    parser.add_argument("--qwen-gate-json", type=Path, default=DEFAULT_QWEN_GATE)
    parser.add_argument("--glm52-layer3-json", type=Path, default=DEFAULT_GLM52_LAYER3)
    parser.add_argument("--glm52-layer77-json", type=Path, default=DEFAULT_GLM52_LAYER77)
    parser.add_argument(
        "--cache-source-scan-json",
        type=Path,
        default=DEFAULT_CACHE_SOURCE_SCAN,
    )
    parser.add_argument(
        "--single-host-cache-attempt-json",
        type=Path,
        default=DEFAULT_SINGLE_HOST_CACHE_ATTEMPT,
    )
    parser.add_argument(
        "--layer-split-recommendation-json",
        type=Path,
        default=DEFAULT_LAYER_SPLIT_RECOMMENDATION,
    )
    parser.add_argument(
        "--rdma-topology-audit-json",
        type=Path,
        default=DEFAULT_RDMA_TOPOLOGY_AUDIT,
    )
    parser.add_argument("--model-card-path", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = build_status(
        workstream1_path=args.workstream1_json,
        workstream1_fixed_bump_path=args.workstream1_fixed_bump_json,
        trackb_path=args.trackb_json,
        trackb_design_path=args.trackb_design_json,
        trackb_source_guard_path=args.trackb_source_guard_json,
        trackb_native_parity_path=args.trackb_native_parity_json,
        trackb_artifact_parity_path=args.trackb_artifact_parity_json,
        trackb_speed_packet_path=args.trackb_speed_packet_json,
        trackb_successor_design_path=args.trackb_successor_design_json,
        trackb_successor_native_parity_path=args.trackb_successor_native_parity_json,
        trackb_successor_artifact_parity_path=args.trackb_successor_artifact_parity_json,
        trackb_successor_speed_packet_path=args.trackb_successor_speed_packet_json,
        trackb_next_design_path=args.trackb_next_design_json,
        trackb_active_route_tile_native_parity_path=(
            args.trackb_active_route_tile_native_parity_json
        ),
        trackb_active_route_tile_artifact_parity_path=(
            args.trackb_active_route_tile_artifact_parity_json
        ),
        trackb_active_route_tile_speed_packet_path=(
            args.trackb_active_route_tile_speed_packet_json
        ),
        trackb_expert_cohort_design_path=args.trackb_expert_cohort_design_json,
        trackb_expert_cohort_native_parity_path=(
            args.trackb_expert_cohort_native_parity_json
        ),
        trackb_expert_cohort_artifact_parity_path=(
            args.trackb_expert_cohort_artifact_parity_json
        ),
        trackb_expert_cohort_speed_packet_path=(
            args.trackb_expert_cohort_speed_packet_json
        ),
        trackb_route_batch_segmented_design_path=(
            args.trackb_route_batch_segmented_design_json
        ),
        trackb_route_batch_segmented_native_parity_path=(
            args.trackb_route_batch_segmented_native_parity_json
        ),
        trackb_route_batch_segmented_artifact_parity_path=(
            args.trackb_route_batch_segmented_artifact_parity_json
        ),
        trackb_route_batch_segmented_speed_packet_path=(
            args.trackb_route_batch_segmented_speed_packet_json
        ),
        trackb_component_stream_partial_reduction_design_path=(
            args.trackb_component_stream_partial_reduction_design_json
        ),
        trackb_component_stream_partial_reduction_native_parity_path=(
            args.trackb_component_stream_partial_reduction_native_parity_json
        ),
        trackb_component_stream_tensorops_rejection_path=(
            args.trackb_component_stream_tensorops_rejection_json
        ),
        trackb_next_family_gate_path=args.trackb_next_family_gate_json,
        trackb_token_cohort_codeword_stream_design_path=(
            args.trackb_token_cohort_codeword_stream_design_json
        ),
        trackb_token_cohort_codeword_stream_native_parity_path=(
            args.trackb_token_cohort_codeword_stream_native_parity_json
        ),
        trackb_token_cohort_codeword_stream_artifact_parity_path=(
            args.trackb_token_cohort_codeword_stream_artifact_parity_json
        ),
        trackb_token_cohort_codeword_stream_speed_packet_path=(
            args.trackb_token_cohort_codeword_stream_speed_packet_json
        ),
        trackb_token_cohort_mma_codeword_tile_design_path=(
            args.trackb_token_cohort_mma_codeword_tile_design_json
        ),
        trackb_token_cohort_mma_codeword_tile_native_parity_path=(
            args.trackb_token_cohort_mma_codeword_tile_native_parity_json
        ),
        trackb_token_cohort_mma_codeword_tile_artifact_parity_path=(
            args.trackb_token_cohort_mma_codeword_tile_artifact_parity_json
        ),
        trackb_token_cohort_mma_codeword_tile_speed_packet_path=(
            args.trackb_token_cohort_mma_codeword_tile_speed_packet_json
        ),
        trackb_output_stationary_codeword_tile_design_path=(
            args.trackb_output_stationary_codeword_tile_design_json
        ),
        trackb_output_stationary_codeword_tile_native_parity_path=(
            args.trackb_output_stationary_codeword_tile_native_parity_json
        ),
        trackb_output_stationary_codeword_tile_artifact_parity_path=(
            args.trackb_output_stationary_codeword_tile_artifact_parity_json
        ),
        trackb_output_stationary_codeword_tile_speed_packet_path=(
            args.trackb_output_stationary_codeword_tile_speed_packet_json
        ),
        trackb_input_stationary_codeword_tile_design_path=(
            args.trackb_input_stationary_codeword_tile_design_json
        ),
        trackb_input_stationary_codeword_tile_native_parity_path=(
            args.trackb_input_stationary_codeword_tile_native_parity_json
        ),
        trackb_input_stationary_codeword_tile_artifact_parity_path=(
            args.trackb_input_stationary_codeword_tile_artifact_parity_json
        ),
        trackb_input_stationary_codeword_tile_speed_packet_path=(
            args.trackb_input_stationary_codeword_tile_speed_packet_json
        ),
        trackb_expert_kblock_codeword_factor_reuse_design_path=(
            args.trackb_expert_kblock_codeword_factor_reuse_design_json
        ),
        trackb_expert_kblock_codeword_factor_reuse_native_parity_path=(
            args.trackb_expert_kblock_codeword_factor_reuse_native_parity_json
        ),
        trackb_expert_kblock_codeword_factor_reuse_artifact_parity_path=(
            args.trackb_expert_kblock_codeword_factor_reuse_artifact_parity_json
        ),
        trackb_expert_kblock_codeword_factor_reuse_speed_packet_path=(
            args.trackb_expert_kblock_codeword_factor_reuse_speed_packet_json
        ),
        trackb_route_codeword_lut_accumulate_design_path=(
            args.trackb_route_codeword_lut_accumulate_design_json
        ),
        trackb_route_codeword_lut_accumulate_native_parity_path=(
            args.trackb_route_codeword_lut_accumulate_native_parity_json
        ),
        trackb_route_codeword_lut_accumulate_artifact_parity_path=(
            args.trackb_route_codeword_lut_accumulate_artifact_parity_json
        ),
        trackb_route_codeword_lut_accumulate_speed_packet_path=(
            args.trackb_route_codeword_lut_accumulate_speed_packet_json
        ),
        trackb_rowwise_codeword_tile_accumulate_design_path=(
            args.trackb_rowwise_codeword_tile_accumulate_design_json
        ),
        trackb_rowwise_codeword_tile_accumulate_native_parity_path=(
            args.trackb_rowwise_codeword_tile_accumulate_native_parity_json
        ),
        trackb_rowwise_codeword_tile_accumulate_artifact_parity_path=(
            args.trackb_rowwise_codeword_tile_accumulate_artifact_parity_json
        ),
        trackb_rowwise_codeword_tile_accumulate_speed_packet_path=(
            args.trackb_rowwise_codeword_tile_accumulate_speed_packet_json
        ),
        trackb_output_tile_local_codeword_lut_design_path=(
            args.trackb_output_tile_local_codeword_lut_design_json
        ),
        trackb_output_tile_local_codeword_lut_native_parity_path=(
            args.trackb_output_tile_local_codeword_lut_native_parity_json
        ),
        trackb_output_tile_local_codeword_lut_artifact_parity_path=(
            args.trackb_output_tile_local_codeword_lut_artifact_parity_json
        ),
        trackb_output_tile_local_codeword_lut_speed_packet_path=(
            args.trackb_output_tile_local_codeword_lut_speed_packet_json
        ),
        trackb_route_microtile_codeword_block_reduce_design_path=(
            args.trackb_route_microtile_codeword_block_reduce_design_json
        ),
        trackb_route_microtile_codeword_block_reduce_native_parity_path=(
            args.trackb_route_microtile_codeword_block_reduce_native_parity_json
        ),
        trackb_route_microtile_codeword_block_reduce_artifact_parity_path=(
            args.trackb_route_microtile_codeword_block_reduce_artifact_parity_json
        ),
        trackb_route_microtile_codeword_block_reduce_speed_packet_path=(
            args.trackb_route_microtile_codeword_block_reduce_speed_packet_json
        ),
        trackb_kblock_wavefront_codeword_scan_design_path=(
            args.trackb_kblock_wavefront_codeword_scan_design_json
        ),
        trackb_kblock_wavefront_codeword_scan_native_parity_path=(
            args.trackb_kblock_wavefront_codeword_scan_native_parity_json
        ),
        trackb_kblock_wavefront_codeword_scan_artifact_parity_path=(
            args.trackb_kblock_wavefront_codeword_scan_artifact_parity_json
        ),
        trackb_kblock_wavefront_codeword_scan_speed_packet_path=(
            args.trackb_kblock_wavefront_codeword_scan_speed_packet_json
        ),
        trackb_token_route_output_stripe_pipeline_design_path=(
            args.trackb_token_route_output_stripe_pipeline_design_json
        ),
        trackb_token_route_output_stripe_pipeline_native_parity_path=(
            args.trackb_token_route_output_stripe_pipeline_native_parity_json
        ),
        trackb_token_route_output_stripe_pipeline_artifact_parity_path=(
            args.trackb_token_route_output_stripe_pipeline_artifact_parity_json
        ),
        trackb_token_route_output_stripe_pipeline_speed_packet_path=(
            args.trackb_token_route_output_stripe_pipeline_speed_packet_json
        ),
        trackb_expert_kblock_scale_slot_stream_design_path=(
            args.trackb_expert_kblock_scale_slot_stream_design_json
        ),
        trackb_expert_kblock_scale_slot_stream_native_parity_path=(
            args.trackb_expert_kblock_scale_slot_stream_native_parity_json
        ),
        trackb_expert_kblock_scale_slot_stream_artifact_parity_path=(
            args.trackb_expert_kblock_scale_slot_stream_artifact_parity_json
        ),
        trackb_expert_kblock_scale_slot_stream_speed_packet_path=(
            args.trackb_expert_kblock_scale_slot_stream_speed_packet_json
        ),
        trackb_scale_group_route_block_reduce_design_path=(
            args.trackb_scale_group_route_block_reduce_design_json
        ),
        trackb_scale_group_route_block_reduce_native_parity_path=(
            args.trackb_scale_group_route_block_reduce_native_parity_json
        ),
        trackb_scale_group_route_block_reduce_artifact_parity_path=(
            args.trackb_scale_group_route_block_reduce_artifact_parity_json
        ),
        trackb_scale_group_route_block_reduce_speed_packet_path=(
            args.trackb_scale_group_route_block_reduce_speed_packet_json
        ),
        trackb_route_block_output_group_stream_design_path=(
            args.trackb_route_block_output_group_stream_design_json
        ),
        trackb_route_block_output_group_stream_native_parity_path=(
            args.trackb_route_block_output_group_stream_native_parity_json
        ),
        trackb_route_block_output_group_stream_artifact_parity_path=(
            args.trackb_route_block_output_group_stream_artifact_parity_json
        ),
        trackb_route_block_output_group_stream_speed_packet_path=(
            args.trackb_route_block_output_group_stream_speed_packet_json
        ),
        trackb_output_group_pretransposed_codeword_stream_design_path=(
            args.trackb_output_group_pretransposed_codeword_stream_design_json
        ),
        trackb_output_group_pretransposed_codeword_stream_native_parity_path=(
            args.trackb_output_group_pretransposed_codeword_stream_native_parity_json
        ),
        trackb_output_group_pretransposed_codeword_stream_artifact_parity_path=(
            args.trackb_output_group_pretransposed_codeword_stream_artifact_parity_json
        ),
        trackb_output_group_pretransposed_codeword_stream_speed_packet_path=(
            args.trackb_output_group_pretransposed_codeword_stream_speed_packet_json
        ),
        trackb_kblock_output_group_route_fused_stream_design_path=(
            args.trackb_kblock_output_group_route_fused_stream_design_json
        ),
        trackb_kblock_output_group_route_fused_stream_native_parity_path=(
            args.trackb_kblock_output_group_route_fused_stream_native_parity_json
        ),
        trackb_kblock_output_group_route_fused_stream_artifact_parity_path=(
            args.trackb_kblock_output_group_route_fused_stream_artifact_parity_json
        ),
        trackb_kblock_output_group_route_fused_stream_speed_packet_path=(
            args.trackb_kblock_output_group_route_fused_stream_speed_packet_json
        ),
        trackb_route_tile_output_swizzle_stream_design_path=(
            args.trackb_route_tile_output_swizzle_stream_design_json
        ),
        trackb_route_tile_output_swizzle_stream_native_parity_path=(
            args.trackb_route_tile_output_swizzle_stream_native_parity_json
        ),
        trackb_route_tile_output_swizzle_stream_artifact_parity_path=(
            args.trackb_route_tile_output_swizzle_stream_artifact_parity_json
        ),
        trackb_route_tile_output_swizzle_stream_speed_packet_path=(
            args.trackb_route_tile_output_swizzle_stream_speed_packet_json
        ),
        trackb_token_topk_output_tile_stream_design_path=(
            args.trackb_token_topk_output_tile_stream_design_json
        ),
        trackb_token_topk_output_tile_stream_native_parity_path=(
            args.trackb_token_topk_output_tile_stream_native_parity_json
        ),
        trackb_token_topk_output_tile_stream_artifact_parity_path=(
            args.trackb_token_topk_output_tile_stream_artifact_parity_json
        ),
        trackb_token_topk_output_tile_stream_speed_packet_path=(
            args.trackb_token_topk_output_tile_stream_speed_packet_json
        ),
        trackb_token_block_output_group_stream_design_path=(
            args.trackb_token_block_output_group_stream_design_json
        ),
        trackb_token_block_output_group_stream_native_parity_path=(
            args.trackb_token_block_output_group_stream_native_parity_json
        ),
        trackb_token_block_output_group_stream_artifact_parity_path=(
            args.trackb_token_block_output_group_stream_artifact_parity_json
        ),
        trackb_token_block_output_group_stream_speed_packet_path=(
            args.trackb_token_block_output_group_stream_speed_packet_json
        ),
        trackb_token_output_stripe_group_stream_design_path=(
            args.trackb_token_output_stripe_group_stream_design_json
        ),
        trackb_token_output_stripe_group_stream_native_parity_path=(
            args.trackb_token_output_stripe_group_stream_native_parity_json
        ),
        trackb_token_output_stripe_group_stream_artifact_parity_path=(
            args.trackb_token_output_stripe_group_stream_artifact_parity_json
        ),
        trackb_token_output_stripe_group_stream_speed_packet_path=(
            args.trackb_token_output_stripe_group_stream_speed_packet_json
        ),
        trackb_token_expert_output_block_stream_design_path=(
            args.trackb_token_expert_output_block_stream_design_json
        ),
        trackb_token_expert_output_block_stream_native_parity_path=(
            args.trackb_token_expert_output_block_stream_native_parity_json
        ),
        trackb_token_expert_output_block_stream_artifact_parity_path=(
            args.trackb_token_expert_output_block_stream_artifact_parity_json
        ),
        trackb_token_expert_output_block_stream_speed_packet_path=(
            args.trackb_token_expert_output_block_stream_speed_packet_json
        ),
        trackb_token_pair_kblock_accumulator_stream_design_path=(
            args.trackb_token_pair_kblock_accumulator_stream_design_json
        ),
        trackb_token_pair_kblock_accumulator_stream_native_parity_path=(
            args.trackb_token_pair_kblock_accumulator_stream_native_parity_json
        ),
        trackb_token_pair_kblock_accumulator_stream_artifact_parity_path=(
            args.trackb_token_pair_kblock_accumulator_stream_artifact_parity_json
        ),
        trackb_token_pair_kblock_accumulator_stream_speed_packet_path=(
            args.trackb_token_pair_kblock_accumulator_stream_speed_packet_json
        ),
        trackb_token_pair_output_group_stream_design_path=(
            args.trackb_token_pair_output_group_stream_design_json
        ),
        trackb_token_pair_output_group_stream_native_parity_path=(
            args.trackb_token_pair_output_group_stream_native_parity_json
        ),
        trackb_token_pair_output_group_stream_artifact_parity_path=(
            args.trackb_token_pair_output_group_stream_artifact_parity_json
        ),
        trackb_token_pair_output_group_stream_speed_packet_path=(
            args.trackb_token_pair_output_group_stream_speed_packet_json
        ),
        trackb_token_pair_slot_topk_output_group_stream_design_path=(
            args.trackb_token_pair_slot_topk_output_group_stream_design_json
        ),
        trackb_token_pair_slot_topk_output_group_stream_native_parity_path=(
            args.trackb_token_pair_slot_topk_output_group_stream_native_parity_json
        ),
        trackb_token_pair_slot_topk_output_group_stream_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_output_group_stream_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_output_group_stream_speed_packet_path=(
            args.trackb_token_pair_slot_topk_output_group_stream_speed_packet_json
        ),
        trackb_token_pair_slot_topk_codeword_group_pipeline_design_path=(
            args.trackb_token_pair_slot_topk_codeword_group_pipeline_design_json
        ),
        trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_path=(
            args.trackb_token_pair_slot_topk_codeword_group_pipeline_native_parity_json
        ),
        trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_codeword_group_pipeline_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_path=(
            args.trackb_token_pair_slot_topk_codeword_group_pipeline_speed_packet_json
        ),
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_path=(
            args.trackb_token_pair_slot_topk_scale_slot_broadcast_stream_design_json
        ),
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_path=(
            args.trackb_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity_json
        ),
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_scale_slot_broadcast_stream_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_path=(
            args.trackb_token_pair_slot_topk_scale_slot_broadcast_stream_speed_packet_json
        ),
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_path=(
            args.trackb_token_pair_slot_topk_route_bucket_codeword_reduce_design_json
        ),
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_path=(
            args.trackb_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity_json
        ),
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_route_bucket_codeword_reduce_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_path=(
            args.trackb_token_pair_slot_topk_route_bucket_codeword_reduce_speed_packet_json
        ),
        trackb_token_pair_slot_topk_kblock_microtile_stream_design_path=(
            args.trackb_token_pair_slot_topk_kblock_microtile_stream_design_json
        ),
        trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_path=(
            args.trackb_token_pair_slot_topk_kblock_microtile_stream_native_parity_json
        ),
        trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_kblock_microtile_stream_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_path=(
            args.trackb_token_pair_slot_topk_kblock_microtile_stream_speed_packet_json
        ),
        trackb_token_pair_slot_topk_output_tile_fused_stream_design_path=(
            args.trackb_token_pair_slot_topk_output_tile_fused_stream_design_json
        ),
        trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_path=(
            args.trackb_token_pair_slot_topk_output_tile_fused_stream_native_parity_json
        ),
        trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_path=(
            args.trackb_token_pair_slot_topk_output_tile_fused_stream_artifact_parity_json
        ),
        trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_path=(
            args.trackb_token_pair_slot_topk_output_tile_fused_stream_speed_packet_json
        ),
        qwen_gate_path=args.qwen_gate_json,
        glm52_layer3_path=args.glm52_layer3_json,
        glm52_layer77_path=args.glm52_layer77_json,
        cache_source_scan_path=args.cache_source_scan_json,
        single_host_cache_attempt_path=args.single_host_cache_attempt_json,
        layer_split_recommendation_path=args.layer_split_recommendation_json,
        rdma_topology_audit_path=args.rdma_topology_audit_json,
        model_card_path=args.model_card_path,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
