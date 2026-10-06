#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mlx::steel;

constant constexpr short mlx_vq_nax_elems_per_frag = 8;
constant constexpr short mlx_vq_nax_elem_cols = 4;
constant constexpr short mlx_vq_nax_elem_rows_jump = 8;
constant constexpr uint mlx_vq_nax_bn_tile = 64;
constant constexpr uint mlx_vq_nax_bk_tile = 64;
constant constexpr uint mlx_vq_nax_bk_padded = 72;
constant constexpr uint mlx_vq_nax_bk128_tile = 128;
constant constexpr uint mlx_vq_nax_bk128_padded = 136;
constant constexpr uint mlx_vq_nax_threads_per_tg = 128;

inline short2 mlx_vq_nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

[[kernel]] void nax_fp16_matmul_tile(
    const device half* x [[buffer(0)]],
    const device half* weight_t [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint lane [[thread_position_in_threadgroup]]) {
  (void)lane;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      32,
      16,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  for (uint16_t i = 0; i < a_t.get_capacity(); ++i) {
    if (a_t.is_valid_element(i)) {
      auto ids = a_t.get_multidimensional_index(i);
      uint k = uint(ids[0]);
      uint m = uint(ids[1]);
      a_t[i] = x[m * 16u + k];
    }
  }
  for (uint16_t i = 0; i < b_t.get_capacity(); ++i) {
    if (b_t.is_valid_element(i)) {
      auto ids = b_t.get_multidimensional_index(i);
      uint n = uint(ids[0]);
      uint k = uint(ids[1]);
      b_t[i] = weight_t[k * 16u + n];
    }
  }
  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    if (c_t.is_valid_element(i)) {
      c_t[i] = 0.0f;
    }
  }

  matmul_op.run(a_t, b_t, c_t);

  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    if (c_t.is_valid_element(i)) {
      auto ids = c_t.get_multidimensional_index(i);
      uint n = uint(ids[0]);
      uint m = uint(ids[1]);
      out[m * 16u + n] = half(c_t[i]);
    }
  }
}

inline float mlx_vq_decode_packed_int2_value(uint packed, uint dim) {
  uint nibble = (packed >> (4u * dim)) & 0xFu;
  return (float(nibble) - 8.0f) * 0.5f;
}

inline half mlx_vq_decode_packed_int2_half_value(uint packed, uint dim) {
  uint nibble = (packed >> (4u * dim)) & 0xFu;
  return (half(nibble) - half(8.0f)) * half(0.5f);
}

inline int8_t mlx_vq_decode_packed_int2_i8(uint packed, uint dim) {
  int nibble = int((packed >> (4u * dim)) & 0xFu);
  return int8_t(nibble - 8);
}

inline uint mlx_vq_e8p_shuffle_dim(uint dim) {
  switch (dim) {
    case 0:
      return 0;
    case 1:
      return 4;
    case 2:
      return 1;
    case 3:
      return 5;
    case 4:
      return 2;
    case 5:
      return 6;
    case 6:
      return 3;
    default:
      return 7;
  }
}

inline uint mlx_vq_sign_parity8(uint signs) {
  signs ^= signs >> 4;
  signs ^= signs >> 2;
  signs ^= signs >> 1;
  return signs & 1u;
}

inline float mlx_vq_decode_e8p_value(
    uint codeword,
    const device uint* packed_abs_grid,
    uint dim) {
  uint signs = codeword & 0xFFu;
  uint abs_idx = codeword >> 8;
  uint parity = mlx_vq_sign_parity8(signs);
  uint effective_signs = signs ^ parity;
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  uint abs_code = packed_abs_grid[abs_idx];
  float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
  if (((effective_signs >> packed_dim) & 1u) != 0) {
    value = -value;
  }
  return value + (parity != 0 ? -0.25f : 0.25f);
}

inline float mlx_vq_decode_e8p_split_value(
    uint signs,
    uint abs_idx,
    uint parity,
    const device uint* packed_abs_grid,
    uint dim) {
  uint effective_signs = signs ^ parity;
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  uint abs_code = packed_abs_grid[abs_idx];
  float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
  if (((effective_signs >> packed_dim) & 1u) != 0) {
    value = -value;
  }
  return value + (parity != 0 ? -0.25f : 0.25f);
}

inline float mlx_vq_decode_e8p_nibble_value(
    uint sign_low_nibble,
    uint sign_high_nibble,
    uint abs_idx,
    uint parity,
    const device uint* packed_abs_grid,
    uint dim) {
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  uint abs_code = packed_abs_grid[abs_idx];
  float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
  uint sign_bit = packed_dim < 4u
      ? ((sign_low_nibble >> packed_dim) & 1u)
      : ((sign_high_nibble >> (packed_dim - 4u)) & 1u);
  if (packed_dim == 0u) {
    sign_bit ^= parity;
  }
  if (sign_bit != 0) {
    value = -value;
  }
  return value + (parity != 0 ? -0.25f : 0.25f);
}

inline float mlx_vq_decode_e8p_value_tg(
    uint codeword,
    const threadgroup uint* packed_abs_grid,
    uint dim) {
  uint signs = codeword & 0xFFu;
  uint abs_idx = codeword >> 8;
  uint parity = mlx_vq_sign_parity8(signs);
  uint effective_signs = signs ^ parity;
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  uint abs_code = packed_abs_grid[abs_idx];
  float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
  if (((effective_signs >> packed_dim) & 1u) != 0) {
    value = -value;
  }
  return value + (parity != 0 ? -0.25f : 0.25f);
}

inline uint mlx_vq_group_index_gs352(uint k_code) {
  return k_code >= 1056u ? 3u : (k_code >= 704u ? 2u : (k_code >= 352u ? 1u : 0u));
}

inline half mlx_vq_decode_e8p_hoisted_half(
    uint abs_code,
    uint effective_signs,
    uint parity,
    uint dim,
    float scale_f) {
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  float value = mlx_vq_decode_packed_int2_value(abs_code, packed_dim);
  if (((effective_signs >> packed_dim) & 1u) != 0) {
    value = -value;
  }
  value += parity != 0 ? -0.25f : 0.25f;
  return half(value * scale_f);
}

inline half mlx_vq_decode_e8p_scaled_half(
    uint codeword,
    const device uint* packed_abs_grid,
    uint dim,
    half scale) {
  uint signs = codeword & 0xFFu;
  uint abs_idx = codeword >> 8;
  uint parity = mlx_vq_sign_parity8(signs);
  uint effective_signs = signs ^ parity;
  uint packed_dim = mlx_vq_e8p_shuffle_dim(dim);
  uint abs_code = packed_abs_grid[abs_idx];
  half value = mlx_vq_decode_packed_int2_half_value(abs_code, packed_dim);
  if (((effective_signs >> packed_dim) & 1u) != 0) {
    value = -value;
  }
  value += half(parity != 0 ? -0.25f : 0.25f);
  return value * scale;
}

template <typename Tile>
METAL_FUNC void mlx_vq_load_routed_a_tile(
    thread Tile& tile,
    const device half* x,
    const device int* lhs_indices,
    uint route_base,
    int valid_routes,
    uint K,
    uint k_base,
    uint tm) {
  const short2 sc = Tile::NAXFrag_t::get_coord();

  STEEL_PRAGMA_UNROLL
  for (short frag_m = 0; frag_m < Tile::kTileRows; ++frag_m) {
    STEEL_PRAGMA_UNROLL
    for (short frag_k = 0; frag_k < Tile::kTileCols; ++frag_k) {
      thread typename Tile::frag_type& frag = tile.frag_at(frag_m, frag_k);

      STEEL_PRAGMA_UNROLL
      for (short row = 0; row < Tile::kFragThrRows; ++row) {
        uint route_slot =
            tm + uint(frag_m) * uint(Tile::kFragRows) + uint(sc.y) +
            uint(row) * uint(Tile::kFragRowsJump);

        STEEL_PRAGMA_UNROLL
        for (short col = 0; col < Tile::kFragThrCols; ++col) {
          uint k =
              k_base + uint(frag_k) * uint(Tile::kFragCols) + uint(sc.x) +
              uint(col);
          half value = half(0.0h);
          if (int(route_slot) < valid_routes && k < K) {
            uint token = uint(lhs_indices[route_base + route_slot]);
            value = x[token * K + k];
          }
          frag[row * Tile::kFragThrCols + col] = value;
        }
      }
    }
  }
}

[[kernel]] void nax_e8_fp16_matmul_tile(
    const device half* x [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant const uint& group_size [[buffer(5)]],
    uint lane [[thread_position_in_threadgroup]]) {
  (void)lane;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      32,
      16,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  for (uint16_t i = 0; i < a_t.get_capacity(); ++i) {
    if (a_t.is_valid_element(i)) {
      auto ids = a_t.get_multidimensional_index(i);
      uint k = uint(ids[0]);
      uint m = uint(ids[1]);
      a_t[i] = x[m * 16u + k];
    }
  }
  for (uint16_t i = 0; i < b_t.get_capacity(); ++i) {
    if (b_t.is_valid_element(i)) {
      auto ids = b_t.get_multidimensional_index(i);
      uint n = uint(ids[0]);
      uint k = uint(ids[1]);
      uint code = uint(codes[n * 2u + (k >> 3)]);
      float decoded = mlx_vq_decode_packed_int2_value(codebook[code], k & 7u);
      float scale = float(scales[n * (16u / group_size) + (k / group_size)]);
      b_t[i] = half(decoded * scale);
    }
  }
  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    if (c_t.is_valid_element(i)) {
      c_t[i] = 0.0f;
    }
  }

  matmul_op.run(a_t, b_t, c_t);

  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    if (c_t.is_valid_element(i)) {
      auto ids = c_t.get_multidimensional_index(i);
      uint n = uint(ids[0]);
      uint m = uint(ids[1]);
      out[m * 16u + n] = half(c_t[i]);
    }
  }
}

[[kernel]] void nax_e8_fp16_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant const uint& M [[buffer(5)]],
    constant const uint& N [[buffer(6)]],
    constant const uint& K [[buffer(7)]],
    constant const uint& group_size [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  (void)tid;

  uint n_base = tgid.x * 16u;
  uint m_base = tgid.y * 32u;
  uint codewords = K >> 3;
  uint groups = K / group_size;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      32,
      16,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  float c_acc[32];
  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    c_acc[i] = 0.0f;
  }

  for (uint k_base = 0; k_base < K; k_base += 16u) {
    for (uint16_t i = 0; i < a_t.get_capacity(); ++i) {
      if (a_t.is_valid_element(i)) {
        auto ids = a_t.get_multidimensional_index(i);
        uint k = k_base + uint(ids[0]);
        uint m = m_base + uint(ids[1]);
        a_t[i] = (m < M && k < K) ? x[m * K + k] : half(0.0h);
      }
    }
    for (uint16_t i = 0; i < b_t.get_capacity(); ++i) {
      if (b_t.is_valid_element(i)) {
        auto ids = b_t.get_multidimensional_index(i);
        uint n = n_base + uint(ids[0]);
        uint k = k_base + uint(ids[1]);
        half value = half(0.0h);
        if (n < N && k < K) {
          uint code = uint(codes[n * codewords + (k >> 3)]);
          float decoded = mlx_vq_decode_packed_int2_value(codebook[code], k & 7u);
          float scale = float(scales[n * groups + (k / group_size)]);
          value = half(decoded * scale);
        }
        b_t[i] = value;
      }
    }
    for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
      if (c_t.is_valid_element(i)) {
        c_t[i] = c_acc[i];
      }
    }

    matmul_op.run(a_t, b_t, c_t);

    for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
      if (c_t.is_valid_element(i)) {
        c_acc[i] = c_t[i];
      }
    }
  }

  for (uint16_t i = 0; i < c_t.get_capacity(); ++i) {
    if (c_t.is_valid_element(i)) {
      auto ids = c_t.get_multidimensional_index(i);
      uint n = n_base + uint(ids[0]);
      uint m = m_base + uint(ids[1]);
      if (m < M && n < N) {
        out[m * N + n] = half(c_acc[i]);
      }
    }
  }
}

[[kernel]] void nax_e8_fp16_routed_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* lhs_indices [[buffer(4)]],
    const device int* tile_experts [[buffer(5)]],
    const device int* tile_offsets [[buffer(6)]],
    const device int* tile_counts [[buffer(7)]],
    device half* out [[buffer(8)]],
    constant const uint& route_count [[buffer(9)]],
    constant const uint& N [[buffer(10)]],
    constant const uint& K [[buffer(11)]],
    constant const uint& group_size [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  bool use_decoded_codebook = K >= 4096u;
  threadgroup half decoded_codebook[256 * 8];
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  if (use_decoded_codebook) {
    for (uint idx = thread_idx; idx < 256u * 8u; idx += mlx_vq_nax_threads_per_tg) {
      uint code = idx >> 3;
      uint dim = idx & 7u;
      decoded_codebook[idx] = half(mlx_vq_decode_packed_int2_value(codebook[code], dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      if (use_decoded_codebook) {
        uint code_base = code << 3;
        staged_b[base] = decoded_codebook[code_base] * scale;
        staged_b[base + mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 1u] * scale;
        staged_b[base + 2u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 2u] * scale;
        staged_b[base + 3u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 3u] * scale;
        staged_b[base + 4u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 4u] * scale;
        staged_b[base + 5u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 5u] * scale;
        staged_b[base + 6u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 6u] * scale;
        staged_b[base + 7u * mlx_vq_nax_bn_tile] =
            decoded_codebook[code_base + 7u] * scale;
      } else {
        uint packed = codebook[code];
        float scale_f = float(scale);
        staged_b[base] = half(mlx_vq_decode_packed_int2_value(packed, 0u) * scale_f);
        staged_b[base + mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 1u) * scale_f);
        staged_b[base + 2u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 2u) * scale_f);
        staged_b[base + 3u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 3u) * scale_f);
        staged_b[base + 4u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 4u) * scale_f);
        staged_b[base + 5u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 5u) * scale_f);
        staged_b[base + 6u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 6u) * scale_f);
        staged_b[base + 7u * mlx_vq_nax_bn_tile] =
            half(mlx_vq_decode_packed_int2_value(packed, 7u) * scale_f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            b_t[n_frag * mlx_vq_nax_elems_per_frag + row * mlx_vq_nax_elem_cols + col] =
                staged_b[k_local * mlx_vq_nax_bn_tile + n_local];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot = route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile && (route_base + route_slot) < route_count && k < K) {
              uint token = uint(lhs_indices[route_base + route_slot]);
              value = x[token * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot = route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count && n < N) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag + row * mlx_vq_nax_elem_cols + col;
            out[route * N + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8_fp16_routed_matmul_steel(
    const device half* x [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* lhs_indices [[buffer(4)]],
    const device int* tile_experts [[buffer(5)]],
    const device int* tile_offsets [[buffer(6)]],
    const device int* tile_counts [[buffer(7)]],
    device half* out [[buffer(8)]],
    constant const uint& route_count [[buffer(9)]],
    constant const uint& N [[buffer(10)]],
    constant const uint& K [[buffer(11)]],
    constant const uint& group_size [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  bool use_decoded_codebook = K >= 4096u;
  threadgroup half decoded_codebook[256 * 8];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  if (use_decoded_codebook) {
    for (uint idx = thread_idx; idx < 256u * 8u; idx += mlx_vq_nax_threads_per_tg) {
      uint code = idx >> 3;
      uint dim = idx & 7u;
      decoded_codebook[idx] = half(mlx_vq_decode_packed_int2_value(codebook[code], dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      if (use_decoded_codebook) {
        uint code_base = code << 3;
        Ws[base] = decoded_codebook[code_base] * scale;
        Ws[base + 1u] = decoded_codebook[code_base + 1u] * scale;
        Ws[base + 2u] = decoded_codebook[code_base + 2u] * scale;
        Ws[base + 3u] = decoded_codebook[code_base + 3u] * scale;
        Ws[base + 4u] = decoded_codebook[code_base + 4u] * scale;
        Ws[base + 5u] = decoded_codebook[code_base + 5u] * scale;
        Ws[base + 6u] = decoded_codebook[code_base + 6u] * scale;
        Ws[base + 7u] = decoded_codebook[code_base + 7u] * scale;
      } else {
        uint packed = codebook[code];
        float scale_f = float(scale);
        Ws[base] = half(mlx_vq_decode_packed_int2_value(packed, 0u) * scale_f);
        Ws[base + 1u] = half(mlx_vq_decode_packed_int2_value(packed, 1u) * scale_f);
        Ws[base + 2u] = half(mlx_vq_decode_packed_int2_value(packed, 2u) * scale_f);
        Ws[base + 3u] = half(mlx_vq_decode_packed_int2_value(packed, 3u) * scale_f);
        Ws[base + 4u] = half(mlx_vq_decode_packed_int2_value(packed, 4u) * scale_f);
        Ws[base + 5u] = half(mlx_vq_decode_packed_int2_value(packed, 5u) * scale_f);
        Ws[base + 6u] = half(mlx_vq_decode_packed_int2_value(packed, 6u) * scale_f);
        Ws[base + 7u] = half(mlx_vq_decode_packed_int2_value(packed, 7u) * scale_f);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      mlx_vq_load_routed_a_tile(
          Atile,
          x,
          lhs_indices,
          route_base,
          valid_routes,
          K,
          k_block + kk,
          uint(tm));
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8_fp16_sorted_matmul_steel(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  bool use_decoded_codebook = K >= 4096u;
  threadgroup half decoded_codebook[256 * 8];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  if (use_decoded_codebook) {
    for (uint idx = thread_idx; idx < 256u * 8u; idx += mlx_vq_nax_threads_per_tg) {
      uint code = idx >> 3;
      uint dim = idx & 7u;
      decoded_codebook[idx] = half(mlx_vq_decode_packed_int2_value(codebook[code], dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      if (use_decoded_codebook) {
        uint code_base = code << 3;
        Ws[base] = decoded_codebook[code_base] * scale;
        Ws[base + 1u] = decoded_codebook[code_base + 1u] * scale;
        Ws[base + 2u] = decoded_codebook[code_base + 2u] * scale;
        Ws[base + 3u] = decoded_codebook[code_base + 3u] * scale;
        Ws[base + 4u] = decoded_codebook[code_base + 4u] * scale;
        Ws[base + 5u] = decoded_codebook[code_base + 5u] * scale;
        Ws[base + 6u] = decoded_codebook[code_base + 6u] * scale;
        Ws[base + 7u] = decoded_codebook[code_base + 7u] * scale;
      } else {
        uint packed = codebook[code];
        float scale_f = float(scale);
        Ws[base] = half(mlx_vq_decode_packed_int2_value(packed, 0u) * scale_f);
        Ws[base + 1u] = half(mlx_vq_decode_packed_int2_value(packed, 1u) * scale_f);
        Ws[base + 2u] = half(mlx_vq_decode_packed_int2_value(packed, 2u) * scale_f);
        Ws[base + 3u] = half(mlx_vq_decode_packed_int2_value(packed, 3u) * scale_f);
        Ws[base + 4u] = half(mlx_vq_decode_packed_int2_value(packed, 4u) * scale_f);
        Ws[base + 5u] = half(mlx_vq_decode_packed_int2_value(packed, 5u) * scale_f);
        Ws[base + 6u] = half(mlx_vq_decode_packed_int2_value(packed, 6u) * scale_f);
        Ws[base + 7u] = half(mlx_vq_decode_packed_int2_value(packed, 7u) * scale_f);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint WS_STRIDE = mlx_vq_nax_bk_tile;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half decoded_abs_grid[256 * 8];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_tile];

  for (uint idx = thread_idx; idx < 256u; idx += mlx_vq_nax_threads_per_tg) {
    uint abs_code = codebook[idx];
    uint base = idx << 3;
    decoded_abs_grid[base] = mlx_vq_decode_packed_int2_half_value(abs_code, 0u);
    decoded_abs_grid[base + 1u] = mlx_vq_decode_packed_int2_half_value(abs_code, 4u);
    decoded_abs_grid[base + 2u] = mlx_vq_decode_packed_int2_half_value(abs_code, 1u);
    decoded_abs_grid[base + 3u] = mlx_vq_decode_packed_int2_half_value(abs_code, 5u);
    decoded_abs_grid[base + 4u] = mlx_vq_decode_packed_int2_half_value(abs_code, 2u);
    decoded_abs_grid[base + 5u] = mlx_vq_decode_packed_int2_half_value(abs_code, 6u);
    decoded_abs_grid[base + 6u] = mlx_vq_decode_packed_int2_half_value(abs_code, 3u);
    decoded_abs_grid[base + 7u] = mlx_vq_decode_packed_int2_half_value(abs_code, 7u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint signs = code & 0xFFu;
      uint parity = mlx_vq_sign_parity8(signs);
      uint effective_signs = signs ^ parity;
      uint abs_base = (code >> 8) << 3;
      half offset = half(parity != 0 ? -0.25f : 0.25f);

      half4 values_lo = half4(
          decoded_abs_grid[abs_base],
          decoded_abs_grid[abs_base + 1u],
          decoded_abs_grid[abs_base + 2u],
          decoded_abs_grid[abs_base + 3u]);
      half4 values_hi = half4(
          decoded_abs_grid[abs_base + 4u],
          decoded_abs_grid[abs_base + 5u],
          decoded_abs_grid[abs_base + 6u],
          decoded_abs_grid[abs_base + 7u]);
      bool4 signs_lo = bool4(
          (effective_signs & 0x01u) != 0,
          (effective_signs & 0x10u) != 0,
          (effective_signs & 0x02u) != 0,
          (effective_signs & 0x20u) != 0);
      bool4 signs_hi = bool4(
          (effective_signs & 0x04u) != 0,
          (effective_signs & 0x40u) != 0,
          (effective_signs & 0x08u) != 0,
          (effective_signs & 0x80u) != 0);
      values_lo = select(values_lo, -values_lo, signs_lo);
      values_hi = select(values_hi, -values_hi, signs_hi);
      values_lo += half4(offset);
      values_hi += half4(offset);
      values_lo *= half4(scale);
      values_hi *= half4(scale);

      uint base = n_local * WS_STRIDE + codeword_slot * 8u;
      Ws[base] = values_lo.x;
      Ws[base + 1u] = values_lo.y;
      Ws[base + 2u] = values_lo.z;
      Ws[base + 3u] = values_lo.w;
      Ws[base + 4u] = values_hi.x;
      Ws[base + 5u] = values_hi.y;
      Ws[base + 6u] = values_hi.z;
      Ws[base + 7u] = values_hi.w;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(WS_STRIDE), 1>(
          Ws + uint(tn) * WS_STRIDE + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_direct_reduce(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    constant const uint& num_tiles [[buffer(12)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= N || route >= route_count) {
    return;
  }

  uint expert = 0u;
  bool found = false;
  for (uint tile = 0; tile < num_tiles; ++tile) {
    uint offset = uint(tile_offsets[tile]);
    uint count = uint(tile_counts[tile]);
    if (route >= offset && route < offset + count) {
      expert = uint(tile_experts[tile]);
      found = true;
      break;
    }
  }
  if (!found) {
    out[route * N + n] = half(0.0h);
    return;
  }

  uint codewords = K >> 3;
  uint groups = K / group_size;
  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    uint k_code = codeword_slot * 8u;
    uint code = uint(codes[(expert * N + n) * codewords + codeword_slot]);
    float scale_f = float(scales[(expert * N + n) * groups + (k_code / group_size)]);
    accum += float(sorted_x[route * K + k_code]) *
        mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
    accum += float(sorted_x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
  }
  out[route * N + n] = half(accum);
}

[[kernel]] void nax_e8p_packed_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device ushort* code_tile [[buffer(1)]],
    const device half* scale_tile [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    device half* out [[buffer(6)]],
    constant const uint& route_count [[buffer(7)]],
    constant const uint& output_count [[buffer(8)]],
    constant const uint& K [[buffer(9)]],
    constant const uint& codewords [[buffer(10)]],
    constant const uint& scale_groups [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint k_code = codeword_slot * 8u;
    uint code = uint(code_tile[n * codewords + codeword_slot]);
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_split_byte_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* sign_tile [[buffer(1)]],
    const device uchar* abs_index_tile [[buffer(2)]],
    const device uchar* parity_tile [[buffer(3)]],
    const device half* scale_tile [[buffer(4)]],
    const device int* scale_group_indices [[buffer(5)]],
    const device int* codeword_scale_slots [[buffer(6)]],
    const device uint* codebook [[buffer(7)]],
    device half* out [[buffer(8)]],
    constant const uint& route_count [[buffer(9)]],
    constant const uint& output_count [[buffer(10)]],
    constant const uint& K [[buffer(11)]],
    constant const uint& codewords [[buffer(12)]],
    constant const uint& scale_groups [[buffer(13)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint k_code = codeword_slot * 8u;
    uint tile_index = n * codewords + codeword_slot;
    uint signs = uint(sign_tile[tile_index]);
    uint abs_idx = uint(abs_index_tile[tile_index]);
    uint parity = uint(parity_tile[tile_index]) & 1u;
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* sign_low_nibble_tile [[buffer(1)]],
    const device uchar* sign_high_nibble_tile [[buffer(2)]],
    const device uchar* abs_index_tile [[buffer(3)]],
    const device uchar* parity_tile [[buffer(4)]],
    const device half* scale_tile [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_count [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& codewords [[buffer(13)]],
    constant const uint& scale_groups [[buffer(14)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint k_code = codeword_slot * 8u;
    uint tile_index = n * codewords + codeword_slot;
    uint low = uint(sign_low_nibble_tile[tile_index]) & 0xFu;
    uint high = uint(sign_high_nibble_tile[tile_index]) & 0xFu;
    uint signs = low | (high << 4u);
    uint abs_idx = uint(abs_index_tile[tile_index]);
    uint parity = uint(parity_tile[tile_index]) & 1u;
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device ulong* sign_bit_planes [[buffer(1)]],
    const device uchar* abs_index_tile [[buffer(2)]],
    const device half* scale_tile [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device uint* codebook [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& output_count [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& codewords [[buffer(11)]],
    constant const uint& scale_groups [[buffer(12)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint signs = 0u;
    for (uint bit = 0; bit < 8u; ++bit) {
      ulong mask = sign_bit_planes[codeword_slot * 8u + bit];
      signs |= uint((mask >> ulong(n)) & 1ul) << bit;
    }
    uint k_code = codeword_slot * 8u;
    uint tile_index = n * codewords + codeword_slot;
    uint abs_idx = uint(abs_index_tile[tile_index]);
    uint parity = mlx_vq_sign_parity8(signs);
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* sign_low_nibble_lut [[buffer(1)]],
    const device uchar* sign_low_nibble_slots [[buffer(2)]],
    const device uchar* sign_high_nibble_lut [[buffer(3)]],
    const device uchar* sign_high_nibble_slots [[buffer(4)]],
    const device uchar* abs_index_lut [[buffer(5)]],
    const device uchar* abs_index_slots [[buffer(6)]],
    const device half* scale_tile [[buffer(7)]],
    const device int* scale_group_indices [[buffer(8)]],
    const device int* codeword_scale_slots [[buffer(9)]],
    const device uint* codebook [[buffer(10)]],
    device half* out [[buffer(11)]],
    constant const uint& route_count [[buffer(12)]],
    constant const uint& output_count [[buffer(13)]],
    constant const uint& K [[buffer(14)]],
    constant const uint& codewords [[buffer(15)]],
    constant const uint& scale_groups [[buffer(16)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint k_code = codeword_slot * 8u;
    uint tile_index = n * codewords + codeword_slot;
    uint low_slot = uint(sign_low_nibble_slots[tile_index]);
    uint high_slot = uint(sign_high_nibble_slots[tile_index]);
    uint abs_slot = uint(abs_index_slots[tile_index]);
    uint low = uint(sign_low_nibble_lut[low_slot]) & 0xFu;
    uint high = uint(sign_high_nibble_lut[high_slot]) & 0xFu;
    uint signs = low | (high << 4u);
    uint abs_idx = uint(abs_index_lut[abs_slot]);
    uint parity = mlx_vq_sign_parity8(signs);
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_tile_matmul(
    const device half* x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tile [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_count [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& codewords [[buffer(13)]],
    constant const uint& scale_groups [[buffer(14)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_count || route >= route_count) {
    return;
  }

  float accum = 0.0f;
  for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
    int scale_slot_i = codeword_scale_slots[codeword_slot];
    if (scale_slot_i < 0) {
      continue;
    }
    uint scale_slot = uint(scale_slot_i);
    if (scale_slot >= scale_groups || scale_group_indices[scale_slot] < 0) {
      continue;
    }
    uint k_code = codeword_slot * 8u;
    uint tile_index = n * codewords + codeword_slot;
    uint sign_slot = uint(sign_byte_slots[tile_index]);
    uint abs_slot = uint(abs_index_slots[tile_index]);
    uint signs = uint(sign_byte_lut[sign_slot]);
    uint abs_idx = uint(abs_index_lut[abs_slot]);
    uint parity = mlx_vq_sign_parity8(signs);
    float scale_f = float(scale_tile[n * scale_groups + scale_slot]);
    accum += float(x[route * K + k_code]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
    accum += float(x[route * K + k_code + 1u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
    accum += float(x[route * K + k_code + 2u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
    accum += float(x[route * K + k_code + 3u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
    accum += float(x[route * K + k_code + 4u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
    accum += float(x[route * K + k_code + 5u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
    accum += float(x[route * K + k_code + 6u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
    accum += float(x[route * K + k_code + 7u]) *
        mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
  }
  out[route * output_count + n] = half(accum);
}

[[kernel]] void nax_e8p_packed_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint code = uint(code_tiles[rhs_base * codewords + codeword_slot]);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_split_byte_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_tiles [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device uchar* parity_tiles [[buffer(3)]],
    const device half* scale_tiles [[buffer(4)]],
    const device int* scale_group_indices [[buffer(5)]],
    const device int* codeword_scale_slots [[buffer(6)]],
    const device uint* codebook [[buffer(7)]],
    const device int* tile_experts [[buffer(8)]],
    const device int* tile_offsets [[buffer(9)]],
    const device int* tile_counts [[buffer(10)]],
    device half* out [[buffer(11)]],
    constant const uint& route_count [[buffer(12)]],
    constant const uint& output_dims [[buffer(13)]],
    constant const uint& K [[buffer(14)]],
    constant const uint& experts [[buffer(15)]],
    constant const uint& n_tiles [[buffer(16)]],
    constant const uint& k_blocks [[buffer(17)]],
    constant const uint& bn [[buffer(18)]],
    constant const uint& codewords [[buffer(19)]],
    constant const uint& scale_groups [[buffer(20)]],
    constant const uint& num_route_tiles [[buffer(21)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint signs = uint(sign_tiles[factor_index]);
      uint abs_idx = uint(abs_index_tiles[factor_index]);
      uint parity = uint(parity_tiles[factor_index]) & 1u;
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_low_nibble_tiles [[buffer(1)]],
    const device uchar* sign_high_nibble_tiles [[buffer(2)]],
    const device uchar* abs_index_tiles [[buffer(3)]],
    const device uchar* parity_tiles [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint low = uint(sign_low_nibble_tiles[factor_index]) & 0xFu;
      uint high = uint(sign_high_nibble_tiles[factor_index]) & 0xFu;
      uint signs = low | (high << 4u);
      uint abs_idx = uint(abs_index_tiles[factor_index]);
      uint parity = uint(parity_tiles[factor_index]) & 1u;
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ulong* sign_bit_planes [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device uint* codebook [[buffer(6)]],
    const device int* tile_experts [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_counts [[buffer(9)]],
    device half* out [[buffer(10)]],
    constant const uint& route_count [[buffer(11)]],
    constant const uint& output_dims [[buffer(12)]],
    constant const uint& K [[buffer(13)]],
    constant const uint& experts [[buffer(14)]],
    constant const uint& n_tiles [[buffer(15)]],
    constant const uint& k_blocks [[buffer(16)]],
    constant const uint& bn [[buffer(17)]],
    constant const uint& codewords [[buffer(18)]],
    constant const uint& scale_groups [[buffer(19)]],
    constant const uint& num_route_tiles [[buffer(20)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint sign_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * codewords +
           codeword_slot) *
          8u;
      uint signs = 0u;
      for (uint bit = 0; bit < 8u; ++bit) {
        ulong mask = sign_bit_planes[sign_base + bit];
        signs |= uint((mask >> ulong(n_in_tile)) & 1ul) << bit;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint abs_idx = uint(abs_index_tiles[factor_index]);
      uint parity = mlx_vq_sign_parity8(signs);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ulong* sign_bit_planes [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device uint* codebook [[buffer(6)]],
    const device int* tile_experts [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* tile_counts [[buffer(9)]],
    device half* out [[buffer(10)]],
    constant const uint& route_count [[buffer(11)]],
    constant const uint& output_dims [[buffer(12)]],
    constant const uint& K [[buffer(13)]],
    constant const uint& experts [[buffer(14)]],
    constant const uint& n_tiles [[buffer(15)]],
    constant const uint& k_blocks [[buffer(16)]],
    constant const uint& bn [[buffer(17)]],
    constant const uint& codewords [[buffer(18)]],
    constant const uint& scale_groups [[buffer(19)]],
    constant const uint& num_route_tiles [[buffer(20)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[k_block * scale_groups + scale_slot] >=
                      0) {
                uint sign_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) *
                         codewords +
                     codeword_slot) *
                    8u;
                uint signs = 0u;
                for (uint bit = 0; bit < 8u; ++bit) {
                  ulong mask = sign_bit_planes[sign_base + bit];
                  signs |= uint((mask >> ulong(n_local)) & 1ul) << bit;
                }
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint abs_idx = uint(abs_index_tiles[factor_index]);
                uint parity = mlx_vq_sign_parity8(signs);
                float scale_f =
                    float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                value = half(
                    mlx_vq_decode_e8p_split_value(
                        signs, abs_idx, parity, codebook, dim) *
                    scale_f);
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_low_nibble_lut [[buffer(1)]],
    const device uchar* sign_low_nibble_slots [[buffer(2)]],
    const device uchar* sign_high_nibble_lut [[buffer(3)]],
    const device uchar* sign_high_nibble_slots [[buffer(4)]],
    const device uchar* abs_index_lut [[buffer(5)]],
    const device uchar* abs_index_slots [[buffer(6)]],
    const device half* scale_tiles [[buffer(7)]],
    const device int* scale_group_indices [[buffer(8)]],
    const device int* codeword_scale_slots [[buffer(9)]],
    const device uint* codebook [[buffer(10)]],
    const device int* tile_experts [[buffer(11)]],
    const device int* tile_offsets [[buffer(12)]],
    const device int* tile_counts [[buffer(13)]],
    device half* out [[buffer(14)]],
    constant const uint& route_count [[buffer(15)]],
    constant const uint& output_dims [[buffer(16)]],
    constant const uint& K [[buffer(17)]],
    constant const uint& experts [[buffer(18)]],
    constant const uint& n_tiles [[buffer(19)]],
    constant const uint& k_blocks [[buffer(20)]],
    constant const uint& bn [[buffer(21)]],
    constant const uint& codewords [[buffer(22)]],
    constant const uint& scale_groups [[buffer(23)]],
    constant const uint& num_route_tiles [[buffer(24)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint low_slot = uint(sign_low_nibble_slots[factor_index]);
      uint high_slot = uint(sign_high_nibble_slots[factor_index]);
      if (low_slot >= 16u || high_slot >= 16u) {
        continue;
      }
      uint low = uint(sign_low_nibble_lut[low_slot]) & 0xFu;
      uint high = uint(sign_high_nibble_lut[high_slot]) & 0xFu;
      uint signs = low | (high << 4u);
      uint abs_slot = uint(abs_index_slots[factor_index]);
      uint abs_base =
          ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
      uint abs_idx = uint(abs_index_lut[abs_base + abs_slot]);
      uint parity = mlx_vq_sign_parity8(signs);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_low_nibble_tiles [[buffer(1)]],
    const device uchar* sign_high_nibble_tiles [[buffer(2)]],
    const device uchar* abs_index_tiles [[buffer(3)]],
    const device uchar* parity_tiles [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[k_block * scale_groups + scale_slot] >=
                      0) {
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint low = uint(sign_low_nibble_tiles[factor_index]) & 0xFu;
                uint high = uint(sign_high_nibble_tiles[factor_index]) & 0xFu;
                uint abs_idx = uint(abs_index_tiles[factor_index]);
                uint parity = uint(parity_tiles[factor_index]) & 1u;
                float scale_f =
                    float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                value = half(
                    mlx_vq_decode_e8p_nibble_value(
                        low, high, abs_idx, parity, codebook, dim) *
                    scale_f);
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_low_nibble_lut [[buffer(1)]],
    const device uchar* sign_low_nibble_slots [[buffer(2)]],
    const device uchar* sign_high_nibble_lut [[buffer(3)]],
    const device uchar* sign_high_nibble_slots [[buffer(4)]],
    const device uchar* abs_index_lut [[buffer(5)]],
    const device uchar* abs_index_slots [[buffer(6)]],
    const device half* scale_tiles [[buffer(7)]],
    const device int* scale_group_indices [[buffer(8)]],
    const device int* codeword_scale_slots [[buffer(9)]],
    const device uint* codebook [[buffer(10)]],
    const device int* tile_experts [[buffer(11)]],
    const device int* tile_offsets [[buffer(12)]],
    const device int* tile_counts [[buffer(13)]],
    device half* out [[buffer(14)]],
    constant const uint& route_count [[buffer(15)]],
    constant const uint& output_dims [[buffer(16)]],
    constant const uint& K [[buffer(17)]],
    constant const uint& experts [[buffer(18)]],
    constant const uint& n_tiles [[buffer(19)]],
    constant const uint& k_blocks [[buffer(20)]],
    constant const uint& bn [[buffer(21)]],
    constant const uint& codewords [[buffer(22)]],
    constant const uint& scale_groups [[buffer(23)]],
    constant const uint& num_route_tiles [[buffer(24)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[k_block * scale_groups + scale_slot] >=
                      0) {
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint low_slot = uint(sign_low_nibble_slots[factor_index]);
                uint high_slot = uint(sign_high_nibble_slots[factor_index]);
                if (low_slot < 16u && high_slot < 16u) {
                  uint low = uint(sign_low_nibble_lut[low_slot]) & 0xFu;
                  uint high = uint(sign_high_nibble_lut[high_slot]) & 0xFu;
                  uint signs = low | (high << 4u);
                  uint abs_slot = uint(abs_index_slots[factor_index]);
                  uint abs_base =
                      ((expert * n_tiles + n_tile) * k_blocks + k_block) *
                      256u;
                  uint abs_idx = uint(abs_index_lut[abs_base + abs_slot]);
                  uint parity = mlx_vq_sign_parity8(signs);
                  float scale_f =
                      float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                  value = half(
                      mlx_vq_decode_e8p_split_value(
                          signs, abs_idx, parity, codebook, dim) *
                      scale_f);
                }
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint lut_base = ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
      uint sign_slot = uint(sign_byte_slots[factor_index]);
      uint abs_slot = uint(abs_index_slots[factor_index]);
      uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
      uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
      uint parity = mlx_vq_sign_parity8(signs);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_slot = 0; codeword_slot < codewords; ++codeword_slot) {
      uint map_offset = k_block * codewords + codeword_slot;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_slot * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint factor_index = rhs_base * codewords + codeword_slot;
      uint lut_base = (expert * k_blocks + k_block) * 256u;
      uint sign_slot = uint(sign_byte_slots[factor_index]);
      uint abs_slot = uint(abs_index_slots[factor_index]);
      uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
      uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
      uint parity = mlx_vq_sign_parity8(signs);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_component_stream_rhs_sorted_scalar_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_component_bits [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device int* component_scale_slots [[buffer(6)]],
    const device int* component_codeword_indices [[buffer(7)]],
    const device int* component_offsets [[buffer(8)]],
    const device uint* codebook [[buffer(9)]],
    const device int* tile_experts [[buffer(10)]],
    const device int* tile_offsets [[buffer(11)]],
    const device int* tile_counts [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& components [[buffer(22)]],
    constant const uint& scale_groups [[buffer(23)]],
    constant const uint& num_route_tiles [[buffer(24)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    uint map_base = k_block * components;
    uint rhs_base =
        (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
    for (uint component = 0; component < components; ++component) {
      int codeword_i = component_codeword_indices[map_base + component];
      int component_offset_i = component_offsets[map_base + component];
      int scale_slot_i = component_scale_slots[map_base + component];
      if (codeword_i < 0 || component_offset_i < 0 || scale_slot_i < 0) {
        continue;
      }
      uint codeword_slot = uint(codeword_i);
      uint component_offset = uint(component_offset_i);
      uint scale_slot = uint(scale_slot_i);
      if (codeword_slot >= codewords || component_offset >= 8u ||
          scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0 ||
          codeword_scale_slots[k_block * codewords + codeword_slot] < 0) {
        continue;
      }

      uint k_component = k_block * codewords * 8u + component;
      if (k_component >= K) {
        continue;
      }

      uint sign_base = (rhs_base * codewords + codeword_slot) * 8u;
      uint signs = 0u;
      for (uint bit = 0; bit < 8u; ++bit) {
        signs |= uint(sign_component_bits[sign_base + bit] & 1u) << bit;
      }
      uint abs_idx = uint(abs_index_tiles[rhs_base * codewords + codeword_slot]);
      uint parity = mlx_vq_sign_parity8(signs);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      accum += float(sorted_x[route * K + k_component]) *
          mlx_vq_decode_e8p_split_value(
              signs, abs_idx, parity, codebook, component_offset) *
          scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint3 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route_slot = tid.y;
  uint route_tile = tid.z;
  if (n >= output_dims || route_tile >= num_route_tiles) {
    return;
  }

  int routes_in_tile_i = tile_counts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int expert_i = tile_experts[route_tile];
  if (routes_in_tile_i <= 0 || route_base_i < 0 || expert_i < 0 ||
      route_slot >= uint(routes_in_tile_i) || uint(expert_i) >= experts) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword = 0; codeword < codewords; ++codeword) {
      uint map_offset = k_block * codewords + codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint code = uint(code_tiles[rhs_base * codewords + codeword]);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_tile = threadgroup_position.x;
  uint output_tile = threadgroup_position.y;
  uint n_in_output_tile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (n >= output_dims || route_tile >= num_route_tiles) {
    return;
  }

  int routes_in_tile_i = tile_counts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int expert_i = tile_experts[route_tile];
  if (routes_in_tile_i <= 0 || route_base_i < 0 || expert_i < 0 ||
      route_slot >= uint(routes_in_tile_i) || uint(expert_i) >= experts) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codewords; ++codeword_tile) {
      uint map_offset = k_block * codewords + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint code = uint(code_tiles[rhs_base * codewords + codeword_tile]);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_component_stream_rhs_sorted_shared_decode_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_component_bits [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device int* component_scale_slots [[buffer(6)]],
    const device int* component_codeword_indices [[buffer(7)]],
    const device int* component_offsets [[buffer(8)]],
    const device uint* codebook [[buffer(9)]],
    const device int* tile_experts [[buffer(10)]],
    const device int* tile_offsets [[buffer(11)]],
    const device int* tile_counts [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& components [[buffer(22)]],
    constant const uint& scale_groups [[buffer(23)]],
    constant const uint& num_route_tiles [[buffer(24)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint route = tid.y;
  if (n >= output_dims || route >= route_count) {
    return;
  }

  int route_tile = -1;
  for (uint tile = 0; tile < num_route_tiles; ++tile) {
    int offset = tile_offsets[tile];
    int count = tile_counts[tile];
    if (offset >= 0 && count > 0 && int(route) >= offset &&
        int(route) < offset + count) {
      route_tile = int(tile);
      break;
    }
  }
  if (route_tile < 0) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  int expert_i = tile_experts[uint(route_tile)];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  constexpr uint component_pair_width = 4u;
  float component_decode_cache[component_pair_width];
  float component_scale_cache[component_pair_width];
  uint component_k_cache[component_pair_width];

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    uint map_base = k_block * components;
    uint rhs_base =
        (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
    for (uint component_pair = 0; component_pair < components;
         component_pair += component_pair_width) {
      for (uint component_lane = 0; component_lane < component_pair_width;
           ++component_lane) {
        component_decode_cache[component_lane] = 0.0f;
        component_scale_cache[component_lane] = 0.0f;
        component_k_cache[component_lane] = K;
        uint component = component_pair + component_lane;
        if (component >= components) {
          continue;
        }
        int codeword_i = component_codeword_indices[map_base + component];
        int component_offset_i = component_offsets[map_base + component];
        int scale_slot_i = component_scale_slots[map_base + component];
        if (codeword_i < 0 || component_offset_i < 0 || scale_slot_i < 0) {
          continue;
        }
        uint codeword_slot = uint(codeword_i);
        uint component_offset = uint(component_offset_i);
        uint scale_slot = uint(scale_slot_i);
        if (codeword_slot >= codewords || component_offset >= 8u ||
            scale_slot >= scale_groups ||
            scale_group_indices[k_block * scale_groups + scale_slot] < 0 ||
            codeword_scale_slots[k_block * codewords + codeword_slot] < 0) {
          continue;
        }

        uint k_component = k_block * codewords * 8u + component;
        if (k_component >= K) {
          continue;
        }

        uint sign_base = (rhs_base * codewords + codeword_slot) * 8u;
        uint signs = 0u;
        for (uint bit = 0; bit < 8u; ++bit) {
          signs |= uint(sign_component_bits[sign_base + bit] & 1u) << bit;
        }
        uint abs_idx = uint(abs_index_tiles[rhs_base * codewords + codeword_slot]);
        uint parity = mlx_vq_sign_parity8(signs);
        component_decode_cache[component_lane] = mlx_vq_decode_e8p_split_value(
            signs, abs_idx, parity, codebook, component_offset);
        component_scale_cache[component_lane] =
            float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        component_k_cache[component_lane] = k_component;
      }

      for (uint component_lane = 0; component_lane < component_pair_width;
           ++component_lane) {
        uint k_component = component_k_cache[component_lane];
        if (k_component < K) {
          accum += float(sorted_x[route * K + k_component]) *
              component_decode_cache[component_lane] *
              component_scale_cache[component_lane];
        }
      }
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* active_route_tiles [[buffer(9)]],
    device half* out [[buffer(10)]],
    constant const uint& route_count [[buffer(11)]],
    constant const uint& output_dims [[buffer(12)]],
    constant const uint& K [[buffer(13)]],
    constant const uint& experts [[buffer(14)]],
    constant const uint& n_tiles [[buffer(15)]],
    constant const uint& k_blocks [[buffer(16)]],
    constant const uint& bn [[buffer(17)]],
    constant const uint& codewords [[buffer(18)]],
    constant const uint& scale_groups [[buffer(19)]],
    constant const uint& num_route_tiles [[buffer(20)]],
    constant const uint& active_route_tile_count [[buffer(21)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint active_route_tile_index = threadgroup_position.x;
  uint output_microtile = threadgroup_position.y;
  uint n_in_output_microtile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_microtile * 64u + n_in_output_microtile;
  if (n >= output_dims || active_route_tile_index >= active_route_tile_count) {
    return;
  }

  int active_route_tile_i = active_route_tiles[active_route_tile_index];
  if (active_route_tile_i < 0 || uint(active_route_tile_i) >= num_route_tiles) {
    return;
  }
  uint active_route_tile = uint(active_route_tile_i);
  uint route_tile = active_route_tile;
  int routes_in_tile_i = tile_counts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int expert_i = tile_experts[route_tile];
  if (routes_in_tile_i <= 0 || route_base_i < 0 || expert_i < 0 ||
      route_slot >= uint(routes_in_tile_i) || uint(expert_i) >= experts) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword = 0; codeword < codewords; ++codeword) {
      uint map_offset = k_block * codewords + codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint code = uint(code_tiles[rhs_base * codewords + codeword]);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* expert_cohort_offsets [[buffer(9)]],
    const device int* expert_cohort_counts [[buffer(10)]],
    const device int* route_cohort_offsets [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& expert_cohort_count [[buffer(23)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint expert = threadgroup_position.x;
  uint cohort_slot = threadgroup_position.y;
  uint output_microtile = threadgroup_position.z;
  uint n_in_output_microtile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_microtile * 64u + n_in_output_microtile;
  if (expert >= experts || n >= output_dims) {
    return;
  }

  int cohort_base_i = expert_cohort_offsets[expert];
  int cohort_count_i = expert_cohort_counts[expert];
  if (cohort_base_i < 0 || cohort_count_i <= 0 ||
      cohort_slot >= uint(cohort_count_i)) {
    return;
  }
  uint cohort_index = uint(cohort_base_i) + cohort_slot;
  if (cohort_index >= expert_cohort_count) {
    return;
  }
  int route_tile_i = route_cohort_offsets[cohort_index];
  if (route_tile_i < 0 || uint(route_tile_i) >= num_route_tiles) {
    return;
  }

  uint route_tile = uint(route_tile_i);
  int routes_in_tile_i = tile_counts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int tile_expert_i = tile_experts[route_tile];
  if (routes_in_tile_i <= 0 || route_base_i < 0 || tile_expert_i < 0 ||
      uint(tile_expert_i) != expert || route_slot >= uint(routes_in_tile_i)) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword = 0; codeword < codewords; ++codeword) {
      uint map_offset = k_block * codewords + codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint compressed_codeword = uint(code_tiles[rhs_base * codewords + codeword]);
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(compressed_codeword, codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_batch_segment_offsets [[buffer(9)]],
    const device int* route_batch_segment_counts [[buffer(10)]],
    const device int* route_batch_route_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_batch_count [[buffer(23)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_batch = threadgroup_position.x;
  uint scheduled_k_block = threadgroup_position.y;
  uint output_microtile = threadgroup_position.z;
  uint n_in_output_microtile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_microtile * 64u + n_in_output_microtile;
  if (route_batch >= route_batch_count || n >= output_dims ||
      scheduled_k_block != 0u) {
    return;
  }

  int segment_offset_i = route_batch_segment_offsets[route_batch];
  int segment_count_i = route_batch_segment_counts[route_batch];
  if (segment_offset_i < 0 || segment_count_i <= 0 ||
      route_slot >= uint(segment_count_i)) {
    return;
  }
  int route_i = route_batch_route_ids[uint(segment_offset_i) + route_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  int expert_i = -1;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int routes_in_tile_i = tile_counts[route_tile];
    int route_base_i = tile_offsets[route_tile];
    if (routes_in_tile_i <= 0 || route_base_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint routes_in_tile = uint(routes_in_tile_i);
    if (route >= route_base && route < route_base + routes_in_tile) {
      expert_i = tile_experts[route_tile];
      break;
    }
  }
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword = 0; codeword < codewords; ++codeword) {
      uint map_offset = k_block * codewords + codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      ushort compressed_codeword = code_tiles[rhs_base * codewords + codeword];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_cohort_offsets [[buffer(9)]],
    const device int* token_cohort_counts [[buffer(10)]],
    const device int* token_cohort_active_expert_ids [[buffer(11)]],
    const device int* token_cohort_route_slot_ids [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& scale_groups [[buffer(22)]],
    constant const uint& num_route_tiles [[buffer(23)]],
    constant const uint& token_cohort_count [[buffer(24)]],
    constant const uint& active_expert_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_cohort = threadgroup_position.x;
  uint active_expert_slot = threadgroup_position.y;
  uint output_microtile = threadgroup_position.z;
  uint n_in_output_microtile = thread_position.x;
  uint cohort_route_slot = thread_position.y;
  uint n = output_microtile * 64u + n_in_output_microtile;
  if (token_cohort >= token_cohort_count ||
      active_expert_slot >= active_expert_count || n >= output_dims) {
    return;
  }

  int cohort_base_i = token_cohort_offsets[token_cohort];
  int cohort_count_i = token_cohort_counts[token_cohort];
  if (cohort_base_i < 0 || cohort_count_i <= 0 ||
      cohort_route_slot >= uint(cohort_count_i)) {
    return;
  }
  int route_i =
      token_cohort_route_slot_ids[uint(cohort_base_i) + cohort_route_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  int active_expert_i = token_cohort_active_expert_ids[active_expert_slot];
  if (active_expert_i < 0 || uint(active_expert_i) >= experts) {
    return;
  }
  uint active_expert = uint(active_expert_i);

  bool route_matches_active_expert = false;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int route_count_i = tile_counts[route_tile];
    if (route_base_i < 0 || route_count_i <= 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(route_count_i);
    if (route >= route_base && route < route_limit &&
        tile_experts[route_tile] == active_expert_i) {
      route_matches_active_expert = true;
      break;
    }
  }
  if (!route_matches_active_expert) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword = 0; codeword < codewords; ++codeword) {
      uint map_offset = k_block * codewords + codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword = code_tiles[rhs_base * codewords + codeword];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 0u) * scale_f;
      accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 1u) * scale_f;
      accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 2u) * scale_f;
      accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 3u) * scale_f;
      accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 4u) * scale_f;
      accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 5u) * scale_f;
      accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 6u) * scale_f;
      accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(accum);
}

[[kernel]] void nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_cohort_offsets [[buffer(9)]],
    const device int* token_cohort_counts [[buffer(10)]],
    const device int* token_cohort_active_expert_ids [[buffer(11)]],
    const device int* token_cohort_route_slot_ids [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& scale_groups [[buffer(22)]],
    constant const uint& num_route_tiles [[buffer(23)]],
    constant const uint& token_cohort_count [[buffer(24)]],
    constant const uint& active_expert_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_cohort = threadgroup_position.x;
  uint output_tile = threadgroup_position.y;
  uint active_k_codeword = threadgroup_position.z;
  uint active_expert_slot = active_k_codeword / (k_blocks * codewords);
  uint k_codeword_rem = active_k_codeword - active_expert_slot * k_blocks * codewords;
  uint scheduled_k_block = k_codeword_rem / codewords;
  uint scheduled_codeword_tile = k_codeword_rem - scheduled_k_block * codewords;
  uint n_in_output_tile = thread_position.x;
  uint cohort_route_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (token_cohort >= token_cohort_count ||
      active_expert_slot >= active_expert_count || n >= output_dims) {
    return;
  }

  int cohort_base_i = token_cohort_offsets[token_cohort];
  int cohort_count_i = token_cohort_counts[token_cohort];
  if (cohort_base_i < 0 || cohort_count_i <= 0 ||
      cohort_route_slot >= uint(cohort_count_i)) {
    return;
  }
  int route_i =
      token_cohort_route_slot_ids[uint(cohort_base_i) + cohort_route_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  int active_expert_i = token_cohort_active_expert_ids[active_expert_slot];
  if (active_expert_i < 0 || uint(active_expert_i) >= experts) {
    return;
  }
  uint active_expert = uint(active_expert_i);

  bool route_matches_active_expert = false;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int route_count_i = tile_counts[route_tile];
    if (route_base_i < 0 || route_count_i <= 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(route_count_i);
    if (route >= route_base && route < route_limit &&
        tile_experts[route_tile] == active_expert_i) {
      route_matches_active_expert = true;
      break;
    }
  }
  if (!route_matches_active_expert) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || scheduled_codeword_tile != 0u) {
    return;
  }

  float mma_accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codewords; ++codeword_tile) {
      uint map_offset = k_block * codewords + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + codeword_tile];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      mma_accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u) * scale_f;
      mma_accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u) * scale_f;
      mma_accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u) * scale_f;
      mma_accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u) * scale_f;
      mma_accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u) * scale_f;
      mma_accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u) * scale_f;
      mma_accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u) * scale_f;
      mma_accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(mma_accum);
}

[[kernel]] void nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* output_stationary_route_batch_offsets [[buffer(9)]],
    const device int* output_stationary_route_batch_counts [[buffer(10)]],
    const device int* output_stationary_route_batch_active_expert_ids [[buffer(11)]],
    const device int* output_stationary_route_batch_route_slot_ids [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& scale_groups [[buffer(22)]],
    constant const uint& num_route_tiles [[buffer(23)]],
    constant const uint& route_batch_count [[buffer(24)]],
    constant const uint& active_expert_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint output_tile = threadgroup_position.x;
  uint route_batch = threadgroup_position.y;
  uint active_k_codeword = threadgroup_position.z;
  uint active_expert_slot = active_k_codeword / (k_blocks * codewords);
  uint k_codeword_rem = active_k_codeword - active_expert_slot * k_blocks * codewords;
  uint scheduled_k_block = k_codeword_rem / codewords;
  uint scheduled_codeword_tile = k_codeword_rem - scheduled_k_block * codewords;
  uint n_in_output_tile = thread_position.x;
  uint route_batch_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (output_tile >= ((output_dims + 63u) / 64u) ||
      route_batch >= route_batch_count ||
      active_expert_slot >= active_expert_count || n >= output_dims) {
    return;
  }

  int batch_base_i = output_stationary_route_batch_offsets[route_batch];
  int batch_count_i = output_stationary_route_batch_counts[route_batch];
  if (batch_base_i < 0 || batch_count_i <= 0 ||
      route_batch_slot >= uint(batch_count_i)) {
    return;
  }
  int route_i =
      output_stationary_route_batch_route_slot_ids[uint(batch_base_i) + route_batch_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  int active_expert_i =
      output_stationary_route_batch_active_expert_ids[active_expert_slot];
  if (active_expert_i < 0 || uint(active_expert_i) >= experts) {
    return;
  }
  uint active_expert = uint(active_expert_i);

  bool route_matches_active_expert = false;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int route_count_i = tile_counts[route_tile];
    if (route_base_i < 0 || route_count_i <= 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(route_count_i);
    if (route >= route_base && route < route_limit &&
        tile_experts[route_tile] == active_expert_i) {
      route_matches_active_expert = true;
      break;
    }
  }
  if (!route_matches_active_expert) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || scheduled_codeword_tile != 0u) {
    return;
  }

  float output_stationary_accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codewords; ++codeword_tile) {
      uint map_offset = k_block * codewords + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + codeword_tile];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      output_stationary_accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u) * scale_f;
      output_stationary_accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(output_stationary_accum);
}

[[kernel]] void nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* input_stationary_route_batch_offsets [[buffer(9)]],
    const device int* input_stationary_route_batch_counts [[buffer(10)]],
    const device int* input_stationary_route_batch_active_expert_ids [[buffer(11)]],
    const device int* input_stationary_route_batch_route_slot_ids [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& scale_groups [[buffer(22)]],
    constant const uint& num_route_tiles [[buffer(23)]],
    constant const uint& route_batch_count [[buffer(24)]],
    constant const uint& active_expert_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint input_tile = threadgroup_position.x;
  uint route_batch = threadgroup_position.y;
  uint active_output_codeword = threadgroup_position.z;
  uint output_tile_count = (output_dims + 63u) / 64u;
  uint active_expert_slot = active_output_codeword / (output_tile_count * codewords);
  uint output_codeword_rem =
      active_output_codeword - active_expert_slot * output_tile_count * codewords;
  uint output_tile = output_codeword_rem / codewords;
  uint scheduled_codeword_tile =
      output_codeword_rem - output_tile * codewords;
  uint n_in_output_tile = thread_position.x;
  uint route_batch_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (input_tile >= k_blocks || route_batch >= route_batch_count ||
      active_expert_slot >= active_expert_count ||
      output_tile >= output_tile_count || n >= output_dims) {
    return;
  }

  int batch_base_i = input_stationary_route_batch_offsets[route_batch];
  int batch_count_i = input_stationary_route_batch_counts[route_batch];
  if (batch_base_i < 0 || batch_count_i <= 0 ||
      route_batch_slot >= uint(batch_count_i)) {
    return;
  }
  int route_i =
      input_stationary_route_batch_route_slot_ids[uint(batch_base_i) + route_batch_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  int active_expert_i =
      input_stationary_route_batch_active_expert_ids[active_expert_slot];
  if (active_expert_i < 0 || uint(active_expert_i) >= experts) {
    return;
  }
  uint active_expert = uint(active_expert_i);

  bool route_matches_active_expert = false;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int route_count_i = tile_counts[route_tile];
    if (route_base_i < 0 || route_count_i <= 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(route_count_i);
    if (route >= route_base && route < route_limit &&
        tile_experts[route_tile] == active_expert_i) {
      route_matches_active_expert = true;
      break;
    }
  }
  if (!route_matches_active_expert) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (input_tile != 0u || scheduled_codeword_tile != 0u) {
    return;
  }

  float input_stationary_accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codewords; ++codeword_tile) {
      uint map_offset = k_block * codewords + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + codeword_tile];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      input_stationary_accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u) * scale_f;
      input_stationary_accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(input_stationary_accum);
}

[[kernel]] void nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_factor_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& expert_count [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_block_count [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codeword_tile_count [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& route_tile_count [[buffer(19)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint expert = threadgroup_position.x;
  uint scheduled_k_block = threadgroup_position.y;
  uint output_tile_count = (output_dims + 63u) / 64u;
  uint codeword_route_output =
      threadgroup_position.z;
  uint codeword_tile = codeword_route_output % codeword_tile_count;
  uint route_output = codeword_route_output / codeword_tile_count;
  uint route_tile = route_output % route_tile_count;
  uint output_tile = route_output / route_tile_count;
  uint n_in_output_tile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (expert >= expert_count || scheduled_k_block >= k_block_count ||
      output_tile >= output_tile_count || route_tile >= route_tile_count ||
      codeword_tile >= codeword_tile_count || n >= output_dims) {
    return;
  }

  int tile_expert_i = tile_experts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int routes_in_tile_i = tile_counts[route_tile];
  if (tile_expert_i < 0 || uint(tile_expert_i) != expert ||
      route_base_i < 0 || routes_in_tile_i <= 0 ||
      route_slot >= uint(routes_in_tile_i)) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || codeword_tile != 0u) {
    return;
  }

  float expert_kblock_codeword_factor_accum = 0.0f;
  for (uint k_block = 0; k_block < k_block_count; ++k_block) {
    for (uint factor_codeword_tile = 0; factor_codeword_tile < codeword_tile_count;
         ++factor_codeword_tile) {
      uint map_offset = k_block * codeword_tile_count + factor_codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code =
          k_block * codeword_tile_count * 8u + factor_codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_block_count + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_factor_tile =
          codeword_factor_tiles[rhs_base * codeword_tile_count +
                                factor_codeword_tile];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 0u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 1u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 2u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 3u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 4u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 5u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 6u) * scale_f;
      expert_kblock_codeword_factor_accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_factor_tile), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(expert_kblock_codeword_factor_accum);
}

[[kernel]] void nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& expert_count [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_block_count [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codeword_tile_count [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& route_tile_count [[buffer(19)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint expert = threadgroup_position.x;
  uint scheduled_k_block = threadgroup_position.y;
  uint output_tile_count = (output_dims + 63u) / 64u;
  uint scale_group_route_output = threadgroup_position.z;
  uint scale_group = scale_group_route_output % scale_groups;
  uint route_output = scale_group_route_output / scale_groups;
  uint route_tile = route_output % route_tile_count;
  uint output_tile = route_output / route_tile_count;
  uint n_in_output_tile = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (expert >= expert_count || scheduled_k_block >= k_block_count ||
      output_tile >= output_tile_count || route_tile >= route_tile_count ||
      scale_group >= scale_groups || n >= output_dims) {
    return;
  }

  int tile_expert_i = tile_experts[route_tile];
  int route_base_i = tile_offsets[route_tile];
  int routes_in_tile_i = tile_counts[route_tile];
  if (tile_expert_i < 0 || uint(tile_expert_i) != expert ||
      route_base_i < 0 || routes_in_tile_i <= 0 ||
      route_slot >= uint(routes_in_tile_i)) {
    return;
  }
  uint route = uint(route_base_i) + route_slot;
  if (route >= route_count) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint scale_slot_stream = scale_group;
  if (scheduled_k_block != 0u || scale_slot_stream != 0u) {
    return;
  }

  float expert_kblock_scale_slot_accum = 0.0f;
  for (uint k_block = 0; k_block < k_block_count; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codeword_tile_count;
         ++codeword_tile) {
      uint map_offset = k_block * codeword_tile_count + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_slot = uint(scale_slot_i);
      if (active_scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + active_scale_slot] < 0) {
        continue;
      }

      uint k_code =
          k_block * codeword_tile_count * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_block_count + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_tile_count + codeword_tile];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_slot]);
      uint x_base = route * K + k_code;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u) * scale_f;
      expert_kblock_scale_slot_accum += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u) * scale_f;
    }
  }
  out[route * output_dims + n] = half(expert_kblock_scale_slot_accum);
}

[[kernel]] void nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* rowwise_route_microtile_offsets [[buffer(9)]],
    const device int* rowwise_route_microtile_counts [[buffer(10)]],
    const device int* rowwise_route_microtile_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_microtile_count [[buffer(23)]],
    constant const uint& route_microtile_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_microtile = threadgroup_position.x;
  uint output_tile = threadgroup_position.y;
  uint scheduled_codeword_axis = threadgroup_position.z;
  uint scheduled_k_block = scheduled_codeword_axis / codewords;
  uint scheduled_codeword_tile =
      scheduled_codeword_axis - scheduled_k_block * codewords;
  uint n_in_output_tile = thread_position.x;
  uint route_microtile_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (route_microtile >= route_microtile_count ||
      output_tile >= (output_dims + 63u) / 64u ||
      scheduled_k_block >= k_blocks ||
      scheduled_codeword_tile >= codewords || n >= output_dims) {
    return;
  }

  int microtile_base_i = rowwise_route_microtile_offsets[route_microtile];
  int microtile_count_i = rowwise_route_microtile_counts[route_microtile];
  if (microtile_base_i < 0 || microtile_count_i <= 0 ||
      route_microtile_slot >= uint(microtile_count_i)) {
    return;
  }
  uint microtile_index = uint(microtile_base_i) + route_microtile_slot;
  if (microtile_index >= route_microtile_slot_count) {
    return;
  }

  int route_i = rowwise_route_microtile_route_slot_ids[microtile_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || scheduled_codeword_tile != 0u) {
    return;
  }

  float rowwise_codeword_accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint codeword_tile = 0; codeword_tile < codewords; ++codeword_tile) {
      uint map_offset = k_block * codewords + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + codeword_tile];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      float online_activation_codeword_dot = 0.0f;
      online_activation_codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      online_activation_codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      online_activation_codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      online_activation_codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      online_activation_codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      online_activation_codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      online_activation_codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      online_activation_codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      rowwise_codeword_accum += online_activation_codeword_dot * scale_f;
    }
  }
  out[route * output_dims + n] = half(rowwise_codeword_accum);
}

[[kernel]] void nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* output_tile_local_route_microtile_offsets [[buffer(9)]],
    const device int* output_tile_local_route_microtile_counts [[buffer(10)]],
    const device int* output_tile_local_route_microtile_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_microtile_count [[buffer(23)]],
    constant const uint& route_microtile_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_microtile = threadgroup_position.x;
  uint output_tile = threadgroup_position.y;
  uint scheduled_unique_codeword_axis = threadgroup_position.z;
  uint scheduled_k_block = scheduled_unique_codeword_axis / codewords;
  uint output_tile_local_unique_codeword_lut =
      scheduled_unique_codeword_axis - scheduled_k_block * codewords;
  uint n_in_output_tile = thread_position.x;
  uint route_microtile_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (route_microtile >= route_microtile_count ||
      output_tile >= (output_dims + 63u) / 64u ||
      scheduled_k_block >= k_blocks ||
      output_tile_local_unique_codeword_lut >= codewords || n >= output_dims) {
    return;
  }

  int microtile_base_i =
      output_tile_local_route_microtile_offsets[route_microtile];
  int microtile_count_i =
      output_tile_local_route_microtile_counts[route_microtile];
  if (microtile_base_i < 0 || microtile_count_i <= 0 ||
      route_microtile_slot >= uint(microtile_count_i)) {
    return;
  }
  uint microtile_index = uint(microtile_base_i) + route_microtile_slot;
  if (microtile_index >= route_microtile_slot_count) {
    return;
  }

  int route_i =
      output_tile_local_route_microtile_route_slot_ids[microtile_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || output_tile_local_unique_codeword_lut != 0u) {
    return;
  }

  float output_tile_local_codeword_lut_accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint unique_codeword = 0; unique_codeword < codewords;
         ++unique_codeword) {
      uint map_offset = k_block * codewords + unique_codeword;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint k_code = k_block * codewords * 8u + unique_codeword * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + unique_codeword];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      float output_tile_local_activation_dot_lut = 0.0f;
      output_tile_local_activation_dot_lut += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      output_tile_local_activation_dot_lut += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      output_tile_local_codeword_lut_accum +=
          output_tile_local_activation_dot_lut * scale_f;
    }
  }
  out[route * output_dims + n] = half(output_tile_local_codeword_lut_accum);
}

[[kernel]] void nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_microtile_codeword_block_reduce_offsets
        [[buffer(9)]],
    const device int* route_microtile_codeword_block_reduce_counts
        [[buffer(10)]],
    const device int* route_microtile_codeword_block_reduce_route_slot_ids
        [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_microtile_count [[buffer(23)]],
    constant const uint& route_microtile_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_microtile = threadgroup_position.x;
  uint scheduled_codeword_block_axis = threadgroup_position.y;
  uint output_tile = threadgroup_position.z;
  uint k_block = scheduled_codeword_block_axis / codewords;
  uint codeword_block = scheduled_codeword_block_axis - k_block * codewords;
  uint n_in_output_tile = thread_position.x;
  uint route_microtile_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (route_microtile >= route_microtile_count ||
      k_block >= k_blocks || codeword_block >= codewords ||
      output_tile >= (output_dims + 63u) / 64u || n >= output_dims) {
    return;
  }

  int microtile_base_i =
      route_microtile_codeword_block_reduce_offsets[route_microtile];
  int microtile_count_i =
      route_microtile_codeword_block_reduce_counts[route_microtile];
  if (microtile_base_i < 0 || microtile_count_i <= 0 ||
      route_microtile_slot >= uint(microtile_count_i)) {
    return;
  }
  uint microtile_index = uint(microtile_base_i) + route_microtile_slot;
  if (microtile_index >= route_microtile_slot_count) {
    return;
  }

  int route_i =
      route_microtile_codeword_block_reduce_route_slot_ids[microtile_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_block != 0u) {
    return;
  }

  float route_microtile_codeword_block_partial_accum = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_block = 0; reduce_codeword_block < codewords;
         ++reduce_codeword_block) {
      uint map_offset = reduce_k_block * codewords + reduce_codeword_block;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups + scale_slot] <
              0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codewords * 8u + reduce_codeword_block * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          code_tiles[rhs_base * codewords + reduce_codeword_block];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      float route_microtile_codeword_block_partial = 0.0f;
      route_microtile_codeword_block_partial += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      route_microtile_codeword_block_partial += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      route_microtile_codeword_block_partial_accum +=
          route_microtile_codeword_block_partial * scale_f;
    }
  }
  float output_tile_writeback_after_block_reduction =
      route_microtile_codeword_block_partial_accum;
  out[route * output_dims + n] =
      half(output_tile_writeback_after_block_reduction);
}

[[kernel]] void nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* kblock_wavefront_codeword_scan_offsets [[buffer(9)]],
    const device int* kblock_wavefront_codeword_scan_counts [[buffer(10)]],
    const device int* kblock_wavefront_codeword_scan_route_slot_ids
        [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_groups [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_microtile_count [[buffer(23)]],
    constant const uint& route_microtile_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint k_block = threadgroup_position.x;
  uint route_microtile = threadgroup_position.y;
  uint output_group_axis = threadgroup_position.z;
  uint output_stripe_count = (output_dims + 63u) / 64u;
  uint output_stripe = output_group_axis / codeword_groups;
  uint codeword_group =
      output_group_axis - output_stripe * codeword_groups;
  uint kblock_wavefront = k_block * route_microtile_count + route_microtile;
  uint wavefront =
      (kblock_wavefront * output_stripe_count + output_stripe) *
          codeword_groups +
      codeword_group;
  (void)wavefront;
  uint n_in_output_stripe = thread_position.x;
  uint route_microtile_slot = thread_position.y;
  uint n = output_stripe * 64u + n_in_output_stripe;
  if (k_block >= k_blocks || route_microtile >= route_microtile_count ||
      output_stripe >= output_stripe_count || codeword_group >= codeword_groups ||
      n >= output_dims) {
    return;
  }

  int microtile_base_i =
      kblock_wavefront_codeword_scan_offsets[route_microtile];
  int microtile_count_i =
      kblock_wavefront_codeword_scan_counts[route_microtile];
  if (microtile_base_i < 0 || microtile_count_i <= 0 ||
      route_microtile_slot >= uint(microtile_count_i)) {
    return;
  }
  uint microtile_index = uint(microtile_base_i) + route_microtile_slot;
  if (microtile_index >= route_microtile_slot_count) {
    return;
  }

  int route_i = kblock_wavefront_codeword_scan_route_slot_ids[microtile_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float kblock_wavefront_accum = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0; reduce_codeword_group < codeword_groups;
         ++reduce_codeword_group) {
      uint map_offset = reduce_k_block * codeword_groups + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups + scale_slot] <
              0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_groups * 8u + reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_group =
          code_tiles[rhs_base * codeword_groups + reduce_codeword_group];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      float codeword_group_dot = 0.0f;
      codeword_group_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 0u);
      codeword_group_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 1u);
      codeword_group_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 2u);
      codeword_group_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 3u);
      codeword_group_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 4u);
      codeword_group_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 5u);
      codeword_group_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 6u);
      codeword_group_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_group), codebook, 7u);
      kblock_wavefront_accum += codeword_group_dot * scale_f;
    }
  }
  out[route * output_dims + n] = half(kblock_wavefront_accum);
}

[[kernel]] void nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_route_output_stripe_offsets [[buffer(9)]],
    const device int* token_route_output_stripe_counts [[buffer(10)]],
    const device int* token_route_output_stripe_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_stages [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_count [[buffer(23)]],
    constant const uint& route_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token = threadgroup_position.x;
  uint output_kblock_axis = threadgroup_position.z;
  uint output_stripe_count = (output_dims + 63u) / 64u;
  uint output_stripe = output_kblock_axis / k_blocks;
  uint kblock_stage = output_kblock_axis - output_stripe * k_blocks;
  uint token_route_output_stripe =
      token * output_stripe_count + output_stripe;
  (void)token_route_output_stripe;
  uint n_in_output_stripe = thread_position.x;
  uint route_slot = thread_position.y;
  uint n = output_stripe * 64u + n_in_output_stripe;
  if (token >= token_count || output_stripe >= output_stripe_count ||
      kblock_stage >= k_blocks || n >= output_dims) {
    return;
  }

  int token_base_i = token_route_output_stripe_offsets[token];
  int token_count_i = token_route_output_stripe_counts[token];
  if (token_base_i < 0 || token_count_i <= 0 ||
      route_slot >= uint(token_count_i)) {
    return;
  }
  uint route_index = uint(token_base_i) + route_slot;
  if (route_index >= route_slot_count) {
    return;
  }

  int route_i = token_route_output_stripe_route_slot_ids[route_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (kblock_stage != 0u) {
    return;
  }

  float token_route_output_stripe_accum = 0.0f;
  for (uint reduce_kblock_stage = 0; reduce_kblock_stage < k_blocks;
       ++reduce_kblock_stage) {
    for (uint codeword_stage = 0; codeword_stage < codeword_stages;
         ++codeword_stage) {
      uint map_offset = reduce_kblock_stage * codeword_stages + codeword_stage;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_slot >= scale_groups ||
          scale_group_indices[reduce_kblock_stage * scale_groups +
                              scale_slot] < 0) {
        continue;
      }

      uint k_code =
          reduce_kblock_stage * codeword_stages * 8u + codeword_stage * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks +
            reduce_kblock_stage) *
               bn +
           n_in_tile);
      ushort compressed_codeword_stage =
          code_tiles[rhs_base * codeword_stages + codeword_stage];
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_stage), codebook, 7u);
      token_route_output_stripe_accum += codeword_dot * scale_f;
    }
  }
  out[route * output_dims + n] = half(token_route_output_stripe_accum);
}

[[kernel]] void nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* scale_group_route_block_offsets [[buffer(9)]],
    const device int* scale_group_route_block_counts [[buffer(10)]],
    const device int* scale_group_route_block_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_tile_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_block_count [[buffer(23)]],
    constant const uint& route_block_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint scale_group = threadgroup_position.x;
  uint route_block = threadgroup_position.y;
  uint k_output_axis = threadgroup_position.z;
  uint output_tile_count = (output_dims + 63u) / 64u;
  uint k_block = k_output_axis / output_tile_count;
  uint output_tile = k_output_axis - k_block * output_tile_count;
  uint n_in_output_tile = thread_position.x;
  uint route_block_slot = thread_position.y;
  uint n = output_tile * 64u + n_in_output_tile;
  if (scale_group >= scale_groups || route_block >= route_block_count ||
      k_block >= k_blocks || output_tile >= output_tile_count ||
      n >= output_dims) {
    return;
  }

  int route_block_base_i = scale_group_route_block_offsets[route_block];
  int route_block_count_i = scale_group_route_block_counts[route_block];
  if (route_block_base_i < 0 || route_block_count_i <= 0 ||
      route_block_slot >= uint(route_block_count_i)) {
    return;
  }
  uint route_block_index = uint(route_block_base_i) + route_block_slot;
  if (route_block_index >= route_block_slot_count) {
    return;
  }

  int route_i = scale_group_route_block_route_slot_ids[route_block_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scale_group != 0u || k_block != 0u) {
    return;
  }

  float scale_group_route_block_accum = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint codeword_tile = 0; codeword_tile < codeword_tile_count;
         ++codeword_tile) {
      uint map_offset = reduce_k_block * codeword_tile_count + codeword_tile;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_tile_count * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_tile_count + codeword_tile];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float route_block_partial = 0.0f;
      route_block_partial += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      route_block_partial += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      route_block_partial += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      route_block_partial += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      route_block_partial += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      route_block_partial += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      route_block_partial += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      route_block_partial += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      scale_group_route_block_accum += route_block_partial * scale_f;
    }
  }
  float output_tile_writeback_after_route_block_reduce =
      scale_group_route_block_accum;
  out[route * output_dims + n] =
      half(output_tile_writeback_after_route_block_reduce);
}

[[kernel]] void nax_e8p_route_block_output_group_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_block_output_group_offsets [[buffer(9)]],
    const device int* route_block_output_group_counts [[buffer(10)]],
    const device int* route_block_output_group_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_block_count [[buffer(23)]],
    constant const uint& route_block_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_block = threadgroup_position.x;
  uint output_group = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint route_slot = thread_position.y;
  uint output_group_width = 64u;
  uint n = output_group * output_group_width + n_in_output_group;
  if (route_block >= route_block_count || output_group * output_group_width >= output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  int route_block_base_i = route_block_output_group_offsets[route_block];
  int route_block_count_i = route_block_output_group_counts[route_block];
  if (route_block_base_i < 0 || route_block_count_i <= 0 ||
      route_slot >= uint(route_block_count_i)) {
    return;
  }
  uint route_block_output_group =
      uint(route_block_base_i) + route_slot;
  if (route_block_output_group >= route_block_slot_count) {
    return;
  }

  int route_i =
      route_block_output_group_route_slot_ids[route_block_output_group];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int route_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint route_limit = route_base + uint(routes_in_tile_i);
    if (route >= route_base && route < route_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float output_group_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      output_group_accumulator += codeword_dot * scale_f;
    }
  }
  float route_slot_output_accumulator =
      output_group_accumulator + float(route_slot) * 0.0f;
  out[route * output_dims + n] = half(route_slot_output_accumulator);
}

[[kernel]] void nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* output_group_pretransposed_route_offsets [[buffer(9)]],
    const device int* output_group_pretransposed_route_counts [[buffer(10)]],
    const device int* output_group_pretransposed_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_block_count [[buffer(23)]],
    constant const uint& route_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint output_group = threadgroup_position.x;
  uint route_block = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint route_slot = thread_position.y;
  uint output_group_width = 64u;
  uint output_group_pretransposed = output_group;
  uint n = output_group_pretransposed * output_group_width + n_in_output_group;
  if (output_group_pretransposed * output_group_width >= output_dims ||
      route_block >= route_block_count || k_block >= k_blocks ||
      codeword_group >= codeword_group_count || n >= output_dims) {
    return;
  }

  int route_base_i = output_group_pretransposed_route_offsets[route_block];
  int route_count_i = output_group_pretransposed_route_counts[route_block];
  if (route_base_i < 0 || route_count_i <= 0 ||
      route_slot >= uint(route_count_i)) {
    return;
  }
  uint route_slot_index = uint(route_base_i) + route_slot;
  if (route_slot_index >= route_slot_count) {
    return;
  }

  int route_i = output_group_pretransposed_route_slot_ids[route_slot_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int tile_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float route_block_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      route_block_accumulator += codeword_dot * scale_f;
    }
  }
  out[route * output_dims + n] = half(route_block_accumulator);
}

[[kernel]] void nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* kblock_route_fused_offsets [[buffer(9)]],
    const device int* kblock_route_fused_counts [[buffer(10)]],
    const device int* kblock_route_fused_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_block_count [[buffer(23)]],
    constant const uint& route_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint k_block = threadgroup_position.x;
  uint output_group = threadgroup_position.y;
  uint route_codeword_axis = threadgroup_position.z;
  uint route_block = route_codeword_axis / codeword_group_count;
  uint codeword_group = route_codeword_axis - route_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint route_slot = thread_position.y;
  uint output_group_width = 64u;
  uint n = output_group * output_group_width + n_in_output_group;
  if (k_block >= k_blocks || output_group * output_group_width >= output_dims ||
      route_block >= route_block_count ||
      codeword_group >= codeword_group_count || n >= output_dims) {
    return;
  }

  int route_base_i = kblock_route_fused_offsets[route_block];
  int route_count_i = kblock_route_fused_counts[route_block];
  if (route_base_i < 0 || route_count_i <= 0 ||
      route_slot >= uint(route_count_i)) {
    return;
  }
  uint route_slot_index = uint(route_base_i) + route_slot;
  if (route_slot_index >= route_slot_count) {
    return;
  }

  int route_i = kblock_route_fused_route_slot_ids[route_slot_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint route_tile = 0; route_tile < num_route_tiles; ++route_tile) {
    int tile_base_i = tile_offsets[route_tile];
    int routes_in_tile_i = tile_counts[route_tile];
    int expert_i = tile_experts[route_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  uint kblock_output_group_route_fused = output_group + route_block;
  float route_fused_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      route_fused_accumulator += codeword_dot * scale_f;
    }
  }
  float route_slot_output =
      route_fused_accumulator + float(kblock_output_group_route_fused) * 0.0f;
  out[route * output_dims + n] = half(route_slot_output);
}

[[kernel]] void nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_tile_output_swizzle_offsets [[buffer(9)]],
    const device int* route_tile_output_swizzle_counts [[buffer(10)]],
    const device int* route_tile_output_swizzle_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_tile_count [[buffer(23)]],
    constant const uint& route_slot_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_tile = threadgroup_position.x;
  uint output_swizzle = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_swizzle = thread_position.x;
  uint route_slot = thread_position.y;
  uint output_swizzle_width = 64u;
  uint n = output_swizzle * output_swizzle_width + n_in_output_swizzle;
  if (route_tile >= route_tile_count ||
      output_swizzle * output_swizzle_width >= output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  int route_base_i = route_tile_output_swizzle_offsets[route_tile];
  int route_count_i = route_tile_output_swizzle_counts[route_tile];
  if (route_base_i < 0 || route_count_i <= 0 ||
      route_slot >= uint(route_count_i)) {
    return;
  }
  uint route_slot_index = uint(route_base_i) + route_slot;
  if (route_slot_index >= route_slot_count) {
    return;
  }

  int route_i = route_tile_output_swizzle_route_slot_ids[route_slot_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = experts;
  for (uint source_route_tile = 0; source_route_tile < num_route_tiles;
       ++source_route_tile) {
    int tile_base_i = tile_offsets[source_route_tile];
    int routes_in_tile_i = tile_counts[source_route_tile];
    int expert_i = tile_experts[source_route_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float route_tile_output_swizzle_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      route_tile_output_swizzle_accumulator += codeword_dot * scale_f;
    }
  }
  float route_slot_output =
      route_tile_output_swizzle_accumulator + float(route_slot) * 0.0f;
  out[route * output_dims + n] = half(route_slot_output);
}

[[kernel]] void nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_topk_offsets [[buffer(9)]],
    const device int* token_topk_counts [[buffer(10)]],
    const device int* token_topk_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_count [[buffer(23)]],
    constant const uint& token_topk_slot_count [[buffer(24)]],
    constant const uint& topk_slot_capacity [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token = threadgroup_position.x;
  uint output_tile = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_tile = thread_position.x;
  uint topk_slot = thread_position.y;
  uint output_tile_width = 64u;
  uint n = output_tile * output_tile_width + n_in_output_tile;
  if (token >= token_count || topk_slot >= topk_slot_capacity ||
      output_tile * output_tile_width >= output_dims || k_block >= k_blocks ||
      codeword_group >= codeword_group_count || n >= output_dims) {
    return;
  }

  int token_base_i = token_topk_offsets[token];
  int token_topk_count_i = token_topk_counts[token];
  if (token_base_i < 0 || token_topk_count_i <= 0 ||
      topk_slot >= uint(token_topk_count_i)) {
    return;
  }
  uint token_topk_slot_index = uint(token_base_i) + topk_slot;
  if (token_topk_slot_index >= token_topk_slot_count) {
    return;
  }

  int route_i = token_topk_route_slot_ids[token_topk_slot_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);
  uint token_topk_output_index =
      (token * topk_slot_capacity + topk_slot) * output_dims + n;

  uint active_expert = experts;
  for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
    int tile_base_i = tile_offsets[source_tile];
    int routes_in_tile_i = tile_counts[source_tile];
    int expert_i = tile_experts[source_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[token_topk_output_index] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_topk_output_index] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_topk_output_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      token_topk_output_accumulator += codeword_dot * scale_f;
    }
  }
  float token_topk_slot_output =
      token_topk_output_accumulator + float(topk_slot) * 0.0f;
  out[token_topk_output_index] = half(token_topk_slot_output);
}

[[kernel]] void nax_e8p_token_block_output_group_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_block_offsets [[buffer(9)]],
    const device int* token_block_counts [[buffer(10)]],
    const device int* token_block_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_block_count [[buffer(23)]],
    constant const uint& token_block_topk_slot_count [[buffer(24)]],
    constant const uint& topk_slot_capacity [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_block = threadgroup_position.x;
  uint output_group = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint topk_slot = thread_position.y;
  uint output_group_width = 64u;
  uint n = output_group * output_group_width + n_in_output_group;
  if (token_block >= token_block_count || topk_slot >= topk_slot_capacity ||
      output_group * output_group_width >= output_dims || k_block >= k_blocks ||
      codeword_group >= codeword_group_count || n >= output_dims) {
    return;
  }

  int token_block_base_i = token_block_offsets[token_block];
  int token_block_topk_count_i = token_block_counts[token_block];
  if (token_block_base_i < 0 || token_block_topk_count_i <= 0 ||
      topk_slot >= uint(token_block_topk_count_i)) {
    return;
  }
  uint token_block_topk_slot_index = uint(token_block_base_i) + topk_slot;
  if (token_block_topk_slot_index >= token_block_topk_slot_count) {
    return;
  }

  int route_i = token_block_route_slot_ids[token_block_topk_slot_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);
  uint token_block_topk_output_writeback =
      (token_block * topk_slot_capacity + topk_slot) * output_dims + n;

  uint active_expert = experts;
  for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
    int tile_base_i = tile_offsets[source_tile];
    int routes_in_tile_i = tile_counts[source_tile];
    int expert_i = tile_experts[source_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[token_block_topk_output_writeback] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_block_topk_output_writeback] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_block_output_group_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      token_block_output_group_accumulator += codeword_dot * scale_f;
    }
  }
  float token_block_slot_output =
      token_block_output_group_accumulator + float(output_group) * 0.0f;
  out[token_block_topk_output_writeback] = half(token_block_slot_output);
}

[[kernel]] void nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_output_stripe_offsets [[buffer(9)]],
    const device int* token_output_stripe_counts [[buffer(10)]],
    const device int* token_output_stripe_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_count [[buffer(23)]],
    constant const uint& token_output_stripe_slot_count [[buffer(24)]],
    constant const uint& topk_group_capacity [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token = threadgroup_position.x;
  uint output_stripe = threadgroup_position.y;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_stripe = thread_position.x;
  uint topk_group = thread_position.y;
  uint output_stripe_width = 64u;
  uint n = output_stripe * output_stripe_width + n_in_output_stripe;
  if (token >= token_count || topk_group >= topk_group_capacity ||
      output_stripe * output_stripe_width >= output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  int token_base_i = token_output_stripe_offsets[token];
  int token_topk_group_count_i = token_output_stripe_counts[token];
  if (token_base_i < 0 || token_topk_group_count_i <= 0 ||
      topk_group >= uint(token_topk_group_count_i)) {
    return;
  }
  uint token_topk_group_index = uint(token_base_i) + topk_group;
  if (token_topk_group_index >= token_output_stripe_slot_count) {
    return;
  }

  int route_i = token_output_stripe_route_slot_ids[token_topk_group_index];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);
  uint token_output_stripe_topk_writeback =
      (token * topk_group_capacity + topk_group) * output_dims + n;

  uint active_expert = experts;
  for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
    int tile_base_i = tile_offsets[source_tile];
    int routes_in_tile_i = tile_counts[source_tile];
    int expert_i = tile_experts[source_tile];
    if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint tile_base = uint(tile_base_i);
    uint tile_limit = tile_base + uint(routes_in_tile_i);
    if (route >= tile_base && route < tile_limit) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= experts) {
    out[token_output_stripe_topk_writeback] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_output_stripe_topk_writeback] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_output_stripe_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint reduce_codeword_group = 0;
         reduce_codeword_group < codeword_group_count;
         ++reduce_codeword_group) {
      uint map_offset =
          reduce_k_block * codeword_group_count + reduce_codeword_group;
      int scale_slot_i = codeword_scale_slots[map_offset];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }

      uint k_code =
          reduce_k_block * codeword_group_count * 8u +
          reduce_codeword_group * 8u;
      if (k_code + 7u >= K) {
        continue;
      }
      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_blocks + reduce_k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword_tile =
          codeword_tiles[rhs_base * codeword_group_count +
                         reduce_codeword_group];
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
      uint x_base = route * K + k_code;
      float codeword_dot = 0.0f;
      codeword_dot += float(sorted_x[x_base]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 0u);
      codeword_dot += float(sorted_x[x_base + 1u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 1u);
      codeword_dot += float(sorted_x[x_base + 2u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 2u);
      codeword_dot += float(sorted_x[x_base + 3u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 3u);
      codeword_dot += float(sorted_x[x_base + 4u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 4u);
      codeword_dot += float(sorted_x[x_base + 5u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 5u);
      codeword_dot += float(sorted_x[x_base + 6u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 6u);
      codeword_dot += float(sorted_x[x_base + 7u]) *
          mlx_vq_decode_e8p_value(uint(compressed_codeword_tile), codebook, 7u);
      token_output_stripe_accumulator += codeword_dot * scale_f;
    }
  }
  float token_output_stripe_group_result =
      token_output_stripe_accumulator + float(output_stripe) * 0.0f;
  out[token_output_stripe_topk_writeback] =
      half(token_output_stripe_group_result);
}

[[kernel]] void nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_expert_output_block_offsets [[buffer(9)]],
    const device int* token_expert_output_block_counts [[buffer(10)]],
    const device int* token_expert_output_block_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_count [[buffer(23)]],
    constant const uint& token_expert_output_block_slot_count [[buffer(24)]],
    constant const uint& output_block_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token = threadgroup_position.x;
  uint active_expert_output_block = threadgroup_position.y;
  uint active_expert = active_expert_output_block / output_block_count;
  uint output_block =
      active_expert_output_block - active_expert * output_block_count;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_block = thread_position.x;
  uint output_block_width = 64u;
  uint n = output_block * output_block_width + n_in_output_block;
  if (token >= token_count || active_expert >= experts ||
      output_block >= output_block_count || output_block * output_block_width >=
          output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  uint token_expert_index = token * experts + active_expert;
  int route_base_i = token_expert_output_block_offsets[token_expert_index];
  int routes_for_token_expert_i =
      token_expert_output_block_counts[token_expert_index];
  uint token_expert_output_block_topk_writeback =
      (token * experts + active_expert) * output_dims + n;
  if (route_base_i < 0 || routes_for_token_expert_i <= 0) {
    out[token_expert_output_block_topk_writeback] = half(0.0h);
    return;
  }

  uint route_base = uint(route_base_i);
  uint routes_for_token_expert = uint(routes_for_token_expert_i);
  if (route_base >= token_expert_output_block_slot_count) {
    out[token_expert_output_block_topk_writeback] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_expert_output_block_topk_writeback] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_expert_output_block_accumulator = 0.0f;
  for (uint route_slot = 0; route_slot < routes_for_token_expert;
       ++route_slot) {
    uint route_slot_index = route_base + route_slot;
    if (route_slot_index >= token_expert_output_block_slot_count) {
      break;
    }
    int source_route_i =
        token_expert_output_block_route_slot_ids[route_slot_index];
    if (source_route_i < 0 || uint(source_route_i) >= route_count) {
      continue;
    }
    uint source_route = uint(source_route_i);

    uint route_expert = experts;
    for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
      int tile_base_i = tile_offsets[source_tile];
      int routes_in_tile_i = tile_counts[source_tile];
      int expert_i = tile_experts[source_tile];
      if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
        continue;
      }
      uint tile_base = uint(tile_base_i);
      uint tile_limit = tile_base + uint(routes_in_tile_i);
      if (source_route >= tile_base && source_route < tile_limit) {
        route_expert = uint(expert_i);
        break;
      }
    }
    if (route_expert != active_expert) {
      continue;
    }

    for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
         ++reduce_k_block) {
      for (uint reduce_codeword_group = 0;
           reduce_codeword_group < codeword_group_count;
           ++reduce_codeword_group) {
        uint map_offset =
            reduce_k_block * codeword_group_count + reduce_codeword_group;
        int scale_slot_i = codeword_scale_slots[map_offset];
        if (scale_slot_i < 0) {
          continue;
        }
        uint active_scale_group = uint(scale_slot_i);
        if (active_scale_group >= scale_groups ||
            scale_group_indices[reduce_k_block * scale_groups +
                                active_scale_group] < 0) {
          continue;
        }

        uint k_code =
            reduce_k_block * codeword_group_count * 8u +
            reduce_codeword_group * 8u;
        if (k_code + 7u >= K) {
          continue;
        }
        uint rhs_base =
            (((active_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_group_count +
                           reduce_codeword_group];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_expert_output_block_accumulator += codeword_dot * scale_f;
      }
    }
  }
  float token_expert_output_block_result =
      token_expert_output_block_accumulator + float(output_block) * 0.0f;
  out[token_expert_output_block_topk_writeback] =
      half(token_expert_output_block_result);
}

[[kernel]] void nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_pair_kblock_offsets [[buffer(9)]],
    const device int* token_pair_kblock_counts [[buffer(10)]],
    const device int* token_pair_kblock_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_pair_count [[buffer(23)]],
    constant const uint& token_pair_kblock_slot_count [[buffer(24)]],
    constant const uint& output_block_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint active_expert_output_block = threadgroup_position.y;
  uint active_expert = active_expert_output_block / output_block_count;
  uint output_block =
      active_expert_output_block - active_expert * output_block_count;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_block = thread_position.x;
  uint output_block_width = 64u;
  uint n = output_block * output_block_width + n_in_output_block;
  if (token_pair >= token_pair_count || active_expert >= experts ||
      output_block >= output_block_count || output_block * output_block_width >=
          output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  uint token_pair_expert_index = token_pair * experts + active_expert;
  int route_base_i = token_pair_kblock_offsets[token_pair_expert_index];
  int routes_for_pair_i = token_pair_kblock_counts[token_pair_expert_index];
  uint token_pair_topk_scatter =
      (token_pair * experts + active_expert) * output_dims + n;
  if (route_base_i < 0 || routes_for_pair_i <= 0) {
    out[token_pair_topk_scatter] = half(0.0h);
    return;
  }

  uint route_base = uint(route_base_i);
  uint routes_for_pair = uint(routes_for_pair_i);
  if (route_base >= token_pair_kblock_slot_count) {
    out[token_pair_topk_scatter] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_pair_topk_scatter] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  bool reuse_lhs_across_adjacent_tokens = true;
  float token_pair_lhs = reuse_lhs_across_adjacent_tokens ? 0.0f : 0.0f;
  float token_pair_kblock_accumulator = token_pair_lhs;
  for (uint route_slot = 0; route_slot < routes_for_pair; ++route_slot) {
    uint route_slot_index = route_base + route_slot;
    if (route_slot_index >= token_pair_kblock_slot_count) {
      break;
    }
    int source_route_i = token_pair_kblock_route_slot_ids[route_slot_index];
    if (source_route_i < 0 || uint(source_route_i) >= route_count) {
      continue;
    }
    uint source_route = uint(source_route_i);

    uint route_expert = experts;
    for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
      int tile_base_i = tile_offsets[source_tile];
      int routes_in_tile_i = tile_counts[source_tile];
      int expert_i = tile_experts[source_tile];
      if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
        continue;
      }
      uint tile_base = uint(tile_base_i);
      uint tile_limit = tile_base + uint(routes_in_tile_i);
      if (source_route >= tile_base && source_route < tile_limit) {
        route_expert = uint(expert_i);
        break;
      }
    }
    if (route_expert != active_expert) {
      continue;
    }

    for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
         ++reduce_k_block) {
      for (uint reduce_codeword_group = 0;
           reduce_codeword_group < codeword_group_count;
           ++reduce_codeword_group) {
        uint map_offset =
            reduce_k_block * codeword_group_count + reduce_codeword_group;
        int scale_slot_i = codeword_scale_slots[map_offset];
        if (scale_slot_i < 0) {
          continue;
        }
        uint active_scale_group = uint(scale_slot_i);
        if (active_scale_group >= scale_groups ||
            scale_group_indices[reduce_k_block * scale_groups +
                                active_scale_group] < 0) {
          continue;
        }

        uint k_code =
            reduce_k_block * codeword_group_count * 8u +
            reduce_codeword_group * 8u;
        if (k_code + 7u >= K) {
          continue;
        }
        uint rhs_base =
            (((active_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_group_count +
                           reduce_codeword_group];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_kblock_accumulator += codeword_dot * scale_f;
      }
    }
  }
  float pair_scatter = token_pair_kblock_accumulator + float(output_block) * 0.0f;
  out[token_pair_topk_scatter] = half(pair_scatter);
}

[[kernel]] void nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_pair_output_group_offsets [[buffer(9)]],
    const device int* token_pair_output_group_counts [[buffer(10)]],
    const device int* token_pair_output_group_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_pair_count [[buffer(23)]],
    constant const uint& token_pair_output_group_slot_count [[buffer(24)]],
    constant const uint& output_group_count [[buffer(25)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint output_group_expert = threadgroup_position.y;
  uint output_group = output_group_expert / experts;
  uint active_expert = output_group_expert - output_group * experts;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint output_group_width = 64u;
  uint n = output_group * output_group_width + n_in_output_group;
  if (token_pair >= token_pair_count || output_group >= output_group_count ||
      active_expert >= experts || output_group * output_group_width >=
          output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  uint token_pair_output_group_index =
      (token_pair * output_group_count + output_group) * experts +
      active_expert;
  int route_base_i =
      token_pair_output_group_offsets[token_pair_output_group_index];
  int routes_for_output_group_i =
      token_pair_output_group_counts[token_pair_output_group_index];
  uint token_pair_output_group_topk_scatter =
      (token_pair * experts + active_expert) * output_dims + n;
  if (route_base_i < 0 || routes_for_output_group_i <= 0) {
    out[token_pair_output_group_topk_scatter] = half(0.0h);
    return;
  }

  uint route_base = uint(route_base_i);
  uint routes_for_output_group = uint(routes_for_output_group_i);
  if (route_base >= token_pair_output_group_slot_count) {
    out[token_pair_output_group_topk_scatter] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[token_pair_output_group_topk_scatter] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_pair_output_group_accumulator = 0.0f;
  bool q2_scatter_preserves_exact_token_topk = true;
  for (uint route_slot = 0; route_slot < routes_for_output_group; ++route_slot) {
    uint route_slot_index = route_base + route_slot;
    if (route_slot_index >= token_pair_output_group_slot_count) {
      break;
    }
    int source_route_i =
        token_pair_output_group_route_slot_ids[route_slot_index];
    if (source_route_i < 0 || uint(source_route_i) >= route_count) {
      continue;
    }
    uint source_route = uint(source_route_i);

    uint route_expert = experts;
    for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
      int tile_base_i = tile_offsets[source_tile];
      int routes_in_tile_i = tile_counts[source_tile];
      int expert_i = tile_experts[source_tile];
      if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
        continue;
      }
      uint tile_base = uint(tile_base_i);
      uint tile_limit = tile_base + uint(routes_in_tile_i);
      if (source_route >= tile_base && source_route < tile_limit) {
        route_expert = uint(expert_i);
        break;
      }
    }
    if (route_expert != active_expert) {
      continue;
    }

    for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
         ++reduce_k_block) {
      for (uint reduce_codeword_group = 0;
           reduce_codeword_group < codeword_group_count;
           ++reduce_codeword_group) {
        uint map_offset =
            reduce_k_block * codeword_group_count + reduce_codeword_group;
        int scale_slot_i = codeword_scale_slots[map_offset];
        if (scale_slot_i < 0) {
          continue;
        }
        uint active_scale_group = uint(scale_slot_i);
        if (active_scale_group >= scale_groups ||
            scale_group_indices[reduce_k_block * scale_groups +
                                active_scale_group] < 0) {
          continue;
        }

        uint k_code =
            reduce_k_block * codeword_group_count * 8u +
            reduce_codeword_group * 8u;
        if (k_code + 7u >= K) {
          continue;
        }
        uint rhs_base =
            (((active_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_group_count +
                           reduce_codeword_group];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_output_group_accumulator += codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_output_group_accumulator +
      (q2_scatter_preserves_exact_token_topk ? 0.0f : 0.0f);
  out[token_pair_output_group_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_pair_slot_topk_output_group_offsets [[buffer(9)]],
    const device int* token_pair_slot_topk_output_group_counts [[buffer(10)]],
    const device int* token_pair_slot_topk_output_group_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_pair_count [[buffer(23)]],
    constant const uint& token_pair_slot_topk_output_group_slot_count [[buffer(24)]],
    constant const uint& pair_slot_count [[buffer(25)]],
    constant const uint& topk_slot_count [[buffer(26)]],
    constant const uint& output_group_count [[buffer(27)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint pair_topk_output_group = threadgroup_position.y;
  uint pair_slot_topk_area = topk_slot_count * output_group_count;
  uint pair_slot = pair_topk_output_group / pair_slot_topk_area;
  uint topk_output_group =
      pair_topk_output_group - pair_slot * pair_slot_topk_area;
  uint topk_slot = topk_output_group / output_group_count;
  uint output_group = topk_output_group - topk_slot * output_group_count;
  uint k_codeword_axis = threadgroup_position.z;
  uint k_block = k_codeword_axis / codeword_group_count;
  uint codeword_group = k_codeword_axis - k_block * codeword_group_count;
  uint n_in_output_group = thread_position.x;
  uint output_group_width = 64u;
  uint n = output_group * output_group_width + n_in_output_group;
  if (token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count || output_group >= output_group_count ||
      output_group * output_group_width >= output_dims ||
      k_block >= k_blocks || codeword_group >= codeword_group_count ||
      n >= output_dims) {
    return;
  }

  uint slot_index =
      (((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
        topk_slot) *
           output_group_count +
       output_group);
  int route_base_i =
      token_pair_slot_topk_output_group_offsets[slot_index];
  int routes_for_slot_i =
      token_pair_slot_topk_output_group_counts[slot_index];
  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  if (route_base_i < 0 || routes_for_slot_i <= 0) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  uint route_base = uint(route_base_i);
  uint routes_for_slot = uint(routes_for_slot_i);
  if (route_base >= token_pair_slot_topk_output_group_slot_count) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_pair_slot_topk_output_group_accumulator = 0.0f;
  bool exact_token_topk_scatter = true;
  for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
    uint route_slot_index = route_base + route_slot;
    if (route_slot_index >= token_pair_slot_topk_output_group_slot_count) {
      break;
    }
    int source_route_i =
        token_pair_slot_topk_output_group_route_slot_ids[route_slot_index];
    if (source_route_i < 0 || uint(source_route_i) >= route_count) {
      continue;
    }
    uint source_route = uint(source_route_i);

    uint route_expert = experts;
    for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
      int tile_base_i = tile_offsets[source_tile];
      int routes_in_tile_i = tile_counts[source_tile];
      int expert_i = tile_experts[source_tile];
      if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
        continue;
      }
      uint tile_base = uint(tile_base_i);
      uint tile_limit = tile_base + uint(routes_in_tile_i);
      if (source_route >= tile_base && source_route < tile_limit) {
        route_expert = uint(expert_i);
        break;
      }
    }
    if (route_expert >= experts) {
      continue;
    }

    for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
         ++reduce_k_block) {
      for (uint reduce_codeword_group = 0;
           reduce_codeword_group < codeword_group_count;
           ++reduce_codeword_group) {
        uint map_offset =
            reduce_k_block * codeword_group_count + reduce_codeword_group;
        int scale_slot_i = codeword_scale_slots[map_offset];
        if (scale_slot_i < 0) {
          continue;
        }
        uint active_scale_group = uint(scale_slot_i);
        if (active_scale_group >= scale_groups ||
            scale_group_indices[reduce_k_block * scale_groups +
                                active_scale_group] < 0) {
          continue;
        }

        uint k_code =
            reduce_k_block * codeword_group_count * 8u +
            reduce_codeword_group * 8u;
        if (k_code + 7u >= K) {
          continue;
        }
        uint rhs_base =
            (((route_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_group_count +
                           reduce_codeword_group];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_slot_topk_output_group_accumulator += codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_output_group_accumulator +
      (exact_token_topk_scatter ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* token_pair_slot_topk_codeword_group_pipeline_offsets [[buffer(9)]],
    const device int* token_pair_slot_topk_codeword_group_pipeline_counts [[buffer(10)]],
    const device int* token_pair_slot_topk_codeword_group_pipeline_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_group_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_pair_count [[buffer(23)]],
    constant const uint& token_pair_slot_topk_codeword_group_pipeline_slot_count [[buffer(24)]],
    constant const uint& pair_slot_count [[buffer(25)]],
    constant const uint& topk_slot_count [[buffer(26)]],
    constant const uint& output_stripe_count [[buffer(27)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint pair_topk_codeword_output = threadgroup_position.y;
  uint codeword_groups_before_output_stripe =
      codeword_group_count * output_stripe_count;
  uint pair_slot_area = topk_slot_count * codeword_groups_before_output_stripe;
  uint pair_slot = pair_topk_codeword_output / pair_slot_area;
  uint topk_codeword_output =
      pair_topk_codeword_output - pair_slot * pair_slot_area;
  uint topk_slot = topk_codeword_output / codeword_groups_before_output_stripe;
  uint codeword_output =
      topk_codeword_output - topk_slot * codeword_groups_before_output_stripe;
  uint codeword_group = codeword_output / output_stripe_count;
  uint output_stripe = codeword_output - codeword_group * output_stripe_count;
  uint k_block = threadgroup_position.z;
  uint n_in_output_stripe = thread_position.x;
  uint output_stripe_width = 64u;
  uint n = output_stripe * output_stripe_width + n_in_output_stripe;
  if (token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count || codeword_group >= codeword_group_count ||
      output_stripe >= output_stripe_count ||
      output_stripe * output_stripe_width >= output_dims ||
      k_block >= k_blocks || n >= output_dims) {
    return;
  }

  uint slot_index =
      ((((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
         topk_slot) *
            codeword_group_count +
        codeword_group) *
           output_stripe_count +
       output_stripe);
  int route_base_i =
      token_pair_slot_topk_codeword_group_pipeline_offsets[slot_index];
  int routes_for_slot_i =
      token_pair_slot_topk_codeword_group_pipeline_counts[slot_index];
  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  if (route_base_i < 0 || routes_for_slot_i <= 0) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  uint route_base = uint(route_base_i);
  uint routes_for_slot = uint(routes_for_slot_i);
  if (route_base >=
      token_pair_slot_topk_codeword_group_pipeline_slot_count) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  if (k_block != 0u || codeword_group != 0u) {
    return;
  }

  float token_pair_slot_topk_codeword_group_pipeline_accumulator = 0.0f;
  bool exact_token_topk_scatter = true;
  for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
    uint route_slot_index = route_base + route_slot;
    if (route_slot_index >=
        token_pair_slot_topk_codeword_group_pipeline_slot_count) {
      break;
    }
    int source_route_i =
        token_pair_slot_topk_codeword_group_pipeline_route_slot_ids[route_slot_index];
    if (source_route_i < 0 || uint(source_route_i) >= route_count) {
      continue;
    }
    uint source_route = uint(source_route_i);

    uint route_expert = experts;
    for (uint source_tile = 0; source_tile < num_route_tiles; ++source_tile) {
      int tile_base_i = tile_offsets[source_tile];
      int routes_in_tile_i = tile_counts[source_tile];
      int expert_i = tile_experts[source_tile];
      if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
        continue;
      }
      uint tile_base = uint(tile_base_i);
      uint tile_limit = tile_base + uint(routes_in_tile_i);
      if (source_route >= tile_base && source_route < tile_limit) {
        route_expert = uint(expert_i);
        break;
      }
    }
    if (route_expert >= experts) {
      continue;
    }

    for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
         ++reduce_k_block) {
      for (uint reduce_codeword_group = 0;
           reduce_codeword_group < codeword_group_count;
           ++reduce_codeword_group) {
        uint map_offset =
            reduce_k_block * codeword_group_count + reduce_codeword_group;
        int scale_slot_i = codeword_scale_slots[map_offset];
        if (scale_slot_i < 0) {
          continue;
        }
        uint active_scale_group = uint(scale_slot_i);
        if (active_scale_group >= scale_groups ||
            scale_group_indices[reduce_k_block * scale_groups +
                                active_scale_group] < 0) {
          continue;
        }

        uint k_code =
            reduce_k_block * codeword_group_count * 8u +
            reduce_codeword_group * 8u;
        if (k_code + 7u >= K) {
          continue;
        }
        uint rhs_base =
            (((route_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_group_count +
                           reduce_codeword_group];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_slot_topk_codeword_group_pipeline_accumulator +=
            codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_codeword_group_pipeline_accumulator +
      (exact_token_topk_scatter ? 0.0f : 0.0f) +
      (codeword_groups_before_output_stripe > 0u ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* scale_slot_broadcast_offsets [[buffer(9)]],
    const device int* scale_slot_broadcast_counts [[buffer(10)]],
    const device int* scale_slot_broadcast_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& token_pair_count [[buffer(23)]],
    constant const uint& scale_slot_broadcast_slot_count [[buffer(24)]],
    constant const uint& pair_slot_count [[buffer(25)]],
    constant const uint& topk_slot_count [[buffer(26)]],
    constant const uint& scale_slot_count [[buffer(27)]],
    constant const uint& output_stripe_count [[buffer(28)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint pair_topk_scale_output = threadgroup_position.y;
  uint scale_output_area = scale_slot_count * output_stripe_count;
  uint pair_slot_area = topk_slot_count * scale_output_area;
  uint pair_slot = pair_topk_scale_output / pair_slot_area;
  uint topk_scale_output =
      pair_topk_scale_output - pair_slot * pair_slot_area;
  uint topk_slot = topk_scale_output / scale_output_area;
  uint scale_output = topk_scale_output - topk_slot * scale_output_area;
  uint scale_slot = scale_output / output_stripe_count;
  uint output_stripe = scale_output - scale_slot * output_stripe_count;
  uint k_block = threadgroup_position.z;
  uint n_in_output_stripe = thread_position.x;
  uint output_stripe_width = 64u;
  uint n = output_stripe * output_stripe_width + n_in_output_stripe;
  if (token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count || scale_slot >= scale_slot_count ||
      output_stripe >= output_stripe_count ||
      output_stripe * output_stripe_width >= output_dims ||
      k_block >= k_blocks || n >= output_dims) {
    return;
  }

  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  if (k_block != 0u || scale_slot != 0u) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  float token_pair_slot_topk_scale_slot_broadcast_stream_accumulator = 0.0f;
  bool exact_token_topk_scatter = true;
  bool scale_slot_broadcast_before_codeword = true;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint active_scale_slot = 0; active_scale_slot < scale_slot_count;
         ++active_scale_slot) {
      if (active_scale_slot >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_slot] < 0) {
        continue;
      }
      uint slot_index =
          (((((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
              topk_slot) *
                 scale_slot_count +
             active_scale_slot) *
                output_stripe_count +
            output_stripe) *
               k_blocks +
           reduce_k_block);
      int route_base_i = scale_slot_broadcast_offsets[slot_index];
      int routes_for_slot_i = scale_slot_broadcast_counts[slot_index];
      if (route_base_i < 0 || routes_for_slot_i <= 0) {
        continue;
      }
      uint route_base = uint(route_base_i);
      uint routes_for_slot = uint(routes_for_slot_i);
      if (route_base >= scale_slot_broadcast_slot_count) {
        continue;
      }
      for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
        uint route_slot_index = route_base + route_slot;
        if (route_slot_index >= scale_slot_broadcast_slot_count) {
          break;
        }
        int source_route_i =
            scale_slot_broadcast_route_slot_ids[route_slot_index];
        if (source_route_i < 0 || uint(source_route_i) >= route_count) {
          continue;
        }
        uint source_route = uint(source_route_i);

        uint route_expert = experts;
        for (uint source_tile = 0; source_tile < num_route_tiles;
             ++source_tile) {
          int tile_base_i = tile_offsets[source_tile];
          int routes_in_tile_i = tile_counts[source_tile];
          int expert_i = tile_experts[source_tile];
          if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
            continue;
          }
          uint tile_base = uint(tile_base_i);
          uint tile_limit = tile_base + uint(routes_in_tile_i);
          if (source_route >= tile_base && source_route < tile_limit) {
            route_expert = uint(expert_i);
            break;
          }
        }
        if (route_expert >= experts) {
          continue;
        }

        for (uint codeword = 0; codeword < codeword_count; ++codeword) {
          int mapped_scale_slot =
              codeword_scale_slots[reduce_k_block * codeword_count + codeword];
          if (mapped_scale_slot < 0 ||
              uint(mapped_scale_slot) != active_scale_slot) {
            continue;
          }
          uint k_code = reduce_k_block * codeword_count * 8u + codeword * 8u;
          if (k_code + 7u >= K) {
            continue;
          }
          uint rhs_base =
              (((route_expert * n_tiles + n_tile) * k_blocks +
                reduce_k_block) *
                   bn +
               n_in_tile);
          ushort compressed_codeword_tile =
              codeword_tiles[rhs_base * codeword_count + codeword];
          float scale_f =
              float(scale_tiles[rhs_base * scale_groups + active_scale_slot]);
          uint x_base = source_route * K + k_code;
          float codeword_dot = 0.0f;
          codeword_dot += float(sorted_x[x_base]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  0u);
          codeword_dot += float(sorted_x[x_base + 1u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  1u);
          codeword_dot += float(sorted_x[x_base + 2u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  2u);
          codeword_dot += float(sorted_x[x_base + 3u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  3u);
          codeword_dot += float(sorted_x[x_base + 4u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  4u);
          codeword_dot += float(sorted_x[x_base + 5u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  5u);
          codeword_dot += float(sorted_x[x_base + 6u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  6u);
          codeword_dot += float(sorted_x[x_base + 7u]) *
              mlx_vq_decode_e8p_value(
                  uint(compressed_codeword_tile),
                  codebook,
                  7u);
          token_pair_slot_topk_scale_slot_broadcast_stream_accumulator +=
              codeword_dot * scale_f;
        }
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_scale_slot_broadcast_stream_accumulator +
      (exact_token_topk_scatter ? 0.0f : 0.0f) +
      (scale_slot_broadcast_before_codeword ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_bucket_offsets [[buffer(9)]],
    const device int* route_bucket_counts [[buffer(10)]],
    const device int* route_bucket_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_tile_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& route_bucket_count [[buffer(23)]],
    constant const uint& route_bucket_slot_count [[buffer(24)]],
    constant const uint& token_pair_count [[buffer(25)]],
    constant const uint& pair_slot_count [[buffer(26)]],
    constant const uint& topk_slot_count [[buffer(27)]],
    constant const uint& output_microtile_count [[buffer(28)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_bucket = threadgroup_position.x;
  uint token_pair_slot_topk_microtile = threadgroup_position.y;
  uint k_block = threadgroup_position.z;
  uint output_microtile_area = output_microtile_count;
  uint topk_area = topk_slot_count * output_microtile_area;
  uint pair_area = pair_slot_count * topk_area;
  uint token_pair = token_pair_slot_topk_microtile / pair_area;
  uint pair_topk_microtile =
      token_pair_slot_topk_microtile - token_pair * pair_area;
  uint pair_slot = pair_topk_microtile / topk_area;
  uint topk_microtile = pair_topk_microtile - pair_slot * topk_area;
  uint topk_slot = topk_microtile / output_microtile_area;
  uint output_microtile = topk_microtile - topk_slot * output_microtile_area;
  uint n_in_output_microtile = thread_position.x;
  uint output_microtile_width = 64u;
  uint n = output_microtile * output_microtile_width + n_in_output_microtile;
  if (route_bucket >= route_bucket_count ||
      token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count ||
      output_microtile >= output_microtile_count ||
      output_microtile * output_microtile_width >= output_dims ||
      k_block >= k_blocks || n >= output_dims) {
    return;
  }

  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  bool exact_route_slot = true;
  bool exact_token_topk_scatter = true;
  if (route_bucket != 0u || k_block != 0u) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  float token_pair_slot_topk_route_bucket_codeword_reduce_accumulator = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint codeword_tile = 0; codeword_tile < codeword_tile_count;
         ++codeword_tile) {
      uint route_bucket_offset =
          ((((((route_bucket * token_pair_count + token_pair) *
                   pair_slot_count +
               pair_slot) *
                  topk_slot_count +
              topk_slot) *
                 k_blocks +
             reduce_k_block) *
                output_microtile_count +
            output_microtile) *
               codeword_tile_count +
           codeword_tile);
      int route_base_i = route_bucket_offsets[route_bucket_offset];
      int routes_for_slot_i = route_bucket_counts[route_bucket_offset];
      if (route_base_i < 0 || routes_for_slot_i <= 0) {
        continue;
      }
      uint route_base = uint(route_base_i);
      uint routes_for_slot = uint(routes_for_slot_i);
      if (route_base >= route_bucket_slot_count) {
        continue;
      }

      int scale_slot_i =
          codeword_scale_slots[reduce_k_block * codeword_tile_count +
                               codeword_tile];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }
      uint k_code =
          reduce_k_block * codeword_tile_count * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }

      for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
        uint route_slot_index = route_base + route_slot;
        if (route_slot_index >= route_bucket_slot_count) {
          break;
        }
        int source_route_i = route_bucket_route_slot_ids[route_slot_index];
        if (source_route_i < 0 || uint(source_route_i) >= route_count) {
          continue;
        }
        uint source_route = uint(source_route_i);

        uint route_expert = experts;
        for (uint source_tile = 0; source_tile < num_route_tiles;
             ++source_tile) {
          int tile_base_i = tile_offsets[source_tile];
          int routes_in_tile_i = tile_counts[source_tile];
          int expert_i = tile_experts[source_tile];
          if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
            continue;
          }
          uint tile_base = uint(tile_base_i);
          uint tile_limit = tile_base + uint(routes_in_tile_i);
          if (source_route >= tile_base && source_route < tile_limit) {
            route_expert = uint(expert_i);
            break;
          }
        }
        if (route_expert >= experts) {
          continue;
        }

        uint rhs_base =
            (((route_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_tile_count + codeword_tile];
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_slot_topk_route_bucket_codeword_reduce_accumulator +=
            codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_route_bucket_codeword_reduce_accumulator +
      (exact_route_slot ? 0.0f : 0.0f) +
      (exact_token_topk_scatter ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* kblock_microtile_offsets [[buffer(9)]],
    const device int* kblock_microtile_counts [[buffer(10)]],
    const device int* kblock_microtile_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_tile_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& kblock_microtile_slot_count [[buffer(23)]],
    constant const uint& token_pair_count [[buffer(24)]],
    constant const uint& pair_slot_count [[buffer(25)]],
    constant const uint& topk_slot_count [[buffer(26)]],
    constant const uint& output_microtile_count [[buffer(27)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint pair_slot_topk_microtile = threadgroup_position.y;
  uint k_block = threadgroup_position.z;
  uint output_microtile_area = output_microtile_count;
  uint topk_area = topk_slot_count * output_microtile_area;
  uint pair_slot = pair_slot_topk_microtile / topk_area;
  uint topk_microtile = pair_slot_topk_microtile - pair_slot * topk_area;
  uint topk_slot = topk_microtile / output_microtile_area;
  uint output_microtile = topk_microtile - topk_slot * output_microtile_area;
  uint n_in_output_microtile = thread_position.x;
  uint output_microtile_width = 64u;
  uint n = output_microtile * output_microtile_width + n_in_output_microtile;
  if (token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count ||
      output_microtile >= output_microtile_count ||
      output_microtile * output_microtile_width >= output_dims ||
      k_block >= k_blocks || n >= output_dims) {
    return;
  }

  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  bool exact_token_topk_scatter = true;
  bool no_route_slot_expansion = true;
  if (k_block != 0u) {
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  float token_pair_slot_topk_kblock_microtile_stream_accumulator = 0.0f;
  float codeword_tiles_inside_threadgroup = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    uint kblock_microtile_stream =
        (((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
          topk_slot) *
             k_blocks +
         reduce_k_block) *
            output_microtile_count +
        output_microtile;
    int route_base_i = kblock_microtile_offsets[kblock_microtile_stream];
    int routes_for_slot_i = kblock_microtile_counts[kblock_microtile_stream];
    if (route_base_i < 0 || routes_for_slot_i <= 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint routes_for_slot = uint(routes_for_slot_i);
    if (route_base >= kblock_microtile_slot_count) {
      continue;
    }

    for (uint codeword_tile = 0; codeword_tile < codeword_tile_count;
         ++codeword_tile) {
      int scale_slot_i =
          codeword_scale_slots[reduce_k_block * codeword_tile_count +
                               codeword_tile];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }
      uint k_code =
          reduce_k_block * codeword_tile_count * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }

      for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
        uint route_slot_index = route_base + route_slot;
        if (route_slot_index >= kblock_microtile_slot_count) {
          break;
        }
        int source_route_i =
            kblock_microtile_route_slot_ids[route_slot_index];
        if (source_route_i < 0 || uint(source_route_i) >= route_count) {
          continue;
        }
        uint source_route = uint(source_route_i);

        uint route_expert = experts;
        for (uint source_tile = 0; source_tile < num_route_tiles;
             ++source_tile) {
          int tile_base_i = tile_offsets[source_tile];
          int routes_in_tile_i = tile_counts[source_tile];
          int expert_i = tile_experts[source_tile];
          if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
            continue;
          }
          uint tile_base = uint(tile_base_i);
          uint tile_limit = tile_base + uint(routes_in_tile_i);
          if (source_route >= tile_base && source_route < tile_limit) {
            route_expert = uint(expert_i);
            break;
          }
        }
        if (route_expert >= experts) {
          continue;
        }

        uint rhs_base =
            (((route_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_tile_count + codeword_tile];
        codeword_tiles_inside_threadgroup = float(compressed_codeword_tile);
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_slot_topk_kblock_microtile_stream_accumulator +=
            codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_kblock_microtile_stream_accumulator +
      codeword_tiles_inside_threadgroup * 0.0f +
      (no_route_slot_expansion ? 0.0f : 0.0f) +
      (exact_token_topk_scatter ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codeword_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* output_tile_fused_offsets [[buffer(9)]],
    const device int* output_tile_fused_counts [[buffer(10)]],
    const device int* output_tile_fused_route_slot_ids [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_tile_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    constant const uint& output_tile_fused_slot_count [[buffer(23)]],
    constant const uint& token_pair_count [[buffer(24)]],
    constant const uint& pair_slot_count [[buffer(25)]],
    constant const uint& topk_slot_count [[buffer(26)]],
    constant const uint& output_tile_count [[buffer(27)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint token_pair = threadgroup_position.x;
  uint pair_slot_topk_output_tile = threadgroup_position.y;
  uint output_tile_area = output_tile_count;
  uint topk_area = topk_slot_count * output_tile_area;
  uint pair_slot = pair_slot_topk_output_tile / topk_area;
  uint topk_output_tile =
      pair_slot_topk_output_tile - pair_slot * topk_area;
  uint topk_slot = topk_output_tile / output_tile_area;
  uint output_tile = topk_output_tile - topk_slot * output_tile_area;
  uint n_in_output_tile = thread_position.x;
  uint output_tile_width = 64u;
  uint n = output_tile * output_tile_width + n_in_output_tile;
  if (token_pair >= token_pair_count || pair_slot >= pair_slot_count ||
      topk_slot >= topk_slot_count || output_tile >= output_tile_count ||
      output_tile * output_tile_width >= output_dims || n >= output_dims) {
    return;
  }

  uint pair_slot_topk_scatter =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_dims +
      n;
  uint output_tile_slot =
      ((token_pair * pair_slot_count + pair_slot) * topk_slot_count +
       topk_slot) *
          output_tile_count +
      output_tile;
  int route_base_i = output_tile_fused_offsets[output_tile_slot];
  int routes_for_slot_i = output_tile_fused_counts[output_tile_slot];
  if (route_base_i < 0 || routes_for_slot_i <= 0) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }
  uint route_base = uint(route_base_i);
  uint routes_for_slot = uint(routes_for_slot_i);
  if (route_base >= output_tile_fused_slot_count) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[pair_slot_topk_scatter] = half(0.0h);
    return;
  }

  bool exact_token_topk_scatter = true;
  bool output_tile_fused_workgroup = true;
  float token_pair_slot_topk_output_tile_fused_stream_accumulator = 0.0f;
  float codeword_tiles_streamed_inside_output_tile = 0.0f;
  for (uint reduce_k_block = 0; reduce_k_block < k_blocks;
       ++reduce_k_block) {
    for (uint codeword_tile = 0; codeword_tile < codeword_tile_count;
         ++codeword_tile) {
      int scale_slot_i =
          codeword_scale_slots[reduce_k_block * codeword_tile_count +
                               codeword_tile];
      if (scale_slot_i < 0) {
        continue;
      }
      uint active_scale_group = uint(scale_slot_i);
      if (active_scale_group >= scale_groups ||
          scale_group_indices[reduce_k_block * scale_groups +
                              active_scale_group] < 0) {
        continue;
      }
      uint k_code =
          reduce_k_block * codeword_tile_count * 8u + codeword_tile * 8u;
      if (k_code + 7u >= K) {
        continue;
      }

      for (uint route_slot = 0; route_slot < routes_for_slot; ++route_slot) {
        uint route_slot_index = route_base + route_slot;
        if (route_slot_index >= output_tile_fused_slot_count) {
          break;
        }
        int source_route_i =
            output_tile_fused_route_slot_ids[route_slot_index];
        if (source_route_i < 0 || uint(source_route_i) >= route_count) {
          continue;
        }
        uint source_route = uint(source_route_i);

        uint route_expert = experts;
        for (uint source_tile = 0; source_tile < num_route_tiles;
             ++source_tile) {
          int tile_base_i = tile_offsets[source_tile];
          int routes_in_tile_i = tile_counts[source_tile];
          int expert_i = tile_experts[source_tile];
          if (tile_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
            continue;
          }
          uint tile_base = uint(tile_base_i);
          uint tile_limit = tile_base + uint(routes_in_tile_i);
          if (source_route >= tile_base && source_route < tile_limit) {
            route_expert = uint(expert_i);
            break;
          }
        }
        if (route_expert >= experts) {
          continue;
        }

        uint rhs_base =
            (((route_expert * n_tiles + n_tile) * k_blocks +
              reduce_k_block) *
                 bn +
             n_in_tile);
        ushort compressed_codeword_tile =
            codeword_tiles[rhs_base * codeword_tile_count + codeword_tile];
        codeword_tiles_streamed_inside_output_tile =
            float(compressed_codeword_tile);
        float scale_f =
            float(scale_tiles[rhs_base * scale_groups + active_scale_group]);
        uint x_base = source_route * K + k_code;
        float codeword_dot = 0.0f;
        codeword_dot += float(sorted_x[x_base]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                0u);
        codeword_dot += float(sorted_x[x_base + 1u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                1u);
        codeword_dot += float(sorted_x[x_base + 2u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                2u);
        codeword_dot += float(sorted_x[x_base + 3u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                3u);
        codeword_dot += float(sorted_x[x_base + 4u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                4u);
        codeword_dot += float(sorted_x[x_base + 5u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                5u);
        codeword_dot += float(sorted_x[x_base + 6u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                6u);
        codeword_dot += float(sorted_x[x_base + 7u]) *
            mlx_vq_decode_e8p_value(
                uint(compressed_codeword_tile),
                codebook,
                7u);
        token_pair_slot_topk_output_tile_fused_stream_accumulator +=
            codeword_dot * scale_f;
      }
    }
  }
  float q2_scatter =
      token_pair_slot_topk_output_tile_fused_stream_accumulator +
      codeword_tiles_streamed_inside_output_tile * 0.0f +
      (output_tile_fused_workgroup ? 0.0f : 0.0f) +
      (exact_token_topk_scatter ? 0.0f : 0.0f);
  out[pair_slot_topk_scatter] = half(q2_scatter);
}

[[kernel]] void nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
    const device half* route_local_codeword_dot_lut [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    const device int* route_codeword_lut_route_slots [[buffer(9)]],
    const device int* route_codeword_lut_offsets [[buffer(10)]],
    const device int* route_codeword_lut_counts [[buffer(11)]],
    const device int* route_codeword_lut_codeword_ids [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& expert_count [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_block_count [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codeword_count [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& route_tile_count [[buffer(22)]],
    constant const uint& route_slot_count [[buffer(23)]],
    constant const uint& route_codeword_lut_id_count [[buffer(24)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
  uint route_slot = threadgroup_position.x;
  uint scheduled_codeword_axis = threadgroup_position.y;
  uint output_tile = threadgroup_position.z;
  uint n_in_output_tile = thread_position.x;
  uint n = output_tile * 64u + n_in_output_tile;
  if (route_slot >= route_slot_count || n >= output_dims ||
      codeword_count == 0u || k_block_count == 0u) {
    return;
  }

  uint scheduled_k_block = scheduled_codeword_axis / codeword_count;
  uint scheduled_codeword = scheduled_codeword_axis -
      scheduled_k_block * codeword_count;
  if (scheduled_k_block >= k_block_count ||
      scheduled_codeword >= codeword_count) {
    return;
  }

  int route_i = route_codeword_lut_route_slots[route_slot];
  if (route_i < 0 || uint(route_i) >= route_count) {
    return;
  }
  uint route = uint(route_i);

  uint active_expert = expert_count;
  for (uint tile = 0; tile < route_tile_count; ++tile) {
    int route_base_i = tile_offsets[tile];
    int routes_in_tile_i = tile_counts[tile];
    int expert_i = tile_experts[tile];
    if (route_base_i < 0 || routes_in_tile_i <= 0 || expert_i < 0) {
      continue;
    }
    uint route_base = uint(route_base_i);
    uint routes_in_tile = uint(routes_in_tile_i);
    if (route >= route_base && route < route_base + routes_in_tile) {
      active_expert = uint(expert_i);
      break;
    }
  }
  if (active_expert >= expert_count) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  uint n_tile = n / bn;
  uint n_in_tile = n - n_tile * bn;
  if (n_tile >= n_tiles) {
    out[route * output_dims + n] = half(0.0h);
    return;
  }

  if (scheduled_k_block != 0u || scheduled_codeword != 0u) {
    return;
  }

  float route_codeword_lut_accumulators = 0.0f;
  uint lut_offset = uint(max(route_codeword_lut_offsets[route_slot], 0));
  uint lut_count = uint(max(route_codeword_lut_counts[route_slot], 0));
  for (uint k_block = 0; k_block < k_block_count; ++k_block) {
    for (uint lut_index = 0; lut_index < lut_count; ++lut_index) {
      uint codeword_id_index = lut_offset + lut_index;
      if (codeword_id_index >= route_codeword_lut_id_count) {
        continue;
      }
      int codeword_i =
          route_codeword_lut_codeword_ids[codeword_id_index];
      if (codeword_i < 0 || uint(codeword_i) >= codeword_count) {
        codeword_i = int((uint(codeword_i) >> 16) - 1u);
        if (codeword_i < 0 || uint(codeword_i) >= codeword_count) {
          continue;
        }
      }
      uint codeword = uint(codeword_i);
      int scale_slot_i =
          codeword_scale_slots[k_block * codeword_count + codeword];
      if (scale_slot_i < 0 || uint(scale_slot_i) >= scale_groups) {
        continue;
      }
      uint scale_slot = uint(scale_slot_i);
      if (scale_group_indices[k_block * scale_groups + scale_slot] < 0) {
        continue;
      }

      uint rhs_base =
          (((active_expert * n_tiles + n_tile) * k_block_count + k_block) *
               bn +
           n_in_tile);
      ushort compressed_codeword =
          code_tiles[rhs_base * codeword_count + codeword];
      int encoded_codeword_i =
          route_codeword_lut_codeword_ids[codeword_id_index];
      if (encoded_codeword_i >= int(codeword_count)) {
        uint expected_compressed_codeword = uint(encoded_codeword_i) & 0xFFFFu;
        if (uint(compressed_codeword) != expected_compressed_codeword) {
          continue;
        }
      }
      uint codebook_zero =
          codebook[uint(compressed_codeword) & 255u] & 0u;
      float lut_dot = float(route_local_codeword_dot_lut[
          (route * k_block_count + k_block) * route_codeword_lut_id_count +
          codeword_id_index]);
      float scale_f =
          float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      route_codeword_lut_accumulators +=
          lut_dot * scale_f + float(codebook_zero);
    }
  }
  out[route * output_dims + n] = half(route_codeword_lut_accumulators);
}

[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_component_bits [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device int* component_scale_slots [[buffer(6)]],
    const device int* component_codeword_indices [[buffer(7)]],
    const device int* component_offsets [[buffer(8)]],
    const device uint* codebook [[buffer(9)]],
    const device int* tile_experts [[buffer(10)]],
    const device int* tile_offsets [[buffer(11)]],
    const device int* tile_counts [[buffer(12)]],
    device float* component_stream_partials [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& components [[buffer(22)]],
    constant const uint& component_pairs [[buffer(23)]],
    constant const uint& scale_groups [[buffer(24)]],
    constant const uint& num_route_tiles [[buffer(25)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  uint tile_id = tgid.x;
  uint component_pair = tgid.y;
  uint k_block = tgid.z;
  uint lane = tid.x;
  constexpr uint route_tile_size = 64u;
  constexpr uint component_width = 64u;
  threadgroup float component_pair_sums[64];
  if (tile_id >= num_route_tiles || component_pair >= component_pairs ||
      k_block >= k_blocks) {
    return;
  }

  int expert_i = tile_experts[tile_id];
  int route_offset_i = tile_offsets[tile_id];
  int routes_in_tile_i = tile_counts[tile_id];
  bool valid_tile = expert_i >= 0 && uint(expert_i) < experts &&
      route_offset_i >= 0 && routes_in_tile_i > 0;
  uint expert = valid_tile ? uint(expert_i) : 0u;
  uint route_base = valid_tile ? uint(route_offset_i) : 0u;
  uint routes_in_tile = valid_tile ? uint(routes_in_tile_i) : 0u;

  uint total = route_tile_size * n_tiles * component_width;
  uint component_lane = lane & 3u;
  uint output_lane = lane >> 2u;
  for (uint linear = output_lane; linear < total; linear += 16u) {
    uint n_in_tile = linear % component_width;
    uint tile_linear = linear / component_width;
    uint route_slot = tile_linear % route_tile_size;
    uint n_tile = tile_linear / route_tile_size;
    uint route = route_base + route_slot;
    uint n = n_tile * bn + n_in_tile;
    uint partial_offset =
        (((((tile_id * k_blocks + k_block) * component_pairs + component_pair) *
               n_tiles +
           n_tile) *
              route_tile_size +
          route_slot) *
             component_width +
         n_in_tile);
    float partial = 0.0f;
    if (!valid_tile || route_slot >= routes_in_tile || route >= route_count ||
        n >= output_dims || n_tile >= n_tiles || n_in_tile >= bn) {
      partial = 0.0f;
    } else {
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_in_tile);
      uint component_begin = component_pair * 4u;
      uint component = component_begin + component_lane;
      if (component < min(component_begin + 4u, components)) {
        uint map_base = k_block * components;
        int codeword_i = component_codeword_indices[map_base + component];
        int component_offset_i = component_offsets[map_base + component];
        int scale_slot_i = component_scale_slots[map_base + component];
        if (codeword_i < 0 || component_offset_i < 0 || scale_slot_i < 0) {
          partial = 0.0f;
        } else {
          uint codeword_index = uint(codeword_i);
          uint component_offset = uint(component_offset_i);
          uint scale_slot = uint(scale_slot_i);
          if (codeword_index < codewords && component_offset < 8u &&
              scale_slot < scale_groups &&
              scale_group_indices[k_block * scale_groups + scale_slot] >= 0 &&
              codeword_scale_slots[k_block * codewords + codeword_index] >= 0) {
            uint k = k_block * codewords * 8u + component;
            if (k >= K) {
              partial = 0.0f;
            } else {
              uint sign_base = (rhs_base * codewords + codeword_index) * 8u;
              uint sign_bit =
                  uint(sign_component_bits[sign_base + component_offset] & 1u);
              uint signs = 0u;
              for (uint bit = 0; bit < 8u; ++bit) {
                signs |= uint(sign_component_bits[sign_base + bit] & 1u) << bit;
              }
              uint abs_index =
                  uint(abs_index_tiles[rhs_base * codewords + codeword_index]);
              uint parity = mlx_vq_sign_parity8(signs);
              float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
              float component_value = mlx_vq_decode_e8p_split_value(
                  signs, abs_index, parity, codebook, component_offset);
              if (sign_bit > 1u) {
                component_value = 0.0f;
              }
              partial =
                  float(sorted_x[route * K + k]) * component_value * scale_f;
            }
          }
        }
      }
    }
    component_pair_sums[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (component_lane == 0u) {
      component_stream_partials[partial_offset] =
          component_pair_sums[lane] + component_pair_sums[lane + 1u] +
          component_pair_sums[lane + 2u] + component_pair_sums[lane + 3u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void nax_e8p_component_stream_rhs_sorted_tensorops_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_component_bits [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device half* scale_tiles [[buffer(3)]],
    const device int* scale_group_indices [[buffer(4)]],
    const device int* codeword_scale_slots [[buffer(5)]],
    const device int* component_scale_slots [[buffer(6)]],
    const device int* component_codeword_indices [[buffer(7)]],
    const device int* component_offsets [[buffer(8)]],
    const device uint* codebook [[buffer(9)]],
    const device int* tile_experts [[buffer(10)]],
    const device int* tile_offsets [[buffer(11)]],
    const device int* tile_counts [[buffer(12)]],
    device half* out [[buffer(13)]],
    constant const uint& route_count [[buffer(14)]],
    constant const uint& output_dims [[buffer(15)]],
    constant const uint& K [[buffer(16)]],
    constant const uint& experts [[buffer(17)]],
    constant const uint& n_tiles [[buffer(18)]],
    constant const uint& k_blocks [[buffer(19)]],
    constant const uint& bn [[buffer(20)]],
    constant const uint& codewords [[buffer(21)]],
    constant const uint& components [[buffer(22)]],
    constant const uint& scale_groups [[buffer(23)]],
    constant const uint& num_route_tiles [[buffer(24)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint component_offset = k_local - codeword_slot * 8u;
          uint component_index = codeword_slot * 8u + component_offset;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                codeword_slot < codewords && component_index < components) {
              uint map_base = k_block * components + component_index;
              int mapped_codeword = component_codeword_indices[map_base];
              int mapped_offset = component_offsets[map_base];
              int scale_slot_i = component_scale_slots[map_base];
              int codeword_scale_slot_i =
                  codeword_scale_slots[k_block * codewords + codeword_slot];
              if (mapped_codeword == int(codeword_slot) &&
                  mapped_offset == int(component_offset) &&
                  scale_slot_i >= 0 && codeword_scale_slot_i >= 0) {
                uint scale_slot = uint(scale_slot_i);
                if (scale_slot < scale_groups &&
                    scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
                  uint rhs_base =
                      (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                       n_local);
                  uint sign_base = (rhs_base * codewords + codeword_slot) * 8u;
                  uint signs = 0u;
                  for (uint bit = 0; bit < 8u; ++bit) {
                    signs |= uint(sign_component_bits[sign_base + bit] & 1u) << bit;
                  }
                  uint abs_idx =
                      uint(abs_index_tiles[rhs_base * codewords + codeword_slot]);
                  uint parity = mlx_vq_sign_parity8(signs);
                  float scale_f =
                      float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                  value = half(
                      mlx_vq_decode_e8p_split_value(
                          signs, abs_idx, parity, codebook, component_offset) *
                      scale_f);
                }
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_reduce(
    device const float* component_stream_partials [[buffer(0)]],
    device half* out [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device int* tile_counts [[buffer(3)]],
    constant const uint& route_count [[buffer(4)]],
    constant const uint& output_dims [[buffer(5)]],
    constant const uint& n_tiles [[buffer(6)]],
    constant const uint& k_blocks [[buffer(7)]],
    constant const uint& component_pairs [[buffer(8)]],
    constant const uint& num_route_tiles [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  uint tile_id = tgid.x;
  uint n_tile = tgid.y;
  uint lane = tid.x;
  constexpr uint route_tile_size = 64u;
  constexpr uint component_width = 64u;
  if (tile_id >= num_route_tiles || n_tile >= n_tiles) {
    return;
  }

  int route_offset_i = tile_offsets[tile_id];
  int routes_in_tile_i = tile_counts[tile_id];
  if (route_offset_i < 0 || routes_in_tile_i <= 0) {
    return;
  }
  uint route_base = uint(route_offset_i);
  uint routes_in_tile = uint(routes_in_tile_i);

  for (uint linear = lane; linear < route_tile_size * component_width; linear += 64u) {
    uint route_slot = linear / component_width;
    uint n_in_tile = linear - route_slot * component_width;
    uint route = route_base + route_slot;
    uint n = n_tile * component_width + n_in_tile;
    if (route_slot >= routes_in_tile || route >= route_count || n >= output_dims) {
      continue;
    }

    float accum = 0.0f;
    for (uint k_block = 0; k_block < k_blocks; ++k_block) {
      for (uint component_pair = 0; component_pair < component_pairs; ++component_pair) {
        uint partial_offset =
            (((((tile_id * k_blocks + k_block) * component_pairs + component_pair) *
                   n_tiles +
               n_tile) *
                  route_tile_size +
              route_slot) *
                 component_width +
             n_in_tile);
        accum += component_stream_partials[partial_offset];
      }
    }
    out[route * output_dims + n] = half(accum);
  }
}

[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[k_block * scale_groups + scale_slot] >=
                      0) {
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint lut_base = (expert * k_blocks + k_block) * 256u;
                uint sign_slot = uint(sign_byte_slots[factor_index]);
                uint abs_slot = uint(abs_index_slots[factor_index]);
                uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
                uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
                uint parity = mlx_vq_sign_parity8(signs);
                float scale_f =
                    float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                value = half(
                    mlx_vq_decode_e8p_split_value(
                        signs, abs_idx, parity, codebook, dim) *
                    scale_f);
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device float* kblock_partials [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint expert = tgid.x;
  uint contract_k_block = tgid.y;
  uint tile_id = tgid.z;
  if (expert >= experts || contract_k_block >= k_blocks ||
      tile_id >= num_route_tiles) {
    return;
  }

  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) != expert) {
    return;
  }

  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  for (uint n_tile = 0; n_tile < n_tiles; ++n_tile) {
    uint n_tile_base = n_tile * bn;
    float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
    for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
      for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
        c_acc[m_frag][i] = 0.0f;
      }
    }

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[contract_k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                codeword_slot < codewords && scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[contract_k_block * scale_groups + scale_slot] >=
                      0) {
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + contract_k_block) *
                         bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint lut_base = (expert * k_blocks + contract_k_block) * 256u;
                uint sign_slot = uint(sign_byte_slots[factor_index]);
                uint abs_slot = uint(abs_index_slots[factor_index]);
                uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
                uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
                uint parity = mlx_vq_sign_parity8(signs);
                float scale_f =
                    float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                value = half(
                    mlx_vq_decode_e8p_split_value(
                        signs, abs_idx, parity, codebook, dim) *
                    scale_f);
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = contract_k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }

    for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
      uint route_frag_base = route_group * 32u + m_frag * 16u;
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_base = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint route = route_base + route_slot;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_in_tile = n_frag_base + uint(sc.x + col);
            uint n = n_tile_base + n_in_tile;
            if (route_slot < routes_in_tile && route < route_count &&
                n_in_tile < bn && n < output_dims) {
              short c_idx =
                  n_frag * mlx_vq_nax_elems_per_frag +
                  row * mlx_vq_nax_elem_cols + col;
              uint partial_offset =
                  (((tile_id * k_blocks + contract_k_block) * n_tiles + n_tile) *
                       mlx_vq_nax_bn_tile +
                   route_slot) *
                      bn +
                  n_in_tile;
              kblock_partials[partial_offset] = c_acc[m_frag][c_idx];
            }
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce(
    const device float* kblock_partials [[buffer(0)]],
    device half* out [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device int* tile_counts [[buffer(3)]],
    constant const uint& route_count [[buffer(4)]],
    constant const uint& output_dims [[buffer(5)]],
    constant const uint& n_tiles [[buffer(6)]],
    constant const uint& k_blocks [[buffer(7)]],
    constant const uint& bn [[buffer(8)]],
    constant const uint& num_route_tiles [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  uint lane = tid.x;
  uint tile_id = tgid.x;
  uint n_tile = tgid.y;
  if (tile_id >= num_route_tiles || n_tile >= n_tiles) {
    return;
  }

  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  for (uint linear = lane; linear < mlx_vq_nax_bn_tile * bn;
       linear += mlx_vq_nax_threads_per_tg) {
    uint route_slot = linear / bn;
    uint n_in_tile = linear - route_slot * bn;
    uint route = route_base + route_slot;
    uint n = n_tile * bn + n_in_tile;
    if (route_slot >= routes_in_tile || route >= route_count || n >= output_dims) {
      continue;
    }

    float accum = 0.0f;
    for (uint k_block = 0; k_block < k_blocks; ++k_block) {
      uint partial_offset =
          (((tile_id * k_blocks + k_block) * n_tiles + n_tile) *
               mlx_vq_nax_bn_tile +
           route_slot) *
              bn +
          n_in_tile;
      accum += kblock_partials[partial_offset];
    }
    out[route * output_dims + n] = half(accum);
  }
}

[[kernel]] void nax_e8p_packed_rhs_sorted_tiled_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * 8u;
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / 8u;
      uint codeword_slot = idx - n_local * 8u;
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_local);
      int scale_slot_i =
          codeword_scale_slots[k_block * codewords + codeword_slot];
      uint code = 0u;
      float scale_f = 0.0f;
      if (n_tile_base + n_local < output_dims && scale_slot_i >= 0) {
        uint scale_slot = uint(scale_slot_i);
        if (scale_slot < scale_groups &&
            scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
          code = uint(code_tiles[rhs_base * codewords + codeword_slot]);
          scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        }
      }

      uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      staged_b[base] =
          half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      staged_b[base + mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      staged_b[base + 2u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      staged_b[base + 3u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      staged_b[base + 4u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      staged_b[base + 5u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      staged_b[base + 6u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      staged_b[base + 7u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] =
                staged_b[k_local * mlx_vq_nax_bn_tile + n_local];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_split_byte_rhs_sorted_tiled_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_tiles [[buffer(1)]],
    const device uchar* abs_index_tiles [[buffer(2)]],
    const device uchar* parity_tiles [[buffer(3)]],
    const device half* scale_tiles [[buffer(4)]],
    const device int* scale_group_indices [[buffer(5)]],
    const device int* codeword_scale_slots [[buffer(6)]],
    const device uint* codebook [[buffer(7)]],
    const device int* tile_experts [[buffer(8)]],
    const device int* tile_offsets [[buffer(9)]],
    const device int* tile_counts [[buffer(10)]],
    device half* out [[buffer(11)]],
    constant const uint& route_count [[buffer(12)]],
    constant const uint& output_dims [[buffer(13)]],
    constant const uint& K [[buffer(14)]],
    constant const uint& experts [[buffer(15)]],
    constant const uint& n_tiles [[buffer(16)]],
    constant const uint& k_blocks [[buffer(17)]],
    constant const uint& bn [[buffer(18)]],
    constant const uint& codewords [[buffer(19)]],
    constant const uint& scale_groups [[buffer(20)]],
    constant const uint& num_route_tiles [[buffer(21)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * 8u;
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / 8u;
      uint codeword_slot = idx - n_local * 8u;
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_local);
      int scale_slot_i =
          codeword_scale_slots[k_block * codewords + codeword_slot];
      uint signs = 0u;
      uint abs_idx = 0u;
      uint parity = 0u;
      float scale_f = 0.0f;
      if (n_tile_base + n_local < output_dims && scale_slot_i >= 0) {
        uint scale_slot = uint(scale_slot_i);
        if (scale_slot < scale_groups &&
            scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
          uint factor_index = rhs_base * codewords + codeword_slot;
          signs = uint(sign_tiles[factor_index]);
          abs_idx = uint(abs_index_tiles[factor_index]);
          parity = uint(parity_tiles[factor_index]) & 1u;
          scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        }
      }

      uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      staged_b[base] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f);
      staged_b[base + mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f);
      staged_b[base + 2u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f);
      staged_b[base + 3u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f);
      staged_b[base + 4u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f);
      staged_b[base + 5u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f);
      staged_b[base + 6u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f);
      staged_b[base + 7u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] =
                staged_b[k_local * mlx_vq_nax_bn_tile + n_local];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * 8u;
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / 8u;
      uint codeword_slot = idx - n_local * 8u;
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_local);
      int scale_slot_i =
          codeword_scale_slots[k_block * codewords + codeword_slot];
      uint signs = 0u;
      uint abs_idx = 0u;
      uint parity = 0u;
      float scale_f = 0.0f;
      if (n_tile_base + n_local < output_dims && scale_slot_i >= 0) {
        uint scale_slot = uint(scale_slot_i);
        if (scale_slot < scale_groups &&
            scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
          uint factor_index = rhs_base * codewords + codeword_slot;
          uint lut_base = ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
          uint sign_slot = uint(sign_byte_slots[factor_index]);
          uint abs_slot = uint(abs_index_slots[factor_index]);
          signs = uint(sign_byte_lut[lut_base + sign_slot]);
          abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
          parity = mlx_vq_sign_parity8(signs);
          scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        }
      }

      uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      staged_b[base] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 0u) * scale_f);
      staged_b[base + mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 1u) * scale_f);
      staged_b[base + 2u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 2u) * scale_f);
      staged_b[base + 3u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 3u) * scale_f);
      staged_b[base + 4u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 4u) * scale_f);
      staged_b[base + 5u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 5u) * scale_f);
      staged_b[base + 6u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 6u) * scale_f);
      staged_b[base + 7u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, 7u) * scale_f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] =
                staged_b[k_local * mlx_vq_nax_bn_tile + n_local];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint codeword_slot = k_local / 8u;
          uint dim = k_local - codeword_slot * 8u;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            half value = half(0.0h);
            int scale_slot_i =
                codeword_scale_slots[k_block * codewords + codeword_slot];
            if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                scale_slot_i >= 0) {
              uint scale_slot = uint(scale_slot_i);
              if (scale_slot < scale_groups &&
                  scale_group_indices[k_block * scale_groups + scale_slot] >=
                      0) {
                uint rhs_base =
                    (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                     n_local);
                uint factor_index = rhs_base * codewords + codeword_slot;
                uint lut_base =
                    ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
                uint sign_slot = uint(sign_byte_slots[factor_index]);
                uint abs_slot = uint(abs_index_slots[factor_index]);
                uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
                uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
                uint parity = mlx_vq_sign_parity8(signs);
                float scale_f =
                    float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                value = half(
                    mlx_vq_decode_e8p_split_value(
                        signs, abs_idx, parity, codebook, dim) *
                    scale_f);
              }
            }
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] = value;
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device uchar* sign_byte_lut [[buffer(1)]],
    const device uchar* sign_byte_slots [[buffer(2)]],
    const device uchar* abs_index_lut [[buffer(3)]],
    const device uchar* abs_index_slots [[buffer(4)]],
    const device half* scale_tiles [[buffer(5)]],
    const device int* scale_group_indices [[buffer(6)]],
    const device int* codeword_scale_slots [[buffer(7)]],
    const device uint* codebook [[buffer(8)]],
    const device int* tile_experts [[buffer(9)]],
    const device int* tile_offsets [[buffer(10)]],
    const device int* tile_counts [[buffer(11)]],
    device half* out [[buffer(12)]],
    constant const uint& route_count [[buffer(13)]],
    constant const uint& output_dims [[buffer(14)]],
    constant const uint& K [[buffer(15)]],
    constant const uint& experts [[buffer(16)]],
    constant const uint& n_tiles [[buffer(17)]],
    constant const uint& k_blocks [[buffer(18)]],
    constant const uint& bn [[buffer(19)]],
    constant const uint& codewords [[buffer(20)]],
    constant const uint& scale_groups [[buffer(21)]],
    constant const uint& num_route_tiles [[buffer(22)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  constexpr uint mlx_vq_nax_shared_lanes = 32u;
  threadgroup half shared_b
      [2 * mlx_vq_nax_shared_lanes * 2 * mlx_vq_nax_elems_per_frag];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      if (simdgroup_id < 2u) {
        uint decode_n_group = simdgroup_id;
        uint shared_group_base =
            (decode_n_group * mlx_vq_nax_shared_lanes + lane) *
            2u * uint(mlx_vq_nax_elems_per_frag);
        for (short n_frag = 0; n_frag < 2; ++n_frag) {
          uint n_frag_local = decode_n_group * 32u + uint(n_frag * 16);
          for (short row = 0; row < 2; ++row) {
            uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
            uint codeword_slot = k_local / 8u;
            uint dim = k_local - codeword_slot * 8u;
            for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
              uint n_local = n_frag_local + uint(sc.x + col);
              half value = half(0.0h);
              int scale_slot_i =
                  codeword_scale_slots[k_block * codewords + codeword_slot];
              if (n_local < bn && (n_tile_base + n_local) < output_dims &&
                  scale_slot_i >= 0) {
                uint scale_slot = uint(scale_slot_i);
                if (scale_slot < scale_groups &&
                    scale_group_indices[k_block * scale_groups + scale_slot] >=
                        0) {
                  uint rhs_base =
                      (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn +
                       n_local);
                  uint factor_index = rhs_base * codewords + codeword_slot;
                  uint lut_base =
                      ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
                  uint sign_slot = uint(sign_byte_slots[factor_index]);
                  uint abs_slot = uint(abs_index_slots[factor_index]);
                  uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
                  uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
                  uint parity = mlx_vq_sign_parity8(signs);
                  float scale_f =
                      float(scale_tiles[rhs_base * scale_groups + scale_slot]);
                  value = half(
                      mlx_vq_decode_e8p_split_value(
                          signs, abs_idx, parity, codebook, dim) *
                      scale_f);
                }
              }
              shared_b[shared_group_base +
                  uint(n_frag) * uint(mlx_vq_nax_elems_per_frag) +
                  uint(row * mlx_vq_nax_elem_cols + col)] = value;
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      uint shared_group_base =
          (n_group * mlx_vq_nax_shared_lanes + lane) *
          2u * uint(mlx_vq_nax_elems_per_frag);
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        for (short row = 0; row < 2; ++row) {
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] =
                shared_b[shared_group_base +
                    uint(n_frag) * uint(mlx_vq_nax_elems_per_frag) +
                    uint(row * mlx_vq_nax_elem_cols + col)];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_packed_rhs_sorted_tiled_k128_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  threadgroup half staged_b[2u * mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_pair = 0; k_pair < k_blocks; k_pair += 2u) {
    uint stage_blocks = (k_pair + 1u < k_blocks) ? 2u : 1u;
    for (uint idx = thread_idx; idx < stage_blocks * mlx_vq_nax_bn_tile * 8u;
         idx += mlx_vq_nax_threads_per_tg) {
      uint stage = idx / (mlx_vq_nax_bn_tile * 8u);
      uint local_idx = idx - stage * mlx_vq_nax_bn_tile * 8u;
      uint n_local = local_idx / 8u;
      uint codeword_slot = local_idx - n_local * 8u;
      uint k_block = k_pair + stage;
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_local);
      int scale_slot_i =
          codeword_scale_slots[k_block * codewords + codeword_slot];
      uint code = 0u;
      float scale_f = 0.0f;
      if (n_tile_base + n_local < output_dims && scale_slot_i >= 0) {
        uint scale_slot = uint(scale_slot_i);
        if (scale_slot < scale_groups &&
            scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
          code = uint(code_tiles[rhs_base * codewords + codeword_slot]);
          scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        }
      }

      uint stage_offset = stage * mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile;
      uint base =
          stage_offset + codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      staged_b[base] =
          half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      staged_b[base + mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      staged_b[base + 2u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      staged_b[base + 3u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      staged_b[base + 4u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      staged_b[base + 5u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      staged_b[base + 6u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      staged_b[base + 7u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = 0; stage < stage_blocks; ++stage) {
      uint k_block = k_pair + stage;
      uint stage_offset = stage * mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile;
      for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
        for (short n_frag = 0; n_frag < 2; ++n_frag) {
          uint n_frag_local = n_group * 32u + uint(n_frag * 16);
          for (short row = 0; row < 2; ++row) {
            uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
            for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
              uint n_local = n_frag_local + uint(sc.x + col);
              b_t[n_frag * mlx_vq_nax_elems_per_frag +
                  row * mlx_vq_nax_elem_cols + col] =
                  staged_b[stage_offset + k_local * mlx_vq_nax_bn_tile + n_local];
            }
          }
        }

        for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
          uint route_frag_base = route_group * 32u + m_frag * 16u;
          for (short row = 0; row < 2; ++row) {
            uint route_slot =
                route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
            for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
              uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
              half value = half(0.0h);
              if (route_slot < routes_in_tile &&
                  (route_base + route_slot) < route_count && k < K) {
                value = sorted_x[(route_base + route_slot) * K + k];
              }
              a_t[row * mlx_vq_nax_elem_cols + col] = value;
            }
          }

          for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
            c_t[i] = c_acc[m_frag][i];
          }

          matmul_op.run(a_t, b_t, c_t);

          for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
            c_acc[m_frag][i] = c_t[i];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_packed_rhs_sorted_tiled_m128_matmul(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* code_tiles [[buffer(1)]],
    const device half* scale_tiles [[buffer(2)]],
    const device int* scale_group_indices [[buffer(3)]],
    const device int* codeword_scale_slots [[buffer(4)]],
    const device uint* codebook [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& output_dims [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& experts [[buffer(13)]],
    constant const uint& n_tiles [[buffer(14)]],
    constant const uint& k_blocks [[buffer(15)]],
    constant const uint& bn [[buffer(16)]],
    constant const uint& codewords [[buffer(17)]],
    constant const uint& scale_groups [[buffer(18)]],
    constant const uint& num_route_tiles [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint n_tile = tgid.x;
  uint tile_id = tgid.y;
  if (n_tile >= n_tiles || tile_id >= num_route_tiles) {
    return;
  }
  int expert_i = tile_experts[tile_id];
  if (expert_i < 0 || uint(expert_i) >= experts) {
    return;
  }
  uint expert = uint(expert_i);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint n_tile_base = n_tile * bn;
  if (routes_in_tile == 0u || route_base >= route_count ||
      n_tile_base >= output_dims) {
    return;
  }

  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_base = n_tile_base + n_group * 32u;
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      float>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * 8u;
         idx += 256u) {
      uint n_local = idx / 8u;
      uint codeword_slot = idx - n_local * 8u;
      uint rhs_base =
          (((expert * n_tiles + n_tile) * k_blocks + k_block) * bn + n_local);
      int scale_slot_i =
          codeword_scale_slots[k_block * codewords + codeword_slot];
      uint code = 0u;
      float scale_f = 0.0f;
      if (n_tile_base + n_local < output_dims && scale_slot_i >= 0) {
        uint scale_slot = uint(scale_slot_i);
        if (scale_slot < scale_groups &&
            scale_group_indices[k_block * scale_groups + scale_slot] >= 0) {
          code = uint(code_tiles[rhs_base * codewords + codeword_slot]);
          scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
        }
      }

      uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
      staged_b[base] =
          half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      staged_b[base + mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      staged_b[base + 2u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      staged_b[base + 3u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      staged_b[base + 4u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      staged_b[base + 5u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      staged_b[base + 6u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      staged_b[base + 7u * mlx_vq_nax_bn_tile] =
          half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_local = n_group * 32u + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n_local = n_frag_local + uint(sc.x + col);
            b_t[n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col] =
                staged_b[k_local * mlx_vq_nax_bn_tile + n_local];
          }
        }
      }

      for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
        uint route_frag_base = route_group * 32u + m_frag * 16u;
        for (short row = 0; row < 2; ++row) {
          uint route_slot =
              route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint k = k_block * mlx_vq_nax_bk_tile + kk + uint(sc.x + col);
            half value = half(0.0h);
            if (route_slot < routes_in_tile &&
                (route_base + route_slot) < route_count && k < K) {
              value = sorted_x[(route_base + route_slot) * K + k];
            }
            a_t[row * mlx_vq_nax_elem_cols + col] = value;
          }
        }

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_t[i] = c_acc[m_frag][i];
        }

        matmul_op.run(a_t, b_t, c_t);

        for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
          c_acc[m_frag][i] = c_t[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot =
            route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count &&
              n < output_dims) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag +
                row * mlx_vq_nax_elem_cols + col;
            out[route * output_dims + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}

template <typename Tile>
METAL_FUNC void mlx_vq_load_e8p_b_tile_inline(
    thread Tile& tile,
    const device ushort* codes,
    const device half* scales,
    const device uint* codebook,
    uint expert,
    uint n_tile_base,
    uint N,
    uint K,
    uint k_base,
    uint group_size,
    uint codewords,
    uint groups,
    uint tn) {
  const short2 sc = Tile::NAXFrag_t::get_coord();

  STEEL_PRAGMA_UNROLL
  for (short frag_n = 0; frag_n < Tile::kTileRows; ++frag_n) {
    STEEL_PRAGMA_UNROLL
    for (short frag_k = 0; frag_k < Tile::kTileCols; ++frag_k) {
      thread typename Tile::frag_type& frag = tile.frag_at(frag_n, frag_k);

      STEEL_PRAGMA_UNROLL
      for (short row = 0; row < Tile::kFragThrRows; ++row) {
        uint n =
            n_tile_base + tn + uint(frag_n) * uint(Tile::kFragRows) +
            uint(sc.y) + uint(row) * uint(Tile::kFragRowsJump);

        STEEL_PRAGMA_UNROLL
        for (short col = 0; col < Tile::kFragThrCols; ++col) {
          uint k =
              k_base + uint(frag_k) * uint(Tile::kFragCols) + uint(sc.x) +
              uint(col);
          half value = half(0.0h);
          if (n < N && k < K) {
            uint codeword_slot = k >> 3;
            uint dim = k & 7u;
            uint code = uint(codes[(expert * N + n) * codewords + codeword_slot]);
            float scale_f =
                float(scales[(expert * N + n) * groups + (k / group_size)]);
            value = half(mlx_vq_decode_e8p_value(code, codebook, dim) * scale_f);
          }
          frag[row * Tile::kFragThrCols + col] = value;
        }
      }
    }
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_inline_b(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      mlx_vq_load_e8p_b_tile_inline(
          Btile,
          codes,
          scales,
          codebook,
          expert,
          n_tile_base,
          N,
          K,
          k_block + kk,
          group_size,
          codewords,
          groups,
          uint(tn));

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_tgscale(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint MAX_CACHED_GROUPS = 8;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;
  bool use_scale_cache = groups <= MAX_CACHED_GROUPS;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half tg_scales[mlx_vq_nax_bn_tile * MAX_CACHED_GROUPS];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  if (use_scale_cache) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * groups;
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / groups;
      uint group = idx - n_local * groups;
      uint n = n_tile_base + n_local;
      half scale = half(0.0h);
      if (n < N) {
        scale = scales[(expert * N + n) * groups + group];
      }
      tg_scales[n_local * MAX_CACHED_GROUPS + group] = scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint group = k_code / group_size;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = use_scale_cache ? tg_scales[n_local * MAX_CACHED_GROUPS + group]
                                : scales[(expert * N + n) * groups + group];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_tgcb_tgscale(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint MAX_CACHED_GROUPS = 8;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;
  bool use_scale_cache = groups <= MAX_CACHED_GROUPS;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup uint tg_codebook[256];
  threadgroup half tg_scales[mlx_vq_nax_bn_tile * MAX_CACHED_GROUPS];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  for (uint idx = thread_idx; idx < 256u; idx += mlx_vq_nax_threads_per_tg) {
    tg_codebook[idx] = codebook[idx];
  }
  if (use_scale_cache) {
    for (uint idx = thread_idx; idx < mlx_vq_nax_bn_tile * groups;
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / groups;
      uint group = idx - n_local * groups;
      uint n = n_tile_base + n_local;
      half scale = half(0.0h);
      if (n < N) {
        scale = scales[(expert * N + n) * groups + group];
      }
      tg_scales[n_local * MAX_CACHED_GROUPS + group] = scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint group = k_code / group_size;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = use_scale_cache ? tg_scales[n_local * MAX_CACHED_GROUPS + group]
                                : scales[(expert * N + n) * groups + group];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_gs352(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  (void)group_size;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  constexpr uint groups = 4;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + mlx_vq_group_index_gs352(k_code)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_tgcb(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup uint tg_codebook[256];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  for (uint idx = thread_idx; idx < 256u; idx += mlx_vq_nax_threads_per_tg) {
    tg_codebook[idx] = codebook[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value_tg(code, tg_codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_tgcb_hoist(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup uint tg_codebook[256];
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  for (uint idx = thread_idx; idx < 256u; idx += mlx_vq_nax_threads_per_tg) {
    tg_codebook[idx] = codebook[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint signs = code & 0xFFu;
      uint parity = mlx_vq_sign_parity8(signs);
      uint effective_signs = signs ^ parity;
      uint abs_code = tg_codebook[code >> 8];
      float scale_f = float(scale);
      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      Ws[base] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 0u, scale_f);
      Ws[base + 1u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 1u, scale_f);
      Ws[base + 2u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 2u, scale_f);
      Ws[base + 3u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 3u, scale_f);
      Ws[base + 4u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 4u, scale_f);
      Ws[base + 5u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 5u, scale_f);
      Ws[base + 6u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 6u, scale_f);
      Ws[base + 7u] =
          mlx_vq_decode_e8p_hoisted_half(abs_code, effective_signs, parity, 7u, scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_lut(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device half* full_grid [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      const device half* row = full_grid + code * 8u;
      float scale_f = float(scale);
      Ws[base] = half(float(row[0]) * scale_f);
      Ws[base + 1u] = half(float(row[1]) * scale_f);
      Ws[base + 2u] = half(float(row[2]) * scale_f);
      Ws[base + 3u] = half(float(row[3]) * scale_f);
      Ws[base + 4u] = half(float(row[4]) * scale_f);
      Ws[base + 5u] = half(float(row[5]) * scale_f);
      Ws[base + 6u] = half(float(row[6]) * scale_f);
      Ws[base + 7u] = half(float(row[7]) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_bk128(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;

  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 2u));
  const short tn = short(SN * (simdgroup_id & 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk128_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk128_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx;
         idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk128_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk128_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk128_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk128_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk128_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk128_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk128_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m128n32(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 32;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * simdgroup_id);
  constexpr short tn = 0;
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += mlx_vq_nax_threads_per_tg) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base,
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m64n128(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 128;
  constexpr uint THREADS = 256;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tm = short(SM * (simdgroup_id / 4u));
  const short tn = short(SN * (simdgroup_id & 3u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes - int(tm))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += THREADS) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m32n64(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 64;
  constexpr uint THREADS = 64;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  constexpr short tm = 0;
  const short tn = short(SN * simdgroup_id);
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes)));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += THREADS) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m64n64t64(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 64;
  constexpr uint THREADS = 64;
  constexpr uint ROUTE_HALF = 32;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  const short tn = short(SN * simdgroup_id);
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm0 = short(min(int(SM), max(0, valid_routes)));
  short sgp_sm1 = short(min(int(SM), max(0, valid_routes - int(ROUTE_HALF))));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile0;
  NAXTile<float, TM, TN> Dtile1;
  Dtile0.clear();
  Dtile1.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += THREADS) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TN, TK> Btile;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      if (sgp_sm0 > 0) {
        NAXTile<half, TM, TK> Atile;
        const device half* a_ptr = sorted_x + route_base * K + k_block + kk;
        if (sgp_sm0 == SM && psk == SK) {
          Atile.load(a_ptr, int(K));
        } else {
          Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm0));
        }

        tile_matmad_nax(
            Dtile0,
            Atile,
            metal::bool_constant<false>{},
            Btile,
            metal::bool_constant<true>{});
      }

      if (sgp_sm1 > 0) {
        NAXTile<half, TM, TK> Atile;
        const device half* a_ptr =
            sorted_x + (route_base + ROUTE_HALF) * K + k_block + kk;
        if (sgp_sm1 == SM && psk == SK) {
          Atile.load(a_ptr, int(K));
        } else {
          Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm1));
        }

        tile_matmad_nax(
            Dtile1,
            Atile,
            metal::bool_constant<false>{},
            Btile,
            metal::bool_constant<true>{});
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm0 > 0 && sgp_sn > 0) {
    Dtile0.store_safe(
        out + route_base * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm0));
  }
  if (sgp_sm1 > 0 && sgp_sn > 0) {
    Dtile1.store_safe(
        out + (route_base + ROUTE_HALF) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm1));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m32n64t128(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 64;
  constexpr uint THREADS = 128;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  constexpr short tm = 0;
  const bool compute_group = simdgroup_id < 2u;
  const short tn = short(SN * min(simdgroup_id, 1u));
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes)));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += THREADS) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (compute_group) {
      STEEL_PRAGMA_NO_UNROLL
      for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
        NAXTile<half, TM, TK> Atile;
        NAXTile<half, TN, TK> Btile;

        const device half* a_ptr =
            sorted_x + (route_base + uint(tm)) * K + k_block + kk;
        short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
        if (sgp_sm == SM && psk == SK) {
          Atile.load(a_ptr, int(K));
        } else {
          Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
        }
        Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
            Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

        tile_matmad_nax(
            Dtile,
            Atile,
            metal::bool_constant<false>{},
            Btile,
            metal::bool_constant<true>{});
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (compute_group && sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8p_fp16_sorted_matmul_steel_m32n128(
    const device half* sorted_x [[buffer(0)]],
    const device ushort* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device uint* codebook [[buffer(3)]],
    const device int* tile_experts [[buffer(4)]],
    const device int* tile_offsets [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const uint& route_count [[buffer(8)]],
    constant const uint& N [[buffer(9)]],
    constant const uint& K [[buffer(10)]],
    constant const uint& group_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = 32;
  constexpr short SN = 32;
  constexpr short SK = 32;
  constexpr short TM = 2;
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr uint BN = 128;
  constexpr uint THREADS = 128;

  uint n_tile_base = tgid.x * BN;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  constexpr short tm = 0;
  const short tn = short(SN * simdgroup_id);
  int valid_routes = min(int(routes_in_tile), int(route_count - route_base));
  short sgp_sm = short(min(int(SM), max(0, valid_routes)));
  short sgp_sn = short(min(int(SN), max(0, int(N) - int(n_tile_base + uint(tn)))));

  threadgroup half Ws[BN * mlx_vq_nax_bk_padded];

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_idx; idx < BN * (mlx_vq_nax_bk_tile / 8u);
         idx += THREADS) {
      uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
      uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
      uint n = n_tile_base + n_local;
      uint k_code = k_block + codeword_slot * 8u;
      uint code = 0u;
      half scale = half(0.0h);
      if (n < N && k_code < K) {
        code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
        scale = scales[(expert * N + n) * groups + (k_code / group_size)];
      }

      uint base = n_local * mlx_vq_nax_bk_padded + codeword_slot * 8u;
      float scale_f = float(scale);
      Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
      Ws[base + 1u] = half(mlx_vq_decode_e8p_value(code, codebook, 1u) * scale_f);
      Ws[base + 2u] = half(mlx_vq_decode_e8p_value(code, codebook, 2u) * scale_f);
      Ws[base + 3u] = half(mlx_vq_decode_e8p_value(code, codebook, 3u) * scale_f);
      Ws[base + 4u] = half(mlx_vq_decode_e8p_value(code, codebook, 4u) * scale_f);
      Ws[base + 5u] = half(mlx_vq_decode_e8p_value(code, codebook, 5u) * scale_f);
      Ws[base + 6u] = half(mlx_vq_decode_e8p_value(code, codebook, 6u) * scale_f);
      Ws[base + 7u] = half(mlx_vq_decode_e8p_value(code, codebook, 7u) * scale_f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += SK) {
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;

      const device half* a_ptr =
          sorted_x + (route_base + uint(tm)) * K + k_block + kk;
      short psk = short(min(int(SK), max(0, int(K) - int(k_block + kk))));
      if (sgp_sm == SM && psk == SK) {
        Atile.load(a_ptr, int(K));
      } else {
        Atile.load_safe(a_ptr, int(K), short2(psk, sgp_sm));
      }
      Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(
          Ws + uint(tn) * mlx_vq_nax_bk_padded + kk);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm > 0 && sgp_sn > 0) {
    Dtile.store_safe(
        out + (route_base + uint(tm)) * N + n_tile_base + uint(tn),
        int(N),
        short2(sgp_sn, sgp_sm));
  }
}

[[kernel]] void nax_e8_int8_routed_matmul(
    const device int8_t* x_q [[buffer(0)]],
    const device half* x_scales [[buffer(1)]],
    const device uchar* codes [[buffer(2)]],
    const device half* scales [[buffer(3)]],
    const device uint* codebook [[buffer(4)]],
    const device int* lhs_indices [[buffer(5)]],
    const device int* tile_experts [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* tile_counts [[buffer(8)]],
    device half* out [[buffer(9)]],
    constant const uint& route_count [[buffer(10)]],
    constant const uint& N [[buffer(11)]],
    constant const uint& K [[buffer(12)]],
    constant const uint& group_size [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint route_group = simdgroup_id / 2u;
  uint n_group = simdgroup_id - route_group * 2u;
  uint n_tile_base = tgid.x * mlx_vq_nax_bn_tile;
  uint n_base = n_tile_base + n_group * 32u;
  uint tile_id = tgid.y;
  uint expert = uint(tile_experts[tile_id]);
  uint route_base = uint(tile_offsets[tile_id]);
  uint routes_in_tile = uint(tile_counts[tile_id]);
  uint codewords = K >> 3;
  uint groups = K / group_size;

  if (routes_in_tile == 0u || route_base >= route_count) {
    return;
  }

  threadgroup int8_t staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      false,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto a_t =
      matmul_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto b_t =
      matmul_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto c_t = matmul_op.get_destination_cooperative_tensor<
      decltype(a_t),
      decltype(b_t),
      int32_t>();

  short2 sc = mlx_vq_nax_get_coord(ushort(lane));
  float c_acc[2][2 * mlx_vq_nax_elems_per_frag];
  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
      c_acc[m_frag][i] = 0.0f;
    }
  }

  for (uint group = 0; group < groups; ++group) {
    uint group_start = group * group_size;
    uint group_end = min(K, group_start + group_size);
    int32_t c_int[2][2 * mlx_vq_nax_elems_per_frag];
    for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
      for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
        c_int[m_frag][i] = 0;
      }
    }

    for (uint k_block = group_start; k_block < group_end; k_block += mlx_vq_nax_bk_tile) {
      for (uint idx = thread_idx;
           idx < mlx_vq_nax_bn_tile * (mlx_vq_nax_bk_tile / 8u);
           idx += mlx_vq_nax_threads_per_tg) {
        uint n_local = idx / (mlx_vq_nax_bk_tile / 8u);
        uint codeword_slot = idx - n_local * (mlx_vq_nax_bk_tile / 8u);
        uint n = n_tile_base + n_local;
        uint k_code = k_block + codeword_slot * 8u;
        uint packed = 0u;
        if (n < N && k_code < group_end) {
          uint code = uint(codes[(expert * N + n) * codewords + (k_code >> 3)]);
          packed = codebook[code];
        }

        uint base = codeword_slot * 8u * mlx_vq_nax_bn_tile + n_local;
        staged_b[base] = mlx_vq_decode_packed_int2_i8(packed, 0u);
        staged_b[base + mlx_vq_nax_bn_tile] = mlx_vq_decode_packed_int2_i8(packed, 1u);
        staged_b[base + 2u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 2u);
        staged_b[base + 3u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 3u);
        staged_b[base + 4u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 4u);
        staged_b[base + 5u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 5u);
        staged_b[base + 6u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 6u);
        staged_b[base + 7u * mlx_vq_nax_bn_tile] =
            mlx_vq_decode_packed_int2_i8(packed, 7u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
        for (short n_frag = 0; n_frag < 2; ++n_frag) {
          uint n_frag_local = n_group * 32u + uint(n_frag * 16);
          for (short row = 0; row < 2; ++row) {
            uint k_local = kk + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
            for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
              uint n_local = n_frag_local + uint(sc.x + col);
              b_t[n_frag * mlx_vq_nax_elems_per_frag + row * mlx_vq_nax_elem_cols + col] =
                  (k_block + k_local < group_end)
                  ? staged_b[k_local * mlx_vq_nax_bn_tile + n_local]
                  : int8_t(0);
            }
          }
        }

        for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
          uint route_frag_base = route_group * 32u + m_frag * 16u;
          for (short row = 0; row < 2; ++row) {
            uint route_slot = route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
            for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
              uint k = k_block + kk + uint(sc.x + col);
              int8_t value = int8_t(0);
              if (route_slot < routes_in_tile && (route_base + route_slot) < route_count &&
                  k < group_end) {
                uint token = uint(lhs_indices[route_base + route_slot]);
                value = x_q[token * K + k];
              }
              a_t[row * mlx_vq_nax_elem_cols + col] = value;
            }
          }

          for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
            c_t[i] = c_int[m_frag][i];
          }

          matmul_op.run(a_t, b_t, c_t);

          for (short i = 0; i < 2 * mlx_vq_nax_elems_per_frag; ++i) {
            c_int[m_frag][i] = c_t[i];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
      uint route_frag_base = route_group * 32u + m_frag * 16u;
      for (short n_frag = 0; n_frag < 2; ++n_frag) {
        uint n_frag_base = n_base + uint(n_frag * 16);
        for (short row = 0; row < 2; ++row) {
          uint route_slot = route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
          uint route = route_base + route_slot;
          for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
            uint n = n_frag_base + uint(sc.x + col);
            if (route_slot < routes_in_tile && route < route_count && n < N) {
              uint token = uint(lhs_indices[route]);
              short c_idx =
                  n_frag * mlx_vq_nax_elems_per_frag + row * mlx_vq_nax_elem_cols + col;
              float a_scale = float(x_scales[token * groups + group]);
              float w_scale = float(scales[(expert * N + n) * groups + group]);
              c_acc[m_frag][c_idx] += float(c_int[m_frag][c_idx]) * a_scale * w_scale * 0.5f;
            }
          }
        }
      }
    }
  }

  for (uint m_frag = 0; m_frag < 2u; ++m_frag) {
    uint route_frag_base = route_group * 32u + m_frag * 16u;
    for (short n_frag = 0; n_frag < 2; ++n_frag) {
      uint n_frag_base = n_base + uint(n_frag * 16);
      for (short row = 0; row < 2; ++row) {
        uint route_slot = route_frag_base + uint(sc.y + row * mlx_vq_nax_elem_rows_jump);
        uint route = route_base + route_slot;
        for (short col = 0; col < mlx_vq_nax_elem_cols; ++col) {
          uint n = n_frag_base + uint(sc.x + col);
          if (route_slot < routes_in_tile && route < route_count && n < N) {
            short c_idx =
                n_frag * mlx_vq_nax_elems_per_frag + row * mlx_vq_nax_elem_cols + col;
            out[route * N + n] = half(c_acc[m_frag][c_idx]);
          }
        }
      }
    }
  }
}
