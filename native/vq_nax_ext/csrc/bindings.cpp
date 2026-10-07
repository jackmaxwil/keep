#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/ops.h"
#include "mlx/version.h"
#include "nax_fp16_primitive.h"

#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

namespace {

nb::object mlx_array_type() {
  return nb::module_::import_("mlx.core").attr("array");
}

mx::array& array_from_handle(nb::handle h, const char* name) {
  auto array_type = mlx_array_type();
  int is_array = PyObject_IsInstance(h.ptr(), array_type.ptr());
  if (is_array < 0) {
    throw nb::python_error();
  }
  if (!is_array) {
    throw nb::type_error((std::string(name) + " must be an mlx.core.array").c_str());
  }
  return *nb::inst_ptr<mx::array>(h);
}

}  // namespace

void predecoded_fp16_matmul_into(
    nb::handle x_h,
    nb::handle weight_t_h,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& weight_t = array_from_handle(weight_t_h, "weight_t");
  auto& out = array_from_handle(out_h, "out");
  auto stream = mx::to_stream({});
  out = mx::matmul(x, weight_t, stream);
}

void predecoded_fp16_gather_mm_into(
    nb::handle x_h,
    nb::handle weight_t_h,
    nb::handle rhs_indices_h,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& weight_t = array_from_handle(weight_t_h, "weight_t");
  auto& rhs_indices = array_from_handle(rhs_indices_h, "rhs_indices");
  auto& out = array_from_handle(out_h, "out");
  auto stream = mx::to_stream({});
  out = mx::gather_mm(x, weight_t, std::nullopt, rhs_indices, true, stream);
}

void nax_fp16_matmul_tile_into(
    nb::handle x_h,
    nb::handle weight_t_h,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& weight_t = array_from_handle(weight_t_h, "weight_t");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::fp16_matmul_tile(x, weight_t, kernel_dir, {});
}

void nax_e8_fp16_matmul_tile_into(
    nb::handle x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_fp16_matmul_tile(x, codes, scales, codebook, group_size, kernel_dir, {});
}

void nax_e8_fp16_matmul_into(
    nb::handle x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_fp16_matmul(x, codes, scales, codebook, group_size, kernel_dir, {});
}

void nax_e8_fp16_routed_matmul_into(
    nb::handle x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle lhs_indices_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& lhs_indices = array_from_handle(lhs_indices_h, "lhs_indices");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_fp16_routed_matmul(
      x,
      codes,
      scales,
      codebook,
      lhs_indices,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8_fp16_routed_steel_matmul_into(
    nb::handle x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle lhs_indices_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& lhs_indices = array_from_handle(lhs_indices_h, "lhs_indices");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_fp16_routed_steel_matmul(
      x,
      codes,
      scales,
      codebook,
      lhs_indices,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8_fp16_sorted_steel_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_fp16_sorted_steel_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_packed_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle code_tile_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& code_tile = array_from_handle(code_tile_h, "code_tile");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_packed_rhs_tile_matmul(
      x,
      code_tile,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
	      kernel_dir,
		      {});
  }

void nax_e8p_split_byte_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle sign_tile_h,
    nb::handle abs_index_tile_h,
    nb::handle parity_tile_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& sign_tile = array_from_handle(sign_tile_h, "sign_tile");
  auto& abs_index_tile = array_from_handle(abs_index_tile_h, "abs_index_tile");
  auto& parity_tile = array_from_handle(parity_tile_h, "parity_tile");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_rhs_tile_matmul(
      x,
      sign_tile,
      abs_index_tile,
      parity_tile,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_abs_index_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle sign_low_nibble_tile_h,
    nb::handle sign_high_nibble_tile_h,
    nb::handle abs_index_tile_h,
    nb::handle parity_tile_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& sign_low_nibble_tile =
      array_from_handle(sign_low_nibble_tile_h, "sign_low_nibble_tile");
  auto& sign_high_nibble_tile =
      array_from_handle(sign_high_nibble_tile_h, "sign_high_nibble_tile");
  auto& abs_index_tile = array_from_handle(abs_index_tile_h, "abs_index_tile");
  auto& parity_tile = array_from_handle(parity_tile_h, "parity_tile");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_abs_index_rhs_tile_matmul(
      x,
      sign_low_nibble_tile,
      sign_high_nibble_tile,
      abs_index_tile,
      parity_tile,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
      kernel_dir,
      {});
}

void nax_e8p_sign_plane_abs_index_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle sign_bit_planes_h,
    nb::handle abs_index_tile_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& sign_bit_planes =
      array_from_handle(sign_bit_planes_h, "sign_bit_planes");
  auto& abs_index_tile = array_from_handle(abs_index_tile_h, "abs_index_tile");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_plane_abs_index_rhs_tile_matmul(
      x,
      sign_bit_planes,
      abs_index_tile,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle sign_low_nibble_lut_h,
    nb::handle sign_low_nibble_slots_h,
    nb::handle sign_high_nibble_lut_h,
    nb::handle sign_high_nibble_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& sign_low_nibble_lut =
      array_from_handle(sign_low_nibble_lut_h, "sign_low_nibble_lut");
  auto& sign_low_nibble_slots =
      array_from_handle(sign_low_nibble_slots_h, "sign_low_nibble_slots");
  auto& sign_high_nibble_lut =
      array_from_handle(sign_high_nibble_lut_h, "sign_high_nibble_lut");
  auto& sign_high_nibble_slots =
      array_from_handle(sign_high_nibble_slots_h, "sign_high_nibble_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_micro_lut_rhs_tile_matmul(
      x,
      sign_low_nibble_lut,
      sign_low_nibble_slots,
      sign_high_nibble_lut,
      sign_high_nibble_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_factor_reuse_rhs_tile_matmul_into(
    nb::handle x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tile_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    int output_count,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x = array_from_handle(x_h, "x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tile = array_from_handle(scale_tile_h, "scale_tile");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_factor_reuse_rhs_tile_matmul(
      x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tile,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      output_count,
      kernel_dir,
      {});
}

void nax_e8p_packed_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_packed_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
	      kernel_dir,
		      {});
	}

void nax_e8p_split_byte_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_tiles_h,
    nb::handle abs_index_tiles_h,
    nb::handle parity_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_tiles = array_from_handle(sign_tiles_h, "sign_tiles");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& parity_tiles = array_from_handle(parity_tiles_h, "parity_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_rhs_sorted_matmul(
      sorted_x,
      sign_tiles,
      abs_index_tiles,
      parity_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_low_nibble_tiles_h,
    nb::handle sign_high_nibble_tiles_h,
    nb::handle abs_index_tiles_h,
    nb::handle parity_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_low_nibble_tiles =
      array_from_handle(sign_low_nibble_tiles_h, "sign_low_nibble_tiles");
  auto& sign_high_nibble_tiles =
      array_from_handle(sign_high_nibble_tiles_h, "sign_high_nibble_tiles");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& parity_tiles = array_from_handle(parity_tiles_h, "parity_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_abs_index_rhs_sorted_matmul(
      sorted_x,
      sign_low_nibble_tiles,
      sign_high_nibble_tiles,
      abs_index_tiles,
      parity_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_plane_abs_index_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_bit_planes_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_bit_planes =
      array_from_handle(sign_bit_planes_h, "sign_bit_planes");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_plane_abs_index_rhs_sorted_matmul(
      sorted_x,
      sign_bit_planes,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_low_nibble_lut_h,
    nb::handle sign_low_nibble_slots_h,
    nb::handle sign_high_nibble_lut_h,
    nb::handle sign_high_nibble_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_low_nibble_lut =
      array_from_handle(sign_low_nibble_lut_h, "sign_low_nibble_lut");
  auto& sign_low_nibble_slots =
      array_from_handle(sign_low_nibble_slots_h, "sign_low_nibble_slots");
  auto& sign_high_nibble_lut =
      array_from_handle(sign_high_nibble_lut_h, "sign_high_nibble_lut");
  auto& sign_high_nibble_slots =
      array_from_handle(sign_high_nibble_slots_h, "sign_high_nibble_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_micro_lut_rhs_sorted_matmul(
      sorted_x,
      sign_low_nibble_lut,
      sign_low_nibble_slots,
      sign_high_nibble_lut,
      sign_high_nibble_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_low_nibble_tiles_h,
    nb::handle sign_high_nibble_tiles_h,
    nb::handle abs_index_tiles_h,
    nb::handle parity_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_low_nibble_tiles =
      array_from_handle(sign_low_nibble_tiles_h, "sign_low_nibble_tiles");
  auto& sign_high_nibble_tiles =
      array_from_handle(sign_high_nibble_tiles_h, "sign_high_nibble_tiles");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& parity_tiles = array_from_handle(parity_tiles_h, "parity_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
      sorted_x,
      sign_low_nibble_tiles,
      sign_high_nibble_tiles,
      abs_index_tiles,
      parity_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_bit_planes_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_bit_planes =
      array_from_handle(sign_bit_planes_h, "sign_bit_planes");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
      sorted_x,
      sign_bit_planes,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_low_nibble_lut_h,
    nb::handle sign_low_nibble_slots_h,
    nb::handle sign_high_nibble_lut_h,
    nb::handle sign_high_nibble_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_low_nibble_lut =
      array_from_handle(sign_low_nibble_lut_h, "sign_low_nibble_lut");
  auto& sign_low_nibble_slots =
      array_from_handle(sign_low_nibble_slots_h, "sign_low_nibble_slots");
  auto& sign_high_nibble_lut =
      array_from_handle(sign_high_nibble_lut_h, "sign_high_nibble_lut");
  auto& sign_high_nibble_slots =
      array_from_handle(sign_high_nibble_slots_h, "sign_high_nibble_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
      sorted_x,
      sign_low_nibble_lut,
      sign_low_nibble_slots,
      sign_high_nibble_lut,
      sign_high_nibble_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_factor_reuse_rhs_sorted_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_kblock_factor_reuse_rhs_sorted_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_component_stream_rhs_sorted_scalar_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_component_bits_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle component_scale_slots_h,
    nb::handle component_codeword_indices_h,
    nb::handle component_offsets_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_component_bits =
      array_from_handle(sign_component_bits_h, "sign_component_bits");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& component_scale_slots =
      array_from_handle(component_scale_slots_h, "component_scale_slots");
  auto& component_codeword_indices =
      array_from_handle(component_codeword_indices_h, "component_codeword_indices");
  auto& component_offsets =
      array_from_handle(component_offsets_h, "component_offsets");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_component_stream_rhs_sorted_scalar_matmul(
      sorted_x,
      sign_component_bits,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      component_scale_slots,
      component_codeword_indices,
      component_offsets,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_slot_codeword_stream_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle active_route_tiles_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& active_route_tiles =
      array_from_handle(active_route_tiles_h, "active_route_tiles");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      active_route_tiles,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle expert_cohort_offsets_h,
    nb::handle expert_cohort_counts_h,
    nb::handle route_cohort_offsets_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& expert_cohort_offsets =
      array_from_handle(expert_cohort_offsets_h, "expert_cohort_offsets");
  auto& expert_cohort_counts =
      array_from_handle(expert_cohort_counts_h, "expert_cohort_counts");
  auto& route_cohort_offsets =
      array_from_handle(route_cohort_offsets_h, "route_cohort_offsets");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      expert_cohort_offsets,
      expert_cohort_counts,
      route_cohort_offsets,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_batch_segment_offsets_h,
    nb::handle route_batch_segment_counts_h,
    nb::handle route_batch_route_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_batch_segment_offsets = array_from_handle(
      route_batch_segment_offsets_h, "route_batch_segment_offsets");
  auto& route_batch_segment_counts = array_from_handle(
      route_batch_segment_counts_h, "route_batch_segment_counts");
  auto& route_batch_route_ids =
      array_from_handle(route_batch_route_ids_h, "route_batch_route_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      route_batch_segment_offsets,
      route_batch_segment_counts,
      route_batch_route_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_cohort_offsets_h,
    nb::handle token_cohort_counts_h,
    nb::handle token_cohort_active_expert_ids_h,
    nb::handle token_cohort_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_cohort_offsets =
      array_from_handle(token_cohort_offsets_h, "token_cohort_offsets");
  auto& token_cohort_counts =
      array_from_handle(token_cohort_counts_h, "token_cohort_counts");
  auto& token_cohort_active_expert_ids = array_from_handle(
      token_cohort_active_expert_ids_h, "token_cohort_active_expert_ids");
  auto& token_cohort_route_slot_ids = array_from_handle(
      token_cohort_route_slot_ids_h, "token_cohort_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_cohort_offsets,
      token_cohort_counts,
      token_cohort_active_expert_ids,
      token_cohort_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_cohort_offsets_h,
    nb::handle token_cohort_counts_h,
    nb::handle token_cohort_active_expert_ids_h,
    nb::handle token_cohort_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_cohort_offsets =
      array_from_handle(token_cohort_offsets_h, "token_cohort_offsets");
  auto& token_cohort_counts =
      array_from_handle(token_cohort_counts_h, "token_cohort_counts");
  auto& token_cohort_active_expert_ids = array_from_handle(
      token_cohort_active_expert_ids_h, "token_cohort_active_expert_ids");
  auto& token_cohort_route_slot_ids = array_from_handle(
      token_cohort_route_slot_ids_h, "token_cohort_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_cohort_offsets,
      token_cohort_counts,
      token_cohort_active_expert_ids,
      token_cohort_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle output_stationary_route_batch_offsets_h,
    nb::handle output_stationary_route_batch_counts_h,
    nb::handle output_stationary_route_batch_active_expert_ids_h,
    nb::handle output_stationary_route_batch_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& output_stationary_route_batch_offsets = array_from_handle(
      output_stationary_route_batch_offsets_h,
      "output_stationary_route_batch_offsets");
  auto& output_stationary_route_batch_counts = array_from_handle(
      output_stationary_route_batch_counts_h,
      "output_stationary_route_batch_counts");
  auto& output_stationary_route_batch_active_expert_ids = array_from_handle(
      output_stationary_route_batch_active_expert_ids_h,
      "output_stationary_route_batch_active_expert_ids");
  auto& output_stationary_route_batch_route_slot_ids = array_from_handle(
      output_stationary_route_batch_route_slot_ids_h,
      "output_stationary_route_batch_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_stationary_route_batch_offsets,
      output_stationary_route_batch_counts,
      output_stationary_route_batch_active_expert_ids,
      output_stationary_route_batch_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle input_stationary_route_batch_offsets_h,
    nb::handle input_stationary_route_batch_counts_h,
    nb::handle input_stationary_route_batch_active_expert_ids_h,
    nb::handle input_stationary_route_batch_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& input_stationary_route_batch_offsets = array_from_handle(
      input_stationary_route_batch_offsets_h,
      "input_stationary_route_batch_offsets");
  auto& input_stationary_route_batch_counts = array_from_handle(
      input_stationary_route_batch_counts_h,
      "input_stationary_route_batch_counts");
  auto& input_stationary_route_batch_active_expert_ids = array_from_handle(
      input_stationary_route_batch_active_expert_ids_h,
      "input_stationary_route_batch_active_expert_ids");
  auto& input_stationary_route_batch_route_slot_ids = array_from_handle(
      input_stationary_route_batch_route_slot_ids_h,
      "input_stationary_route_batch_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      input_stationary_route_batch_offsets,
      input_stationary_route_batch_counts,
      input_stationary_route_batch_active_expert_ids,
      input_stationary_route_batch_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_factor_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_factor_tiles =
      array_from_handle(codeword_factor_tiles_h, "codeword_factor_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul(
      sorted_x,
      codeword_factor_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul_into(
    nb::handle route_local_codeword_dot_lut_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_codeword_lut_route_slots_h,
    nb::handle route_codeword_lut_offsets_h,
    nb::handle route_codeword_lut_counts_h,
    nb::handle route_codeword_lut_codeword_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& route_local_codeword_dot_lut = array_from_handle(
      route_local_codeword_dot_lut_h, "route_local_codeword_dot_lut");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_codeword_lut_route_slots = array_from_handle(
      route_codeword_lut_route_slots_h, "route_codeword_lut_route_slots");
  auto& route_codeword_lut_offsets = array_from_handle(
      route_codeword_lut_offsets_h, "route_codeword_lut_offsets");
  auto& route_codeword_lut_counts = array_from_handle(
      route_codeword_lut_counts_h, "route_codeword_lut_counts");
  auto& route_codeword_lut_codeword_ids = array_from_handle(
      route_codeword_lut_codeword_ids_h, "route_codeword_lut_codeword_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
      route_local_codeword_dot_lut,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      route_codeword_lut_route_slots,
      route_codeword_lut_offsets,
      route_codeword_lut_counts,
      route_codeword_lut_codeword_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle rowwise_route_microtile_offsets_h,
    nb::handle rowwise_route_microtile_counts_h,
    nb::handle rowwise_route_microtile_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& rowwise_route_microtile_offsets = array_from_handle(
      rowwise_route_microtile_offsets_h, "rowwise_route_microtile_offsets");
  auto& rowwise_route_microtile_counts = array_from_handle(
      rowwise_route_microtile_counts_h, "rowwise_route_microtile_counts");
  auto& rowwise_route_microtile_route_slot_ids = array_from_handle(
      rowwise_route_microtile_route_slot_ids_h,
      "rowwise_route_microtile_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      rowwise_route_microtile_offsets,
      rowwise_route_microtile_counts,
      rowwise_route_microtile_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle output_tile_local_route_microtile_offsets_h,
    nb::handle output_tile_local_route_microtile_counts_h,
    nb::handle output_tile_local_route_microtile_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& output_tile_local_route_microtile_offsets = array_from_handle(
      output_tile_local_route_microtile_offsets_h,
      "output_tile_local_route_microtile_offsets");
  auto& output_tile_local_route_microtile_counts = array_from_handle(
      output_tile_local_route_microtile_counts_h,
      "output_tile_local_route_microtile_counts");
  auto& output_tile_local_route_microtile_route_slot_ids = array_from_handle(
      output_tile_local_route_microtile_route_slot_ids_h,
      "output_tile_local_route_microtile_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_tile_local_route_microtile_offsets,
      output_tile_local_route_microtile_counts,
      output_tile_local_route_microtile_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_microtile_codeword_block_reduce_offsets_h,
    nb::handle route_microtile_codeword_block_reduce_counts_h,
    nb::handle route_microtile_codeword_block_reduce_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_microtile_codeword_block_reduce_offsets = array_from_handle(
      route_microtile_codeword_block_reduce_offsets_h,
      "route_microtile_codeword_block_reduce_offsets");
  auto& route_microtile_codeword_block_reduce_counts = array_from_handle(
      route_microtile_codeword_block_reduce_counts_h,
      "route_microtile_codeword_block_reduce_counts");
  auto& route_microtile_codeword_block_reduce_route_slot_ids =
      array_from_handle(
          route_microtile_codeword_block_reduce_route_slot_ids_h,
          "route_microtile_codeword_block_reduce_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      route_microtile_codeword_block_reduce_offsets,
      route_microtile_codeword_block_reduce_counts,
      route_microtile_codeword_block_reduce_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle kblock_wavefront_codeword_scan_offsets_h,
    nb::handle kblock_wavefront_codeword_scan_counts_h,
    nb::handle kblock_wavefront_codeword_scan_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& kblock_wavefront_codeword_scan_offsets = array_from_handle(
      kblock_wavefront_codeword_scan_offsets_h,
      "kblock_wavefront_codeword_scan_offsets");
  auto& kblock_wavefront_codeword_scan_counts = array_from_handle(
      kblock_wavefront_codeword_scan_counts_h,
      "kblock_wavefront_codeword_scan_counts");
  auto& kblock_wavefront_codeword_scan_route_slot_ids = array_from_handle(
      kblock_wavefront_codeword_scan_route_slot_ids_h,
      "kblock_wavefront_codeword_scan_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      kblock_wavefront_codeword_scan_offsets,
      kblock_wavefront_codeword_scan_counts,
      kblock_wavefront_codeword_scan_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_route_output_stripe_offsets_h,
    nb::handle token_route_output_stripe_counts_h,
    nb::handle token_route_output_stripe_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_route_output_stripe_offsets = array_from_handle(
      token_route_output_stripe_offsets_h,
      "token_route_output_stripe_offsets");
  auto& token_route_output_stripe_counts = array_from_handle(
      token_route_output_stripe_counts_h,
      "token_route_output_stripe_counts");
  auto& token_route_output_stripe_route_slot_ids = array_from_handle(
      token_route_output_stripe_route_slot_ids_h,
      "token_route_output_stripe_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_route_output_stripe_offsets,
      token_route_output_stripe_counts,
      token_route_output_stripe_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle scale_group_route_block_offsets_h,
    nb::handle scale_group_route_block_counts_h,
    nb::handle scale_group_route_block_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& scale_group_route_block_offsets = array_from_handle(
      scale_group_route_block_offsets_h, "scale_group_route_block_offsets");
  auto& scale_group_route_block_counts = array_from_handle(
      scale_group_route_block_counts_h, "scale_group_route_block_counts");
  auto& scale_group_route_block_route_slot_ids = array_from_handle(
      scale_group_route_block_route_slot_ids_h,
      "scale_group_route_block_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      scale_group_route_block_offsets,
      scale_group_route_block_counts,
      scale_group_route_block_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_block_output_group_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_block_output_group_offsets_h,
    nb::handle route_block_output_group_counts_h,
    nb::handle route_block_output_group_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_block_output_group_offsets = array_from_handle(
      route_block_output_group_offsets_h,
      "route_block_output_group_offsets");
  auto& route_block_output_group_counts = array_from_handle(
      route_block_output_group_counts_h,
      "route_block_output_group_counts");
  auto& route_block_output_group_route_slot_ids = array_from_handle(
      route_block_output_group_route_slot_ids_h,
      "route_block_output_group_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_block_output_group_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      route_block_output_group_offsets,
      route_block_output_group_counts,
      route_block_output_group_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle output_group_pretransposed_route_offsets_h,
    nb::handle output_group_pretransposed_route_counts_h,
    nb::handle output_group_pretransposed_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& output_group_pretransposed_route_offsets = array_from_handle(
      output_group_pretransposed_route_offsets_h,
      "output_group_pretransposed_route_offsets");
  auto& output_group_pretransposed_route_counts = array_from_handle(
      output_group_pretransposed_route_counts_h,
      "output_group_pretransposed_route_counts");
  auto& output_group_pretransposed_route_slot_ids = array_from_handle(
      output_group_pretransposed_route_slot_ids_h,
      "output_group_pretransposed_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_group_pretransposed_route_offsets,
      output_group_pretransposed_route_counts,
      output_group_pretransposed_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle kblock_route_fused_offsets_h,
    nb::handle kblock_route_fused_counts_h,
    nb::handle kblock_route_fused_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& kblock_route_fused_offsets =
      array_from_handle(kblock_route_fused_offsets_h, "kblock_route_fused_offsets");
  auto& kblock_route_fused_counts =
      array_from_handle(kblock_route_fused_counts_h, "kblock_route_fused_counts");
  auto& kblock_route_fused_route_slot_ids = array_from_handle(
      kblock_route_fused_route_slot_ids_h,
      "kblock_route_fused_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      kblock_route_fused_offsets,
      kblock_route_fused_counts,
      kblock_route_fused_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_tile_output_swizzle_offsets_h,
    nb::handle route_tile_output_swizzle_counts_h,
    nb::handle route_tile_output_swizzle_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_tile_output_swizzle_offsets = array_from_handle(
      route_tile_output_swizzle_offsets_h,
      "route_tile_output_swizzle_offsets");
  auto& route_tile_output_swizzle_counts = array_from_handle(
      route_tile_output_swizzle_counts_h,
      "route_tile_output_swizzle_counts");
  auto& route_tile_output_swizzle_route_slot_ids = array_from_handle(
      route_tile_output_swizzle_route_slot_ids_h,
      "route_tile_output_swizzle_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      route_tile_output_swizzle_offsets,
      route_tile_output_swizzle_counts,
      route_tile_output_swizzle_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_topk_offsets_h,
    nb::handle token_topk_counts_h,
    nb::handle token_topk_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_topk_offsets =
      array_from_handle(token_topk_offsets_h, "token_topk_offsets");
  auto& token_topk_counts =
      array_from_handle(token_topk_counts_h, "token_topk_counts");
  auto& token_topk_route_slot_ids = array_from_handle(
      token_topk_route_slot_ids_h,
      "token_topk_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_topk_output_tile_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_topk_offsets,
      token_topk_counts,
      token_topk_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_block_output_group_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_block_offsets_h,
    nb::handle token_block_counts_h,
    nb::handle token_block_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_block_offsets =
      array_from_handle(token_block_offsets_h, "token_block_offsets");
  auto& token_block_counts =
      array_from_handle(token_block_counts_h, "token_block_counts");
  auto& token_block_route_slot_ids = array_from_handle(
      token_block_route_slot_ids_h,
      "token_block_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_block_output_group_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_block_offsets,
      token_block_counts,
      token_block_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_output_stripe_offsets_h,
    nb::handle token_output_stripe_counts_h,
    nb::handle token_output_stripe_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_output_stripe_offsets = array_from_handle(
      token_output_stripe_offsets_h,
      "token_output_stripe_offsets");
  auto& token_output_stripe_counts = array_from_handle(
      token_output_stripe_counts_h,
      "token_output_stripe_counts");
  auto& token_output_stripe_route_slot_ids = array_from_handle(
      token_output_stripe_route_slot_ids_h,
      "token_output_stripe_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_output_stripe_group_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_output_stripe_offsets,
      token_output_stripe_counts,
      token_output_stripe_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_expert_output_block_offsets_h,
    nb::handle token_expert_output_block_counts_h,
    nb::handle token_expert_output_block_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_expert_output_block_offsets = array_from_handle(
      token_expert_output_block_offsets_h,
      "token_expert_output_block_offsets");
  auto& token_expert_output_block_counts = array_from_handle(
      token_expert_output_block_counts_h,
      "token_expert_output_block_counts");
  auto& token_expert_output_block_route_slot_ids = array_from_handle(
      token_expert_output_block_route_slot_ids_h,
      "token_expert_output_block_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_expert_output_block_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_expert_output_block_offsets,
      token_expert_output_block_counts,
      token_expert_output_block_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_pair_kblock_offsets_h,
    nb::handle token_pair_kblock_counts_h,
    nb::handle token_pair_kblock_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_pair_kblock_offsets = array_from_handle(
      token_pair_kblock_offsets_h,
      "token_pair_kblock_offsets");
  auto& token_pair_kblock_counts = array_from_handle(
      token_pair_kblock_counts_h,
      "token_pair_kblock_counts");
  auto& token_pair_kblock_route_slot_ids = array_from_handle(
      token_pair_kblock_route_slot_ids_h,
      "token_pair_kblock_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_pair_kblock_offsets,
      token_pair_kblock_counts,
      token_pair_kblock_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_pair_output_group_offsets_h,
    nb::handle token_pair_output_group_counts_h,
    nb::handle token_pair_output_group_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_pair_output_group_offsets = array_from_handle(
      token_pair_output_group_offsets_h,
      "token_pair_output_group_offsets");
  auto& token_pair_output_group_counts = array_from_handle(
      token_pair_output_group_counts_h,
      "token_pair_output_group_counts");
  auto& token_pair_output_group_route_slot_ids = array_from_handle(
      token_pair_output_group_route_slot_ids_h,
      "token_pair_output_group_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_pair_output_group_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_pair_output_group_offsets,
      token_pair_output_group_counts,
      token_pair_output_group_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_pair_slot_topk_output_group_offsets_h,
    nb::handle token_pair_slot_topk_output_group_counts_h,
    nb::handle token_pair_slot_topk_output_group_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_pair_slot_topk_output_group_offsets = array_from_handle(
      token_pair_slot_topk_output_group_offsets_h,
      "token_pair_slot_topk_output_group_offsets");
  auto& token_pair_slot_topk_output_group_counts = array_from_handle(
      token_pair_slot_topk_output_group_counts_h,
      "token_pair_slot_topk_output_group_counts");
  auto& token_pair_slot_topk_output_group_route_slot_ids = array_from_handle(
      token_pair_slot_topk_output_group_route_slot_ids_h,
      "token_pair_slot_topk_output_group_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul(
      sorted_x,
      codeword_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      token_pair_slot_topk_output_group_offsets,
      token_pair_slot_topk_output_group_counts,
      token_pair_slot_topk_output_group_route_slot_ids,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle token_pair_slot_topk_codeword_group_pipeline_offsets_h,
    nb::handle token_pair_slot_topk_codeword_group_pipeline_counts_h,
    nb::handle token_pair_slot_topk_codeword_group_pipeline_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& token_pair_slot_topk_codeword_group_pipeline_offsets =
      array_from_handle(
          token_pair_slot_topk_codeword_group_pipeline_offsets_h,
          "token_pair_slot_topk_codeword_group_pipeline_offsets");
  auto& token_pair_slot_topk_codeword_group_pipeline_counts =
      array_from_handle(
          token_pair_slot_topk_codeword_group_pipeline_counts_h,
          "token_pair_slot_topk_codeword_group_pipeline_counts");
  auto& token_pair_slot_topk_codeword_group_pipeline_route_slot_ids =
      array_from_handle(
          token_pair_slot_topk_codeword_group_pipeline_route_slot_ids_h,
          "token_pair_slot_topk_codeword_group_pipeline_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::
      e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul(
          sorted_x,
          codeword_tiles,
          scale_tiles,
          scale_group_indices,
          codeword_scale_slots,
          codebook,
          tile_experts,
          tile_offsets,
          tile_counts,
          token_pair_slot_topk_codeword_group_pipeline_offsets,
          token_pair_slot_topk_codeword_group_pipeline_counts,
          token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
          output_dims,
          kernel_dir,
          {});
}

void nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle scale_slot_broadcast_offsets_h,
    nb::handle scale_slot_broadcast_counts_h,
    nb::handle scale_slot_broadcast_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& scale_slot_broadcast_offsets =
      array_from_handle(
          scale_slot_broadcast_offsets_h,
          "scale_slot_broadcast_offsets");
  auto& scale_slot_broadcast_counts =
      array_from_handle(
          scale_slot_broadcast_counts_h,
          "scale_slot_broadcast_counts");
  auto& scale_slot_broadcast_route_slot_ids =
      array_from_handle(
          scale_slot_broadcast_route_slot_ids_h,
          "scale_slot_broadcast_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::
      e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul(
          sorted_x,
          codeword_tiles,
          scale_tiles,
          scale_group_indices,
          codeword_scale_slots,
          codebook,
          tile_experts,
          tile_offsets,
          tile_counts,
          scale_slot_broadcast_offsets,
          scale_slot_broadcast_counts,
          scale_slot_broadcast_route_slot_ids,
          output_dims,
          kernel_dir,
          {});
}

void nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle route_bucket_offsets_h,
    nb::handle route_bucket_counts_h,
    nb::handle route_bucket_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& route_bucket_offsets =
      array_from_handle(route_bucket_offsets_h, "route_bucket_offsets");
  auto& route_bucket_counts =
      array_from_handle(route_bucket_counts_h, "route_bucket_counts");
  auto& route_bucket_route_slot_ids = array_from_handle(
      route_bucket_route_slot_ids_h,
      "route_bucket_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::
      e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul(
          sorted_x,
          codeword_tiles,
          scale_tiles,
          scale_group_indices,
          codeword_scale_slots,
          codebook,
          tile_experts,
          tile_offsets,
          tile_counts,
          route_bucket_offsets,
          route_bucket_counts,
          route_bucket_route_slot_ids,
          output_dims,
          kernel_dir,
          {});
}

void nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle kblock_microtile_offsets_h,
    nb::handle kblock_microtile_counts_h,
    nb::handle kblock_microtile_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& kblock_microtile_offsets =
      array_from_handle(kblock_microtile_offsets_h, "kblock_microtile_offsets");
  auto& kblock_microtile_counts =
      array_from_handle(kblock_microtile_counts_h, "kblock_microtile_counts");
  auto& kblock_microtile_route_slot_ids = array_from_handle(
      kblock_microtile_route_slot_ids_h,
      "kblock_microtile_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::
      e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul(
          sorted_x,
          codeword_tiles,
          scale_tiles,
          scale_group_indices,
          codeword_scale_slots,
          codebook,
          tile_experts,
          tile_offsets,
          tile_counts,
          kblock_microtile_offsets,
          kblock_microtile_counts,
          kblock_microtile_route_slot_ids,
          output_dims,
          kernel_dir,
          {});
}

void nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codeword_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    nb::handle output_tile_fused_offsets_h,
    nb::handle output_tile_fused_counts_h,
    nb::handle output_tile_fused_route_slot_ids_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codeword_tiles =
      array_from_handle(codeword_tiles_h, "codeword_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& output_tile_fused_offsets = array_from_handle(
      output_tile_fused_offsets_h,
      "output_tile_fused_offsets");
  auto& output_tile_fused_counts = array_from_handle(
      output_tile_fused_counts_h,
      "output_tile_fused_counts");
  auto& output_tile_fused_route_slot_ids = array_from_handle(
      output_tile_fused_route_slot_ids_h,
      "output_tile_fused_route_slot_ids");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::
      e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul(
          sorted_x,
          codeword_tiles,
          scale_tiles,
          scale_group_indices,
          codeword_scale_slots,
          codebook,
          tile_experts,
          tile_offsets,
          tile_counts,
          output_tile_fused_offsets,
          output_tile_fused_counts,
          output_tile_fused_route_slot_ids,
          output_dims,
          kernel_dir,
          {});
}

void nax_e8p_component_stream_rhs_sorted_partial_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_component_bits_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle component_scale_slots_h,
    nb::handle component_codeword_indices_h,
    nb::handle component_offsets_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_component_bits =
      array_from_handle(sign_component_bits_h, "sign_component_bits");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& component_scale_slots =
      array_from_handle(component_scale_slots_h, "component_scale_slots");
  auto& component_codeword_indices =
      array_from_handle(component_codeword_indices_h, "component_codeword_indices");
  auto& component_offsets =
      array_from_handle(component_offsets_h, "component_offsets");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_component_stream_rhs_sorted_partial_matmul(
      sorted_x,
      sign_component_bits,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      component_scale_slots,
      component_codeword_indices,
      component_offsets,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_component_stream_rhs_sorted_tensorops_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_component_bits_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle component_scale_slots_h,
    nb::handle component_codeword_indices_h,
    nb::handle component_offsets_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_component_bits =
      array_from_handle(sign_component_bits_h, "sign_component_bits");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& component_scale_slots =
      array_from_handle(component_scale_slots_h, "component_scale_slots");
  auto& component_codeword_indices =
      array_from_handle(component_codeword_indices_h, "component_codeword_indices");
  auto& component_offsets =
      array_from_handle(component_offsets_h, "component_offsets");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_component_stream_rhs_sorted_tensorops_matmul(
      sorted_x,
      sign_component_bits,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      component_scale_slots,
      component_codeword_indices,
      component_offsets,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_component_stream_rhs_sorted_shared_decode_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_component_bits_h,
    nb::handle abs_index_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle component_scale_slots_h,
    nb::handle component_codeword_indices_h,
    nb::handle component_offsets_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_component_bits =
      array_from_handle(sign_component_bits_h, "sign_component_bits");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& component_scale_slots =
      array_from_handle(component_scale_slots_h, "component_scale_slots");
  auto& component_codeword_indices =
      array_from_handle(component_codeword_indices_h, "component_codeword_indices");
  auto& component_offsets =
      array_from_handle(component_offsets_h, "component_offsets");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_component_stream_rhs_sorted_shared_decode_matmul(
      sorted_x,
      sign_component_bits,
      abs_index_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      component_scale_slots,
      component_codeword_indices,
      component_offsets,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_packed_rhs_sorted_tiled_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_packed_rhs_sorted_tiled_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_rhs_sorted_tiled_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_tiles_h,
    nb::handle abs_index_tiles_h,
    nb::handle parity_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_tiles = array_from_handle(sign_tiles_h, "sign_tiles");
  auto& abs_index_tiles = array_from_handle(abs_index_tiles_h, "abs_index_tiles");
  auto& parity_tiles = array_from_handle(parity_tiles_h, "parity_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_rhs_sorted_tiled_matmul(
      sorted_x,
      sign_tiles,
      abs_index_tiles,
      parity_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul_into(
    nb::handle sorted_x_h,
    nb::handle sign_byte_lut_h,
    nb::handle sign_byte_slots_h,
    nb::handle abs_index_lut_h,
    nb::handle abs_index_slots_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& sign_byte_lut = array_from_handle(sign_byte_lut_h, "sign_byte_lut");
  auto& sign_byte_slots =
      array_from_handle(sign_byte_slots_h, "sign_byte_slots");
  auto& abs_index_lut = array_from_handle(abs_index_lut_h, "abs_index_lut");
  auto& abs_index_slots =
      array_from_handle(abs_index_slots_h, "abs_index_slots");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
      sorted_x,
      sign_byte_lut,
      sign_byte_slots,
      abs_index_lut,
      abs_index_slots,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_packed_rhs_sorted_tiled_m128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_packed_rhs_sorted_tiled_m128_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_packed_rhs_sorted_tiled_k128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle code_tiles_h,
    nb::handle scale_tiles_h,
    nb::handle scale_group_indices_h,
    nb::handle codeword_scale_slots_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int output_dims,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& code_tiles = array_from_handle(code_tiles_h, "code_tiles");
  auto& scale_tiles = array_from_handle(scale_tiles_h, "scale_tiles");
  auto& scale_group_indices =
      array_from_handle(scale_group_indices_h, "scale_group_indices");
  auto& codeword_scale_slots =
      array_from_handle(codeword_scale_slots_h, "codeword_scale_slots");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_packed_rhs_sorted_tiled_k128_matmul(
      sorted_x,
      code_tiles,
      scale_tiles,
      scale_group_indices,
      codeword_scale_slots,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      output_dims,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_direct_reduce_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_direct_reduce_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_inline_b_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_inline_b_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_gs352_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_gs352_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_lut_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle full_grid_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& full_grid = array_from_handle(full_grid_h, "full_grid");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_lut_matmul(
      sorted_x,
      codes,
      scales,
      full_grid,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_tgcb_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_tgcb_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
	      {});
}

void nax_e8p_fp16_sorted_steel_tgscale_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_tgscale_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_tgcb_hoist_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_bk128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_bk128_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m128n32_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m128n32_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m64n128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m64n128_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m32n64_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m32n64_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m64n64t64_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m64n64t64_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m32n64t128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m32n64t128_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8p_fp16_sorted_steel_m32n128_matmul_into(
    nb::handle sorted_x_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& sorted_x = array_from_handle(sorted_x_h, "sorted_x");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8p_fp16_sorted_steel_m32n128_matmul(
      sorted_x,
      codes,
      scales,
      codebook,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

void nax_e8_int8_routed_matmul_into(
    nb::handle x_q_h,
    nb::handle x_scales_h,
    nb::handle codes_h,
    nb::handle scales_h,
    nb::handle codebook_h,
    nb::handle lhs_indices_h,
    nb::handle tile_experts_h,
    nb::handle tile_offsets_h,
    nb::handle tile_counts_h,
    int group_size,
    const std::string& kernel_dir,
    nb::handle out_h) {
  auto& x_q = array_from_handle(x_q_h, "x_q");
  auto& x_scales = array_from_handle(x_scales_h, "x_scales");
  auto& codes = array_from_handle(codes_h, "codes");
  auto& scales = array_from_handle(scales_h, "scales");
  auto& codebook = array_from_handle(codebook_h, "codebook");
  auto& lhs_indices = array_from_handle(lhs_indices_h, "lhs_indices");
  auto& tile_experts = array_from_handle(tile_experts_h, "tile_experts");
  auto& tile_offsets = array_from_handle(tile_offsets_h, "tile_offsets");
  auto& tile_counts = array_from_handle(tile_counts_h, "tile_counts");
  auto& out = array_from_handle(out_h, "out");
  out = vqnax::e8_int8_routed_matmul(
      x_q,
      x_scales,
      codes,
      scales,
      codebook,
      lhs_indices,
      tile_experts,
      tile_offsets,
      tile_counts,
      group_size,
      kernel_dir,
      {});
}

std::string build_info() {
  return "vq_nax_ext scaffold linked against MLX " + std::string(mx::version());
}

NB_MODULE(_vqnax, m) {
  m.doc() = "GLM VQ NAX native extension scaffold";
  m.def("build_info", &build_info);
  m.def("is_available", []() { return true; });
  m.def(
      "predecoded_fp16_matmul_into",
      &predecoded_fp16_matmul_into,
      "x"_a,
      "weight_t"_a,
      "out"_a,
      "Build-smoke FP16 matmul path for the future predecoded NAX primitive.");
  m.def(
      "predecoded_fp16_gather_mm_into",
      &predecoded_fp16_gather_mm_into,
      "x"_a,
      "weight_t"_a,
      "rhs_indices"_a,
      "out"_a,
      "Build-smoke FP16 gather_mm path for the future predecoded NAX primitive.");
  m.def(
      "nax_fp16_matmul_tile_into",
      &nax_fp16_matmul_tile_into,
      "x"_a,
      "weight_t"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps FP16 matmul smoke tile.");
  m.def(
      "nax_e8_fp16_matmul_tile_into",
      &nax_e8_fp16_matmul_tile_into,
      "x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 FP16 smoke tile.");
  m.def(
      "nax_e8_fp16_matmul_into",
      &nax_e8_fp16_matmul_into,
      "x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 FP16 dense matmul.");
  m.def(
      "nax_e8_fp16_routed_matmul_into",
      &nax_e8_fp16_routed_matmul_into,
      "x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "lhs_indices"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 FP16 sorted-routed matmul.");
  m.def(
      "nax_e8_fp16_routed_steel_matmul_into",
      &nax_e8_fp16_routed_steel_matmul_into,
      "x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "lhs_indices"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 FP16 sorted-routed Steel-tile probe.");
  m.def(
      "nax_e8_fp16_sorted_steel_matmul_into",
      &nax_e8_fp16_sorted_steel_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 FP16 contiguous sorted-A Steel probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_matmul_into",
      &nax_e8p_fp16_sorted_steel_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel probe.");
  m.def(
      "nax_e8p_packed_rhs_tile_matmul_into",
      &nax_e8p_packed_rhs_tile_matmul_into,
      "x"_a,
      "code_tile"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 packed RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_split_byte_rhs_tile_matmul_into",
      &nax_e8p_split_byte_rhs_tile_matmul_into,
      "x"_a,
      "sign_tile"_a,
      "abs_index_tile"_a,
      "parity_tile"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_sign_nibble_abs_index_rhs_tile_matmul_into",
      &nax_e8p_sign_nibble_abs_index_rhs_tile_matmul_into,
      "x"_a,
      "sign_low_nibble_tile"_a,
      "sign_high_nibble_tile"_a,
      "abs_index_tile"_a,
      "parity_tile"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_sign_plane_abs_index_rhs_tile_matmul_into",
      &nax_e8p_sign_plane_abs_index_rhs_tile_matmul_into,
      "x"_a,
      "sign_bit_planes"_a,
      "abs_index_tile"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-plane RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul_into",
      &nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul_into,
      "x"_a,
      "sign_low_nibble_lut"_a,
      "sign_low_nibble_slots"_a,
      "sign_high_nibble_lut"_a,
      "sign_high_nibble_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble micro-LUT RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_split_byte_factor_reuse_rhs_tile_matmul_into",
      &nax_e8p_split_byte_factor_reuse_rhs_tile_matmul_into,
      "x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tile"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "output_count"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte factor-reuse RHS tile matmul correctness probe.");
  m.def(
      "nax_e8p_packed_rhs_sorted_matmul_into",
      &nax_e8p_packed_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 packed RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_split_byte_rhs_sorted_matmul_into",
      &nax_e8p_split_byte_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_tiles"_a,
      "abs_index_tiles"_a,
      "parity_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul_into",
      &nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_low_nibble_tiles"_a,
      "sign_high_nibble_tiles"_a,
      "abs_index_tiles"_a,
      "parity_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_sign_plane_abs_index_rhs_sorted_matmul_into",
      &nax_e8p_sign_plane_abs_index_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_bit_planes"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-plane RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul_into",
      &nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_low_nibble_lut"_a,
      "sign_low_nibble_slots"_a,
      "sign_high_nibble_lut"_a,
      "sign_high_nibble_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble micro-LUT RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul_into",
      &nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul_into,
      "sorted_x"_a,
      "sign_low_nibble_tiles"_a,
      "sign_high_nibble_tiles"_a,
      "abs_index_tiles"_a,
      "parity_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble RHS sorted-route TensorOps probe.");
  m.def(
      "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul_into",
      &nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul_into,
      "sorted_x"_a,
      "sign_bit_planes"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-plane RHS sorted-route TensorOps probe.");
  m.def(
      "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul_into",
      &nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul_into,
      "sorted_x"_a,
      "sign_low_nibble_lut"_a,
      "sign_low_nibble_slots"_a,
      "sign_high_nibble_lut"_a,
      "sign_high_nibble_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 sign-nibble micro-LUT RHS sorted-route TensorOps probe.");
  m.def(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul_into",
      &nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte factor-reuse RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul_into",
      &nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert/K-block factor-reuse RHS sorted-route matmul correctness probe.");
  m.def(
      "nax_e8p_component_stream_rhs_sorted_scalar_matmul_into",
      &nax_e8p_component_stream_rhs_sorted_scalar_matmul_into,
      "sorted_x"_a,
      "sign_component_bits"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "component_scale_slots"_a,
      "component_codeword_indices"_a,
      "component_offsets"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 component-stream RHS sorted-route scalar oracle.");
  m.def(
      "nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul_into",
      &nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-slot codeword-stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul_into",
      &nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-slot MMA codeword-tile RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul_into",
      &nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "active_route_tiles"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 active-route-tile codeword outer-product RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul_into",
      &nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "expert_cohort_offsets"_a,
      "expert_cohort_counts"_a,
      "route_cohort_offsets"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert-cohort codeword-broadcast RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul_into",
      &nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_batch_segment_offsets"_a,
      "route_batch_segment_counts"_a,
      "route_batch_route_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-batch segmented codeword-reduce RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_cohort_offsets"_a,
      "token_cohort_counts"_a,
      "token_cohort_active_expert_ids"_a,
      "token_cohort_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-cohort codeword-stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul_into",
      &nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_cohort_offsets"_a,
      "token_cohort_counts"_a,
      "token_cohort_active_expert_ids"_a,
      "token_cohort_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-cohort MMA codeword-tile RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul_into",
      &nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_stationary_route_batch_offsets"_a,
      "output_stationary_route_batch_counts"_a,
      "output_stationary_route_batch_active_expert_ids"_a,
      "output_stationary_route_batch_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 output-stationary codeword-tile RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul_into",
      &nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "input_stationary_route_batch_offsets"_a,
      "input_stationary_route_batch_counts"_a,
      "input_stationary_route_batch_active_expert_ids"_a,
      "input_stationary_route_batch_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 input-stationary codeword-tile RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul_into",
      &nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_factor_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert/K-block codeword factor-reuse RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul_into",
      &nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul_into,
      "route_local_codeword_dot_lut"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_codeword_lut_route_slots"_a,
      "route_codeword_lut_offsets"_a,
      "route_codeword_lut_counts"_a,
      "route_codeword_lut_codeword_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-codeword LUT accumulate RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul_into",
      &nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "rowwise_route_microtile_offsets"_a,
      "rowwise_route_microtile_counts"_a,
      "rowwise_route_microtile_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 rowwise codeword-tile accumulate RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul_into",
      &nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_tile_local_route_microtile_offsets"_a,
      "output_tile_local_route_microtile_counts"_a,
      "output_tile_local_route_microtile_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 output-tile-local codeword LUT RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul_into",
      &nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_microtile_codeword_block_reduce_offsets"_a,
      "route_microtile_codeword_block_reduce_counts"_a,
      "route_microtile_codeword_block_reduce_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-microtile codeword-block reduce RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul_into",
      &nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "kblock_wavefront_codeword_scan_offsets"_a,
      "kblock_wavefront_codeword_scan_counts"_a,
      "kblock_wavefront_codeword_scan_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 k-block wavefront codeword-scan RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul_into",
      &nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_route_output_stripe_offsets"_a,
      "token_route_output_stripe_counts"_a,
      "token_route_output_stripe_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-route output-stripe pipeline RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul_into",
      &nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert/K-block scale-slot stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul_into",
      &nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "scale_group_route_block_offsets"_a,
      "scale_group_route_block_counts"_a,
      "scale_group_route_block_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 scale-group route-block reduce RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_block_output_group_stream_rhs_sorted_matmul_into",
      &nax_e8p_route_block_output_group_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_block_output_group_offsets"_a,
      "route_block_output_group_counts"_a,
      "route_block_output_group_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-block output-group stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul_into",
      &nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_group_pretransposed_route_offsets"_a,
      "output_group_pretransposed_route_counts"_a,
      "output_group_pretransposed_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 output-group-pretransposed codeword-stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul_into",
      &nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "kblock_route_fused_offsets"_a,
      "kblock_route_fused_counts"_a,
      "kblock_route_fused_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 K-block route-fused RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul_into",
      &nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_tile_output_swizzle_offsets"_a,
      "route_tile_output_swizzle_counts"_a,
      "route_tile_output_swizzle_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 route-tile output-swizzle RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_topk_offsets"_a,
      "token_topk_counts"_a,
      "token_topk_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token top-k output-tile RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_block_output_group_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_block_output_group_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_block_offsets"_a,
      "token_block_counts"_a,
      "token_block_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-block output-group RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_output_stripe_offsets"_a,
      "token_output_stripe_counts"_a,
      "token_output_stripe_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token output-stripe group RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_expert_output_block_offsets"_a,
      "token_expert_output_block_counts"_a,
      "token_expert_output_block_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-expert output-block RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_pair_kblock_offsets"_a,
      "token_pair_kblock_counts"_a,
      "token_pair_kblock_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair K-block accumulator RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_pair_output_group_offsets"_a,
      "token_pair_output_group_counts"_a,
      "token_pair_output_group_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair output-group RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_pair_slot_topk_output_group_offsets"_a,
      "token_pair_slot_topk_output_group_counts"_a,
      "token_pair_slot_topk_output_group_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k output-group RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "token_pair_slot_topk_codeword_group_pipeline_offsets"_a,
      "token_pair_slot_topk_codeword_group_pipeline_counts"_a,
      "token_pair_slot_topk_codeword_group_pipeline_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k codeword-group pipeline RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "scale_slot_broadcast_offsets"_a,
      "scale_slot_broadcast_counts"_a,
      "scale_slot_broadcast_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k scale-slot broadcast stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "route_bucket_offsets"_a,
      "route_bucket_counts"_a,
      "route_bucket_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k route-bucket codeword-reduce RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "kblock_microtile_offsets"_a,
      "kblock_microtile_counts"_a,
      "kblock_microtile_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k K-block microtile stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul_into",
      &nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul_into,
      "sorted_x"_a,
      "codeword_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_tile_fused_offsets"_a,
      "output_tile_fused_counts"_a,
      "output_tile_fused_route_slot_ids"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 token-pair slot/top-k output-tile fused stream RHS sorted-route source guardrail.");
  m.def(
      "nax_e8p_component_stream_rhs_sorted_partial_matmul_into",
      &nax_e8p_component_stream_rhs_sorted_partial_matmul_into,
      "sorted_x"_a,
      "sign_component_bits"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "component_scale_slots"_a,
      "component_codeword_indices"_a,
      "component_offsets"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 component-stream RHS sorted-route partial-reduction scaffold.");
  m.def(
      "nax_e8p_component_stream_rhs_sorted_tensorops_matmul_into",
      &nax_e8p_component_stream_rhs_sorted_tensorops_matmul_into,
      "sorted_x"_a,
      "sign_component_bits"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "component_scale_slots"_a,
      "component_codeword_indices"_a,
      "component_offsets"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 component-stream RHS sorted-route TensorOps parity candidate.");
  m.def(
      "nax_e8p_component_stream_rhs_sorted_shared_decode_matmul_into",
      &nax_e8p_component_stream_rhs_sorted_shared_decode_matmul_into,
      "sorted_x"_a,
      "sign_component_bits"_a,
      "abs_index_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "component_scale_slots"_a,
      "component_codeword_indices"_a,
      "component_offsets"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 component-stream RHS sorted-route shared-decode parity scaffold.");
  m.def(
      "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul_into",
      &nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert/K-block factor-reuse RHS sorted-route TensorOps probe.");
  m.def(
      "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul_into",
      &nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 expert/K-block factor-reuse RHS sorted-route v2 contract probe.");
  m.def(
      "nax_e8p_packed_rhs_sorted_tiled_matmul_into",
      &nax_e8p_packed_rhs_sorted_tiled_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 packed RHS sorted-route tiled TensorOps probe.");
  m.def(
      "nax_e8p_split_byte_rhs_sorted_tiled_matmul_into",
      &nax_e8p_split_byte_rhs_sorted_tiled_matmul_into,
      "sorted_x"_a,
      "sign_tiles"_a,
      "abs_index_tiles"_a,
      "parity_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte RHS sorted-route tiled TensorOps probe.");
  m.def(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul_into",
      &nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte factor-reuse RHS sorted-route tiled TensorOps probe.");
  m.def(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul_into",
      &nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte factor-reuse RHS sorted-route shared-decode TensorOps probe.");
  m.def(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul_into",
      &nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul_into,
      "sorted_x"_a,
      "sign_byte_lut"_a,
      "sign_byte_slots"_a,
      "abs_index_lut"_a,
      "abs_index_slots"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 split-byte factor-reuse RHS sorted-route shared-n-decode TensorOps probe.");
  m.def(
      "nax_e8p_packed_rhs_sorted_tiled_m128_matmul_into",
      &nax_e8p_packed_rhs_sorted_tiled_m128_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 packed RHS sorted-route M128 tiled TensorOps probe.");
  m.def(
      "nax_e8p_packed_rhs_sorted_tiled_k128_matmul_into",
      &nax_e8p_packed_rhs_sorted_tiled_k128_matmul_into,
      "sorted_x"_a,
      "code_tiles"_a,
      "scale_tiles"_a,
      "scale_group_indices"_a,
      "codeword_scale_slots"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "output_dims"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled E8P FP16 packed RHS sorted-route K128 tiled TensorOps probe.");
  m.def(
      "nax_e8p_fp16_sorted_direct_reduce_matmul_into",
      &nax_e8p_fp16_sorted_direct_reduce_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled fused E8P FP16 contiguous sorted-A direct-reduce probe with no decoded-B threadgroup staging.");
  m.def(
      "nax_e8p_fp16_sorted_inline_b_matmul_into",
      &nax_e8p_fp16_sorted_inline_b_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A inline-B probe with no decoded-B threadgroup staging.");
  m.def(
      "nax_e8p_fp16_sorted_steel_gs352_matmul_into",
      &nax_e8p_fp16_sorted_steel_gs352_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel Air down group-size-352 probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_lut_matmul_into",
      &nax_e8p_fp16_sorted_steel_lut_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "full_grid"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel full-grid LUT probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_tgcb_matmul_into",
      &nax_e8p_fp16_sorted_steel_tgcb_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel threadgroup-codebook probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_tgscale_matmul_into",
      &nax_e8p_fp16_sorted_steel_tgscale_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel threadgroup-scale probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul_into",
      &nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel threadgroup-codebook plus threadgroup-scale probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul_into",
      &nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel threadgroup-codebook hoist probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_bk128_matmul_into",
      &nax_e8p_fp16_sorted_steel_bk128_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel BK128 probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m128n32_matmul_into",
      &nax_e8p_fp16_sorted_steel_m128n32_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M128N32 probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m64n128_matmul_into",
      &nax_e8p_fp16_sorted_steel_m64n128_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M64N128 probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m32n64_matmul_into",
      &nax_e8p_fp16_sorted_steel_m32n64_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M32N64 probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m64n64t64_matmul_into",
      &nax_e8p_fp16_sorted_steel_m64n64t64_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M64N64 two-compute-simdgroup probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m32n64t128_matmul_into",
      &nax_e8p_fp16_sorted_steel_m32n64t128_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M32N64 128-thread staging probe.");
  m.def(
      "nax_e8p_fp16_sorted_steel_m32n128_matmul_into",
      &nax_e8p_fp16_sorted_steel_m32n128_matmul_into,
      "sorted_x"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8P FP16 contiguous sorted-A Steel M32N128 probe.");
  m.def(
      "nax_e8_int8_routed_matmul_into",
      &nax_e8_int8_routed_matmul_into,
      "x_q"_a,
      "x_scales"_a,
      "codes"_a,
      "scales"_a,
      "codebook"_a,
      "lhs_indices"_a,
      "tile_experts"_a,
      "tile_offsets"_a,
      "tile_counts"_a,
      "group_size"_a,
      "kernel_dir"_a,
      "out"_a,
      "Runtime-compiled Metal 4 TensorOps fused E8 INT8 sorted-routed matmul.");
}
