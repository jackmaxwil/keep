#include "nax_fp16_primitive.h"

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vqnax {

using namespace mlx::core;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open Metal source: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::optional<std::string> quoted_include(const std::string& line) {
  auto include_pos = line.find("#include");
  if (include_pos == std::string::npos) {
    return std::nullopt;
  }
  auto first_quote = line.find('"', include_pos);
  if (first_quote == std::string::npos) {
    return std::nullopt;
  }
  auto second_quote = line.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return std::nullopt;
  }
  return line.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::filesystem::path resolve_include(
    const std::string& include,
    const std::filesystem::path& source_dir) {
  if (include.rfind("mlx/", 0) == 0) {
    std::filesystem::path mlx_root(VQNAX_MLX_INCLUDE_DIR);
    return mlx_root / include;
  }
  return source_dir / include;
}

std::string expand_metal_includes(
    const std::string& source,
    const std::filesystem::path& source_dir,
    std::unordered_set<std::string>& included) {
  std::stringstream input(source);
  std::stringstream output;
  std::string line;
  while (std::getline(input, line)) {
    auto include = quoted_include(line);
    if (!include.has_value()) {
      output << line << '\n';
      continue;
    }
    auto path = resolve_include(*include, source_dir);
    auto path_string = path.string();
    if (!included.insert(path_string).second) {
      output << '\n';
      continue;
    }
    output << "\n// begin expanded include " << *include << "\n";
    output << expand_metal_includes(
        read_file(path_string), path.parent_path(), included);
    output << "// end expanded include " << *include << "\n";
  }
  return output.str();
}

class PipelineCache {
 public:
  static PipelineCache& instance() {
    static PipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string& kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir) {
      return;
    }
    auto& dev = metal::device(mlx::core::Device::gpu);
    library_ = compile_source(dev.mtl_device(), kernel_dir + "/nax_fp16_matmul.metal");
    kernel_dir_ = kernel_dir;
    pipelines_.clear();
    initialized_ = true;
  }

  MTL::ComputePipelineState* get(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
      return it->second;
    }
    auto& dev = metal::device(mlx::core::Device::gpu);
    auto* pso = make_pipeline(dev.mtl_device(), name);
    pipelines_[name] = pso;
    return pso;
  }

 private:
  MTL::Library* compile_source(MTL::Device* mtl_device, const std::string& source_path) {
    std::unordered_set<std::string> included;
    std::filesystem::path path(source_path);
    std::string source =
        expand_metal_includes(read_file(source_path), path.parent_path(), included);
    @autoreleasepool {
      NSString* src = [NSString stringWithUTF8String:source.c_str()];
      MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
      opts.languageVersion = MTLLanguageVersion4_0;
      NSError* error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLLibrary> lib_objc = [device_objc newLibraryWithSource:src
                                                          options:opts
                                                            error:&error];
      if (!lib_objc) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Metal compile failed (" + source_path + "): " + err);
      }
      return (__bridge MTL::Library*)(void*)CFBridgingRetain(lib_objc);
    }
  }

  MTL::ComputePipelineState* make_pipeline(MTL::Device* mtl_device, const std::string& name) {
    @autoreleasepool {
      NSString* function_name = [NSString stringWithUTF8String:name.c_str()];
      id<MTLLibrary> library_objc = (__bridge id<MTLLibrary>)library_;
      id<MTLFunction> function = [library_objc newFunctionWithName:function_name];
      if (!function) {
        throw std::runtime_error("Kernel not found: " + name);
      }
      NSError* error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLComputePipelineState> pso =
          [device_objc newComputePipelineStateWithFunction:function error:&error];
      if (!pso) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Pipeline failed for " + name + ": " + err);
      }
      return (__bridge MTL::ComputePipelineState*)(void*)CFBridgingRetain(pso);
    }
  }

  bool initialized_ = false;
  std::string kernel_dir_;
  MTL::Library* library_ = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
  std::mutex mutex_;
};

}  // namespace

void NaxFp16MatmulTile::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& weight_t = inputs[1];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);
  auto* pso = cache.get("nax_fp16_matmul_tile");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weight_t, 1);
  encoder.set_output_array(out, 2);
  encoder.dispatch_threadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(32, 1, 1));
}

void NaxE8Fp16MatmulTile::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);
  auto* pso = cache.get("nax_e8_fp16_matmul_tile");

  uint32_t group_size = static_cast<uint32_t>(group_size_);
  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(group_size, 5);
  encoder.dispatch_threadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(32, 1, 1));
}

void NaxE8Fp16Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);
  auto* pso = cache.get("nax_e8_fp16_matmul");

  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_m = (M + 31u) / 32u;
  uint32_t tiles_n = (N + 15u) / 16u;

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(M, 5);
  encoder.set_bytes(N, 6);
  encoder.set_bytes(K, 7);
  encoder.set_bytes(group_size, 8);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, tiles_m, 1),
      MTL::Size::Make(32, 1, 1));
}

void NaxE8Fp16RoutedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& lhs_indices = inputs[4];
  auto& tile_experts = inputs[5];
  auto& tile_offsets = inputs[6];
  auto& tile_counts = inputs[7];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(lhs_indices.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8_fp16_routed_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(lhs_indices, 4);
  encoder.set_input_array(tile_experts, 5);
  encoder.set_input_array(tile_offsets, 6);
  encoder.set_input_array(tile_counts, 7);
  encoder.set_output_array(out, 8);
  encoder.set_bytes(route_count, 9);
  encoder.set_bytes(N, 10);
  encoder.set_bytes(K, 11);
  encoder.set_bytes(group_size, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8Fp16RoutedSteelMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& lhs_indices = inputs[4];
  auto& tile_experts = inputs[5];
  auto& tile_offsets = inputs[6];
  auto& tile_counts = inputs[7];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(lhs_indices.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8_fp16_routed_matmul_steel");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(lhs_indices, 4);
  encoder.set_input_array(tile_experts, 5);
  encoder.set_input_array(tile_offsets, 6);
  encoder.set_input_array(tile_counts, 7);
  encoder.set_output_array(out, 8);
  encoder.set_bytes(route_count, 9);
  encoder.set_bytes(N, 10);
  encoder.set_bytes(K, 11);
  encoder.set_bytes(group_size, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8Fp16SortedSteelMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8_fp16_sorted_matmul_steel");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PPackedRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& code_tile = inputs[1];
  auto& scale_tile = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(code_tile.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_packed_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(code_tile, 1);
  encoder.set_input_array(scale_tile, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_output_array(out, 6);
  encoder.set_bytes(route_count, 7);
  encoder.set_bytes(output_count, 8);
  encoder.set_bytes(K, 9);
  encoder.set_bytes(codewords, 10);
  encoder.set_bytes(scale_groups, 11);
  encoder.dispatch_threadgroups(
	      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
		      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSplitByteRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& sign_tile = inputs[1];
  auto& abs_index_tile = inputs[2];
  auto& parity_tile = inputs[3];
  auto& scale_tile = inputs[4];
  auto& scale_group_indices = inputs[5];
  auto& codeword_scale_slots = inputs[6];
  auto& codebook = inputs[7];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(sign_tile.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_split_byte_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(sign_tile, 1);
  encoder.set_input_array(abs_index_tile, 2);
  encoder.set_input_array(parity_tile, 3);
  encoder.set_input_array(scale_tile, 4);
  encoder.set_input_array(scale_group_indices, 5);
  encoder.set_input_array(codeword_scale_slots, 6);
  encoder.set_input_array(codebook, 7);
  encoder.set_output_array(out, 8);
  encoder.set_bytes(route_count, 9);
  encoder.set_bytes(output_count, 10);
  encoder.set_bytes(K, 11);
  encoder.set_bytes(codewords, 12);
  encoder.set_bytes(scale_groups, 13);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignNibbleAbsIndexRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& sign_low_nibble_tile = inputs[1];
  auto& sign_high_nibble_tile = inputs[2];
  auto& abs_index_tile = inputs[3];
  auto& parity_tile = inputs[4];
  auto& scale_tile = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_tile.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_sign_nibble_abs_index_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(sign_low_nibble_tile, 1);
  encoder.set_input_array(sign_high_nibble_tile, 2);
  encoder.set_input_array(abs_index_tile, 3);
  encoder.set_input_array(parity_tile, 4);
  encoder.set_input_array(scale_tile, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_count, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(codewords, 13);
  encoder.set_bytes(scale_groups, 14);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignPlaneAbsIndexRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& sign_bit_planes = inputs[1];
  auto& abs_index_tile = inputs[2];
  auto& scale_tile = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& codebook = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(sign_bit_planes.shape(0));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_sign_plane_abs_index_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(sign_bit_planes, 1);
  encoder.set_input_array(abs_index_tile, 2);
  encoder.set_input_array(scale_tile, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(codebook, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(output_count, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(codewords, 11);
  encoder.set_bytes(scale_groups, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignNibbleMicroLUTRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& sign_low_nibble_lut = inputs[1];
  auto& sign_low_nibble_slots = inputs[2];
  auto& sign_high_nibble_lut = inputs[3];
  auto& sign_high_nibble_slots = inputs[4];
  auto& abs_index_lut = inputs[5];
  auto& abs_index_slots = inputs[6];
  auto& scale_tile = inputs[7];
  auto& scale_group_indices = inputs[8];
  auto& codeword_scale_slots = inputs[9];
  auto& codebook = inputs[10];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_slots.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(sign_low_nibble_lut, 1);
  encoder.set_input_array(sign_low_nibble_slots, 2);
  encoder.set_input_array(sign_high_nibble_lut, 3);
  encoder.set_input_array(sign_high_nibble_slots, 4);
  encoder.set_input_array(abs_index_lut, 5);
  encoder.set_input_array(abs_index_slots, 6);
  encoder.set_input_array(scale_tile, 7);
  encoder.set_input_array(scale_group_indices, 8);
  encoder.set_input_array(codeword_scale_slots, 9);
  encoder.set_input_array(codebook, 10);
  encoder.set_output_array(out, 11);
  encoder.set_bytes(route_count, 12);
  encoder.set_bytes(output_count, 13);
  encoder.set_bytes(K, 14);
  encoder.set_bytes(codewords, 15);
  encoder.set_bytes(scale_groups, 16);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSplitByteFactorReuseRHSTileMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tile = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  uint32_t output_count = static_cast<uint32_t>(output_count_);
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tile.shape(1));
  auto* pso = cache.get("nax_e8p_split_byte_factor_reuse_rhs_tile_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tile, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_count, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(codewords, 13);
  encoder.set_bytes(scale_groups, 14);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_count + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PPackedRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_packed_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
	      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSplitByteRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_tiles = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& parity_tiles = inputs[3];
  auto& scale_tiles = inputs[4];
  auto& scale_group_indices = inputs[5];
  auto& codeword_scale_slots = inputs[6];
  auto& codebook = inputs[7];
  auto& tile_experts = inputs[8];
  auto& tile_offsets = inputs[9];
  auto& tile_counts = inputs[10];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_split_byte_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_tiles, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(parity_tiles, 3);
  encoder.set_input_array(scale_tiles, 4);
  encoder.set_input_array(scale_group_indices, 5);
  encoder.set_input_array(codeword_scale_slots, 6);
  encoder.set_input_array(codebook, 7);
  encoder.set_input_array(tile_experts, 8);
  encoder.set_input_array(tile_offsets, 9);
  encoder.set_input_array(tile_counts, 10);
  encoder.set_output_array(out, 11);
  encoder.set_bytes(route_count, 12);
  encoder.set_bytes(output_dims, 13);
  encoder.set_bytes(K, 14);
  encoder.set_bytes(experts, 15);
  encoder.set_bytes(n_tiles, 16);
  encoder.set_bytes(k_blocks, 17);
  encoder.set_bytes(bn, 18);
  encoder.set_bytes(codewords, 19);
  encoder.set_bytes(scale_groups, 20);
  encoder.set_bytes(num_route_tiles, 21);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignNibbleAbsIndexRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_low_nibble_tiles = inputs[1];
  auto& sign_high_nibble_tiles = inputs[2];
  auto& abs_index_tiles = inputs[3];
  auto& parity_tiles = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_low_nibble_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_low_nibble_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_low_nibble_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_low_nibble_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_low_nibble_tiles, 1);
  encoder.set_input_array(sign_high_nibble_tiles, 2);
  encoder.set_input_array(abs_index_tiles, 3);
  encoder.set_input_array(parity_tiles, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignPlaneAbsIndexRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_bit_planes = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& codebook = inputs[6];
  auto& tile_experts = inputs[7];
  auto& tile_offsets = inputs[8];
  auto& tile_counts = inputs[9];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_bit_planes.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_bit_planes.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_bit_planes.shape(2));
  uint32_t codewords = static_cast<uint32_t>(sign_bit_planes.shape(3));
  uint32_t bn = static_cast<uint32_t>(abs_index_tiles.shape(3));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_sign_plane_abs_index_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_bit_planes, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(codebook, 6);
  encoder.set_input_array(tile_experts, 7);
  encoder.set_input_array(tile_offsets, 8);
  encoder.set_input_array(tile_counts, 9);
  encoder.set_output_array(out, 10);
  encoder.set_bytes(route_count, 11);
  encoder.set_bytes(output_dims, 12);
  encoder.set_bytes(K, 13);
  encoder.set_bytes(experts, 14);
  encoder.set_bytes(n_tiles, 15);
  encoder.set_bytes(k_blocks, 16);
  encoder.set_bytes(bn, 17);
  encoder.set_bytes(codewords, 18);
  encoder.set_bytes(scale_groups, 19);
  encoder.set_bytes(num_route_tiles, 20);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignNibbleMicroLUTRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_low_nibble_lut = inputs[1];
  auto& sign_low_nibble_slots = inputs[2];
  auto& sign_high_nibble_lut = inputs[3];
  auto& sign_high_nibble_slots = inputs[4];
  auto& abs_index_lut = inputs[5];
  auto& abs_index_slots = inputs[6];
  auto& scale_tiles = inputs[7];
  auto& scale_group_indices = inputs[8];
  auto& codeword_scale_slots = inputs[9];
  auto& codebook = inputs[10];
  auto& tile_experts = inputs[11];
  auto& tile_offsets = inputs[12];
  auto& tile_counts = inputs[13];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_low_nibble_slots.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_low_nibble_slots.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_low_nibble_slots.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_low_nibble_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_low_nibble_lut, 1);
  encoder.set_input_array(sign_low_nibble_slots, 2);
  encoder.set_input_array(sign_high_nibble_lut, 3);
  encoder.set_input_array(sign_high_nibble_slots, 4);
  encoder.set_input_array(abs_index_lut, 5);
  encoder.set_input_array(abs_index_slots, 6);
  encoder.set_input_array(scale_tiles, 7);
  encoder.set_input_array(scale_group_indices, 8);
  encoder.set_input_array(codeword_scale_slots, 9);
  encoder.set_input_array(codebook, 10);
  encoder.set_input_array(tile_experts, 11);
  encoder.set_input_array(tile_offsets, 12);
  encoder.set_input_array(tile_counts, 13);
  encoder.set_output_array(out, 14);
  encoder.set_bytes(route_count, 15);
  encoder.set_bytes(output_dims, 16);
  encoder.set_bytes(K, 17);
  encoder.set_bytes(experts, 18);
  encoder.set_bytes(n_tiles, 19);
  encoder.set_bytes(k_blocks, 20);
  encoder.set_bytes(bn, 21);
  encoder.set_bytes(codewords, 22);
  encoder.set_bytes(scale_groups, 23);
  encoder.set_bytes(num_route_tiles, 24);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_low_nibble_tiles = inputs[1];
  auto& sign_high_nibble_tiles = inputs[2];
  auto& abs_index_tiles = inputs[3];
  auto& parity_tiles = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_low_nibble_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_low_nibble_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_low_nibble_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_low_nibble_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso =
      cache.get("nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_low_nibble_tiles, 1);
  encoder.set_input_array(sign_high_nibble_tiles, 2);
  encoder.set_input_array(abs_index_tiles, 3);
  encoder.set_input_array(parity_tiles, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_bit_planes = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& codebook = inputs[6];
  auto& tile_experts = inputs[7];
  auto& tile_offsets = inputs[8];
  auto& tile_counts = inputs[9];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_bit_planes.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_bit_planes.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_bit_planes.shape(2));
  uint32_t codewords = static_cast<uint32_t>(sign_bit_planes.shape(3));
  uint32_t bn = static_cast<uint32_t>(abs_index_tiles.shape(3));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso =
      cache.get("nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_bit_planes, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(codebook, 6);
  encoder.set_input_array(tile_experts, 7);
  encoder.set_input_array(tile_offsets, 8);
  encoder.set_input_array(tile_counts, 9);
  encoder.set_output_array(out, 10);
  encoder.set_bytes(route_count, 11);
  encoder.set_bytes(output_dims, 12);
  encoder.set_bytes(K, 13);
  encoder.set_bytes(experts, 14);
  encoder.set_bytes(n_tiles, 15);
  encoder.set_bytes(k_blocks, 16);
  encoder.set_bytes(bn, 17);
  encoder.set_bytes(codewords, 18);
  encoder.set_bytes(scale_groups, 19);
  encoder.set_bytes(num_route_tiles, 20);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_low_nibble_lut = inputs[1];
  auto& sign_low_nibble_slots = inputs[2];
  auto& sign_high_nibble_lut = inputs[3];
  auto& sign_high_nibble_slots = inputs[4];
  auto& abs_index_lut = inputs[5];
  auto& abs_index_slots = inputs[6];
  auto& scale_tiles = inputs[7];
  auto& scale_group_indices = inputs[8];
  auto& codeword_scale_slots = inputs[9];
  auto& codebook = inputs[10];
  auto& tile_experts = inputs[11];
  auto& tile_offsets = inputs[12];
  auto& tile_counts = inputs[13];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_low_nibble_slots.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_low_nibble_slots.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_low_nibble_slots.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_low_nibble_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_low_nibble_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso =
      cache.get("nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_low_nibble_lut, 1);
  encoder.set_input_array(sign_low_nibble_slots, 2);
  encoder.set_input_array(sign_high_nibble_lut, 3);
  encoder.set_input_array(sign_high_nibble_slots, 4);
  encoder.set_input_array(abs_index_lut, 5);
  encoder.set_input_array(abs_index_slots, 6);
  encoder.set_input_array(scale_tiles, 7);
  encoder.set_input_array(scale_group_indices, 8);
  encoder.set_input_array(codeword_scale_slots, 9);
  encoder.set_input_array(codebook, 10);
  encoder.set_input_array(tile_experts, 11);
  encoder.set_input_array(tile_offsets, 12);
  encoder.set_input_array(tile_counts, 13);
  encoder.set_output_array(out, 14);
  encoder.set_bytes(route_count, 15);
  encoder.set_bytes(output_dims, 16);
  encoder.set_bytes(K, 17);
  encoder.set_bytes(experts, 18);
  encoder.set_bytes(n_tiles, 19);
  encoder.set_bytes(k_blocks, 20);
  encoder.set_bytes(bn, 21);
  encoder.set_bytes(codewords, 22);
  encoder.set_bytes(scale_groups, 23);
  encoder.set_bytes(num_route_tiles, 24);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSplitByteFactorReuseRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PExpertKBlockFactorReuseRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PComponentStreamRHSSortedScalarMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_component_bits = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& component_scale_slots = inputs[6];
  auto& component_codeword_indices = inputs[7];
  auto& component_offsets = inputs[8];
  auto& codebook = inputs[9];
  auto& tile_experts = inputs[10];
  auto& tile_offsets = inputs[11];
  auto& tile_counts = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_component_bits.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_component_bits.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_component_bits.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_component_bits.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_component_bits.shape(4));
  uint32_t components = static_cast<uint32_t>(component_offsets.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_scalar_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_component_bits, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(component_scale_slots, 6);
  encoder.set_input_array(component_codeword_indices, 7);
  encoder.set_input_array(component_offsets, 8);
  encoder.set_input_array(codebook, 9);
  encoder.set_input_array(tile_experts, 10);
  encoder.set_input_array(tile_offsets, 11);
  encoder.set_input_array(tile_counts, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(components, 22);
  encoder.set_bytes(scale_groups, 23);
  encoder.set_bytes(num_route_tiles, 24);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PRouteSlotCodewordStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  const char* route_slot_contract =
      "route_tiles_x_route_slots_x_k_blocks_x_codewords";
  (void)route_slot_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(output_dims, 64u, num_route_tiles),
      MTL::Size::Make(1, 64, 1));
}

void NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t num_output_tiles = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  const char* route_slot_contract =
      "route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles";
  (void)route_slot_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, num_output_tiles, 1),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& active_route_tiles = inputs[9];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t active_route_tile_count =
      static_cast<uint32_t>(active_route_tiles.shape(0));
  uint32_t output_microtiles = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(active_route_tiles, 9);
  encoder.set_output_array(out, 10);
  encoder.set_bytes(route_count, 11);
  encoder.set_bytes(output_dims, 12);
  encoder.set_bytes(K, 13);
  encoder.set_bytes(experts, 14);
  encoder.set_bytes(n_tiles, 15);
  encoder.set_bytes(k_blocks, 16);
  encoder.set_bytes(bn, 17);
  encoder.set_bytes(codewords, 18);
  encoder.set_bytes(scale_groups, 19);
  encoder.set_bytes(num_route_tiles, 20);
  encoder.set_bytes(active_route_tile_count, 21);
  const char* active_route_tile_contract =
      "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles";
  (void)active_route_tile_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(active_route_tile_count, output_microtiles, 1),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& expert_cohort_offsets = inputs[9];
  auto& expert_cohort_counts = inputs[10];
  auto& route_cohort_offsets = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t expert_cohort_count =
      static_cast<uint32_t>(route_cohort_offsets.shape(0));
  uint32_t output_microtiles = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(expert_cohort_offsets, 9);
  encoder.set_input_array(expert_cohort_counts, 10);
  encoder.set_input_array(route_cohort_offsets, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(expert_cohort_count, 23);
  const char* expert_cohort_contract =
      "experts_x_route_cohorts_x_output_microtiles_x_k_blocks";
  (void)expert_cohort_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, expert_cohort_count, output_microtiles),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_batch_segment_offsets = inputs[9];
  auto& route_batch_segment_counts = inputs[10];
  auto& route_batch_route_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_batch_count =
      static_cast<uint32_t>(route_batch_segment_offsets.shape(0));
  uint32_t output_microtiles = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_batch_segment_offsets, 9);
  encoder.set_input_array(route_batch_segment_counts, 10);
  encoder.set_input_array(route_batch_route_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_batch_count, 23);
  const char* route_batch_dispatch_contract =
      "route_batches_x_k_blocks_x_output_microtiles_x_codewords";
  (void)route_batch_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(route_batch_count, k_blocks, output_microtiles),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PTokenCohortCodewordStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_cohort_offsets = inputs[9];
  auto& token_cohort_counts = inputs[10];
  auto& token_cohort_active_expert_ids = inputs[11];
  auto& token_cohort_route_slot_ids = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_cohort_count =
      static_cast<uint32_t>(token_cohort_offsets.shape(0));
  uint32_t active_expert_count =
      static_cast<uint32_t>(token_cohort_active_expert_ids.shape(0));
  uint32_t output_microtiles = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_cohort_offsets, 9);
  encoder.set_input_array(token_cohort_counts, 10);
  encoder.set_input_array(token_cohort_active_expert_ids, 11);
  encoder.set_input_array(token_cohort_route_slot_ids, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(scale_groups, 22);
  encoder.set_bytes(num_route_tiles, 23);
  encoder.set_bytes(token_cohort_count, 24);
  encoder.set_bytes(active_expert_count, 25);
  const char* token_cohort_dispatch_contract =
      "token_cohorts_x_active_experts_x_k_blocks_x_codewords";
  (void)token_cohort_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(token_cohort_count, active_expert_count, output_microtiles),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_cohort_offsets = inputs[9];
  auto& token_cohort_counts = inputs[10];
  auto& token_cohort_active_expert_ids = inputs[11];
  auto& token_cohort_route_slot_ids = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_cohort_count =
      static_cast<uint32_t>(token_cohort_offsets.shape(0));
  uint32_t active_expert_count =
      static_cast<uint32_t>(token_cohort_active_expert_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t codeword_tile_count = codewords;
  auto* pso =
      cache.get("nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_cohort_offsets, 9);
  encoder.set_input_array(token_cohort_counts, 10);
  encoder.set_input_array(token_cohort_active_expert_ids, 11);
  encoder.set_input_array(token_cohort_route_slot_ids, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(scale_groups, 22);
  encoder.set_bytes(num_route_tiles, 23);
  encoder.set_bytes(token_cohort_count, 24);
  encoder.set_bytes(active_expert_count, 25);
  const char* token_cohort_mma_dispatch_contract =
      "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles";
  (void)token_cohort_mma_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_cohort_count,
          output_tile_count,
          active_expert_count * k_blocks * codeword_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8POutputStationaryCodewordTileRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& output_stationary_route_batch_offsets = inputs[9];
  auto& output_stationary_route_batch_counts = inputs[10];
  auto& output_stationary_route_batch_active_expert_ids = inputs[11];
  auto& output_stationary_route_batch_route_slot_ids = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_batch_count =
      static_cast<uint32_t>(output_stationary_route_batch_offsets.shape(0));
  uint32_t active_expert_count = static_cast<uint32_t>(
      output_stationary_route_batch_active_expert_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t codeword_tile_count = codewords;
  auto* pso =
      cache.get("nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(output_stationary_route_batch_offsets, 9);
  encoder.set_input_array(output_stationary_route_batch_counts, 10);
  encoder.set_input_array(output_stationary_route_batch_active_expert_ids, 11);
  encoder.set_input_array(output_stationary_route_batch_route_slot_ids, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(scale_groups, 22);
  encoder.set_bytes(num_route_tiles, 23);
  encoder.set_bytes(route_batch_count, 24);
  encoder.set_bytes(active_expert_count, 25);
  const char* output_stationary_dispatch_contract =
      "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles";
  (void)output_stationary_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          output_tile_count,
          route_batch_count,
          active_expert_count * k_blocks * codeword_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PInputStationaryCodewordTileRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& input_stationary_route_batch_offsets = inputs[9];
  auto& input_stationary_route_batch_counts = inputs[10];
  auto& input_stationary_route_batch_active_expert_ids = inputs[11];
  auto& input_stationary_route_batch_route_slot_ids = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_batch_count =
      static_cast<uint32_t>(input_stationary_route_batch_offsets.shape(0));
  uint32_t active_expert_count = static_cast<uint32_t>(
      input_stationary_route_batch_active_expert_ids.shape(0));
  uint32_t input_tile_count = k_blocks;
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t codeword_tile_count = codewords;
  auto* pso =
      cache.get("nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(input_stationary_route_batch_offsets, 9);
  encoder.set_input_array(input_stationary_route_batch_counts, 10);
  encoder.set_input_array(input_stationary_route_batch_active_expert_ids, 11);
  encoder.set_input_array(input_stationary_route_batch_route_slot_ids, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(scale_groups, 22);
  encoder.set_bytes(num_route_tiles, 23);
  encoder.set_bytes(route_batch_count, 24);
  encoder.set_bytes(active_expert_count, 25);
  const char* input_stationary_dispatch_contract =
      "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles";
  (void)input_stationary_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          input_tile_count,
          route_batch_count,
          active_expert_count * output_tile_count * codeword_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_factor_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t expert_count = static_cast<uint32_t>(codeword_factor_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_factor_tiles.shape(1));
  uint32_t k_block_count =
      static_cast<uint32_t>(codeword_factor_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_factor_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_factor_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t route_tile_count = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_factor_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(expert_count, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_block_count, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codeword_tile_count, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(route_tile_count, 19);
  const char* expert_kblock_codeword_factor_dispatch_contract =
      "experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles";
  (void)expert_kblock_codeword_factor_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          expert_count,
          k_block_count,
          output_tile_count * route_tile_count * codeword_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t expert_count = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_block_count = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t route_tile_count = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(expert_count, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_block_count, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codeword_tile_count, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(route_tile_count, 19);
  const char* expert_kblock_scale_slot_stream_dispatch_contract =
      "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles";
  (void)expert_kblock_scale_slot_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          expert_count,
          k_block_count,
          scale_groups * route_tile_count * output_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& route_local_codeword_dot_lut = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_codeword_lut_route_slots = inputs[9];
  auto& route_codeword_lut_offsets = inputs[10];
  auto& route_codeword_lut_counts = inputs[11];
  auto& route_codeword_lut_codeword_ids = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count =
      static_cast<uint32_t>(route_local_codeword_dot_lut.shape(0));
  uint32_t k_block_count =
      static_cast<uint32_t>(route_local_codeword_dot_lut.shape(1));
  uint32_t codeword_count =
      static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t expert_count = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t route_tile_count = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_slot_count =
      static_cast<uint32_t>(route_codeword_lut_route_slots.shape(0));
  uint32_t route_codeword_lut_id_count =
      static_cast<uint32_t>(route_codeword_lut_codeword_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(route_local_codeword_dot_lut, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_codeword_lut_route_slots, 9);
  encoder.set_input_array(route_codeword_lut_offsets, 10);
  encoder.set_input_array(route_codeword_lut_counts, 11);
  encoder.set_input_array(route_codeword_lut_codeword_ids, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(expert_count, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_block_count, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(route_tile_count, 22);
  encoder.set_bytes(route_slot_count, 23);
  encoder.set_bytes(route_codeword_lut_id_count, 24);
  const char* route_codeword_lut_dispatch_contract =
      "route_slots_x_k_blocks_x_codewords_then_output_tiles";
  (void)route_codeword_lut_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_slot_count,
          k_block_count * codeword_count,
          output_tile_count),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& rowwise_route_microtile_offsets = inputs[9];
  auto& rowwise_route_microtile_counts = inputs[10];
  auto& rowwise_route_microtile_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_microtile_count =
      static_cast<uint32_t>(rowwise_route_microtile_offsets.shape(0));
  uint32_t route_microtile_slot_count = static_cast<uint32_t>(
      rowwise_route_microtile_route_slot_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t codeword_tile_count = codewords;
  auto* pso =
      cache.get("nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(rowwise_route_microtile_offsets, 9);
  encoder.set_input_array(rowwise_route_microtile_counts, 10);
  encoder.set_input_array(rowwise_route_microtile_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_microtile_count, 23);
  encoder.set_bytes(route_microtile_slot_count, 24);
  const char* rowwise_dispatch_contract =
      "route_microtiles_x_output_tiles_x_k_blocks_x_codeword_tiles";
  (void)rowwise_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_microtile_count,
          output_tile_count,
          k_blocks * codeword_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8POutputTileLocalCodewordLutRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& output_tile_local_route_microtile_offsets = inputs[9];
  auto& output_tile_local_route_microtile_counts = inputs[10];
  auto& output_tile_local_route_microtile_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_microtile_count = static_cast<uint32_t>(
      output_tile_local_route_microtile_offsets.shape(0));
  uint32_t route_microtile_slot_count = static_cast<uint32_t>(
      output_tile_local_route_microtile_route_slot_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t unique_codeword_count = codewords;
  auto* pso =
      cache.get("nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(output_tile_local_route_microtile_offsets, 9);
  encoder.set_input_array(output_tile_local_route_microtile_counts, 10);
  encoder.set_input_array(output_tile_local_route_microtile_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_microtile_count, 23);
  encoder.set_bytes(route_microtile_slot_count, 24);
  const char* output_tile_local_dispatch_contract =
      "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows";
  (void)output_tile_local_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_microtile_count,
          output_tile_count,
          k_blocks * unique_codeword_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_microtile_codeword_block_reduce_offsets = inputs[9];
  auto& route_microtile_codeword_block_reduce_counts = inputs[10];
  auto& route_microtile_codeword_block_reduce_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_microtile_count = static_cast<uint32_t>(
      route_microtile_codeword_block_reduce_offsets.shape(0));
  uint32_t route_microtile_slot_count = static_cast<uint32_t>(
      route_microtile_codeword_block_reduce_route_slot_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  uint32_t codeword_block_count = codewords;
  auto* pso =
      cache.get("nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_microtile_codeword_block_reduce_offsets, 9);
  encoder.set_input_array(route_microtile_codeword_block_reduce_counts, 10);
  encoder.set_input_array(
      route_microtile_codeword_block_reduce_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_microtile_count, 23);
  encoder.set_bytes(route_microtile_slot_count, 24);
  const char* route_microtile_codeword_block_reduce_dispatch_contract =
      "route_microtiles_x_k_blocks_x_codeword_blocks_then_output_tiles";
  (void)route_microtile_codeword_block_reduce_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_microtile_count,
          k_blocks * codeword_block_count,
          output_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& kblock_wavefront_codeword_scan_offsets = inputs[9];
  auto& kblock_wavefront_codeword_scan_counts = inputs[10];
  auto& kblock_wavefront_codeword_scan_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codeword_groups = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_microtile_count =
      static_cast<uint32_t>(kblock_wavefront_codeword_scan_offsets.shape(0));
  uint32_t route_microtile_slot_count = static_cast<uint32_t>(
      kblock_wavefront_codeword_scan_route_slot_ids.shape(0));
  uint32_t output_stripe_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(kblock_wavefront_codeword_scan_offsets, 9);
  encoder.set_input_array(kblock_wavefront_codeword_scan_counts, 10);
  encoder.set_input_array(kblock_wavefront_codeword_scan_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_groups, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_microtile_count, 23);
  encoder.set_bytes(route_microtile_slot_count, 24);
  const char* kblock_wavefront_codeword_scan_dispatch_contract =
      "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups";
  (void)kblock_wavefront_codeword_scan_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          k_blocks,
          route_microtile_count,
          output_stripe_count * codeword_groups),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_route_output_stripe_offsets = inputs[9];
  auto& token_route_output_stripe_counts = inputs[10];
  auto& token_route_output_stripe_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codeword_stages = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_count =
      static_cast<uint32_t>(token_route_output_stripe_offsets.shape(0));
  uint32_t route_slot_count = static_cast<uint32_t>(
      token_route_output_stripe_route_slot_ids.shape(0));
  uint32_t output_stripe_count = (output_dims + 63u) / 64u;
  uint32_t kblock_stage_count = k_blocks;
  auto* pso =
      cache.get("nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_route_output_stripe_offsets, 9);
  encoder.set_input_array(token_route_output_stripe_counts, 10);
  encoder.set_input_array(token_route_output_stripe_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_stages, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_count, 23);
  encoder.set_bytes(route_slot_count, 24);
  const char* token_route_output_stripe_pipeline_dispatch_contract =
      "tokens_x_route_slots_x_output_stripes_x_kblock_stages";
  (void)token_route_output_stripe_pipeline_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_count,
          1,
          output_stripe_count * kblock_stage_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& scale_group_route_block_offsets = inputs[9];
  auto& scale_group_route_block_counts = inputs[10];
  auto& scale_group_route_block_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_block_count =
      static_cast<uint32_t>(scale_group_route_block_offsets.shape(0));
  uint32_t route_block_slot_count = static_cast<uint32_t>(
      scale_group_route_block_route_slot_ids.shape(0));
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(scale_group_route_block_offsets, 9);
  encoder.set_input_array(scale_group_route_block_counts, 10);
  encoder.set_input_array(scale_group_route_block_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_tile_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_block_count, 23);
  encoder.set_bytes(route_block_slot_count, 24);
  const char* scale_group_route_block_reduce_dispatch_contract =
      "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles";
  (void)scale_group_route_block_reduce_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          scale_groups,
          route_block_count,
          k_blocks * output_tile_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_block_output_group_offsets = inputs[9];
  auto& route_block_output_group_counts = inputs[10];
  auto& route_block_output_group_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_block_count =
      static_cast<uint32_t>(route_block_output_group_offsets.shape(0));
  uint32_t route_block_slot_count = static_cast<uint32_t>(
      route_block_output_group_route_slot_ids.shape(0));
  uint32_t output_group_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_route_block_output_group_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_block_output_group_offsets, 9);
  encoder.set_input_array(route_block_output_group_counts, 10);
  encoder.set_input_array(route_block_output_group_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_block_count, 23);
  encoder.set_bytes(route_block_slot_count, 24);
  const char* route_block_output_group_stream_dispatch_contract =
      "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups";
  (void)route_block_output_group_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_block_count,
          output_group_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& output_group_pretransposed_route_offsets = inputs[9];
  auto& output_group_pretransposed_route_counts = inputs[10];
  auto& output_group_pretransposed_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_block_count = static_cast<uint32_t>(
      output_group_pretransposed_route_offsets.shape(0));
  uint32_t route_slot_count = static_cast<uint32_t>(
      output_group_pretransposed_route_slot_ids.shape(0));
  uint32_t output_group_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(output_group_pretransposed_route_offsets, 9);
  encoder.set_input_array(output_group_pretransposed_route_counts, 10);
  encoder.set_input_array(output_group_pretransposed_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_block_count, 23);
  encoder.set_bytes(route_slot_count, 24);
  const char* output_group_pretransposed_codeword_stream_dispatch_contract =
      "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups";
  (void)output_group_pretransposed_codeword_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          output_group_count,
          route_block_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& kblock_route_fused_offsets = inputs[9];
  auto& kblock_route_fused_counts = inputs[10];
  auto& kblock_route_fused_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_block_count =
      static_cast<uint32_t>(kblock_route_fused_offsets.shape(0));
  uint32_t route_slot_count =
      static_cast<uint32_t>(kblock_route_fused_route_slot_ids.shape(0));
  uint32_t output_group_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(kblock_route_fused_offsets, 9);
  encoder.set_input_array(kblock_route_fused_counts, 10);
  encoder.set_input_array(kblock_route_fused_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_block_count, 23);
  encoder.set_bytes(route_slot_count, 24);
  const char* kblock_output_group_route_fused_stream_dispatch_contract =
      "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups";
  (void)kblock_output_group_route_fused_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          k_blocks,
          output_group_count,
          route_block_count * codeword_group_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_tile_output_swizzle_offsets = inputs[9];
  auto& route_tile_output_swizzle_counts = inputs[10];
  auto& route_tile_output_swizzle_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_tile_count =
      static_cast<uint32_t>(route_tile_output_swizzle_offsets.shape(0));
  uint32_t route_slot_count =
      static_cast<uint32_t>(route_tile_output_swizzle_route_slot_ids.shape(0));
  uint32_t output_swizzle_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_tile_output_swizzle_offsets, 9);
  encoder.set_input_array(route_tile_output_swizzle_counts, 10);
  encoder.set_input_array(route_tile_output_swizzle_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_tile_count, 23);
  encoder.set_bytes(route_slot_count, 24);
  const char* route_tile_output_swizzle_stream_dispatch_contract =
      "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups";
  (void)route_tile_output_swizzle_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_tile_count,
          output_swizzle_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 8, 1));
}

void NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_topk_offsets = inputs[9];
  auto& token_topk_counts = inputs[10];
  auto& token_topk_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_count = static_cast<uint32_t>(token_topk_offsets.shape(0));
  uint32_t token_topk_slot_count =
      static_cast<uint32_t>(token_topk_route_slot_ids.shape(0));
  uint32_t topk_slot_capacity =
      (token_topk_slot_count + token_count - 1u) / token_count;
  uint32_t output_tile_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_topk_offsets, 9);
  encoder.set_input_array(token_topk_counts, 10);
  encoder.set_input_array(token_topk_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_count, 23);
  encoder.set_bytes(token_topk_slot_count, 24);
  encoder.set_bytes(topk_slot_capacity, 25);
  const char* token_topk_output_tile_stream_dispatch_contract =
      "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups";
  (void)token_topk_output_tile_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_count,
          output_tile_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, topk_slot_capacity, 1));
}

void NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_block_offsets = inputs[9];
  auto& token_block_counts = inputs[10];
  auto& token_block_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_block_count =
      static_cast<uint32_t>(token_block_offsets.shape(0));
  uint32_t token_block_topk_slot_count =
      static_cast<uint32_t>(token_block_route_slot_ids.shape(0));
  uint32_t topk_slot_capacity =
      (token_block_topk_slot_count + token_block_count - 1u) /
      token_block_count;
  uint32_t output_group_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_token_block_output_group_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_block_offsets, 9);
  encoder.set_input_array(token_block_counts, 10);
  encoder.set_input_array(token_block_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_block_count, 23);
  encoder.set_bytes(token_block_topk_slot_count, 24);
  encoder.set_bytes(topk_slot_capacity, 25);
  const char* token_block_output_group_stream_dispatch_contract =
      "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups";
  (void)token_block_output_group_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_block_count,
          output_group_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, topk_slot_capacity, 1));
}

void NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_output_stripe_offsets = inputs[9];
  auto& token_output_stripe_counts = inputs[10];
  auto& token_output_stripe_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_count =
      static_cast<uint32_t>(token_output_stripe_offsets.shape(0));
  uint32_t token_output_stripe_slot_count =
      static_cast<uint32_t>(token_output_stripe_route_slot_ids.shape(0));
  uint32_t topk_group_capacity =
      (token_output_stripe_slot_count + token_count - 1u) / token_count;
  uint32_t output_stripe_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_output_stripe_offsets, 9);
  encoder.set_input_array(token_output_stripe_counts, 10);
  encoder.set_input_array(token_output_stripe_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_count, 23);
  encoder.set_bytes(token_output_stripe_slot_count, 24);
  encoder.set_bytes(topk_group_capacity, 25);
  const char* token_output_stripe_group_stream_dispatch_contract =
      "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups";
  (void)token_output_stripe_group_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_count,
          output_stripe_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, topk_group_capacity, 1));
}

void NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_expert_output_block_offsets = inputs[9];
  auto& token_expert_output_block_counts = inputs[10];
  auto& token_expert_output_block_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_count =
      static_cast<uint32_t>(token_expert_output_block_offsets.shape(0));
  uint32_t token_expert_output_block_slot_count =
      static_cast<uint32_t>(token_expert_output_block_route_slot_ids.shape(0));
  uint32_t output_block_count = (output_dims + 63u) / 64u;
  auto* pso =
      cache.get("nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_expert_output_block_offsets, 9);
  encoder.set_input_array(token_expert_output_block_counts, 10);
  encoder.set_input_array(token_expert_output_block_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_count, 23);
  encoder.set_bytes(token_expert_output_block_slot_count, 24);
  encoder.set_bytes(output_block_count, 25);
  const char* token_expert_output_block_stream_dispatch_contract =
      "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups";
  (void)token_expert_output_block_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_count,
          experts * output_block_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_pair_kblock_offsets = inputs[9];
  auto& token_pair_kblock_counts = inputs[10];
  auto& token_pair_kblock_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(token_pair_kblock_offsets.shape(0));
  uint32_t token_pair_kblock_slot_count =
      static_cast<uint32_t>(token_pair_kblock_route_slot_ids.shape(0));
  uint32_t output_block_count = (output_dims + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_pair_kblock_offsets, 9);
  encoder.set_input_array(token_pair_kblock_counts, 10);
  encoder.set_input_array(token_pair_kblock_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_pair_count, 23);
  encoder.set_bytes(token_pair_kblock_slot_count, 24);
  encoder.set_bytes(output_block_count, 25);
  const char* token_pair_kblock_accumulator_stream_dispatch_contract =
      "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups";
  (void)token_pair_kblock_accumulator_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          experts * output_block_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_pair_output_group_offsets = inputs[9];
  auto& token_pair_output_group_counts = inputs[10];
  auto& token_pair_output_group_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(token_pair_output_group_offsets.shape(0));
  uint32_t output_group_count =
      static_cast<uint32_t>(token_pair_output_group_offsets.shape(1));
  uint32_t token_pair_output_group_slot_count =
      static_cast<uint32_t>(token_pair_output_group_route_slot_ids.shape(0));
  auto* pso = cache.get("nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_pair_output_group_offsets, 9);
  encoder.set_input_array(token_pair_output_group_counts, 10);
  encoder.set_input_array(token_pair_output_group_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_pair_count, 23);
  encoder.set_bytes(token_pair_output_group_slot_count, 24);
  encoder.set_bytes(output_group_count, 25);
  const char* token_pair_output_group_stream_dispatch_contract =
      "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups";
  (void)token_pair_output_group_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          output_group_count * experts,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_pair_slot_topk_output_group_offsets = inputs[9];
  auto& token_pair_slot_topk_output_group_counts = inputs[10];
  auto& token_pair_slot_topk_output_group_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_pair_count = static_cast<uint32_t>(
      token_pair_slot_topk_output_group_offsets.shape(0));
  uint32_t pair_slot_count = static_cast<uint32_t>(
      token_pair_slot_topk_output_group_offsets.shape(1));
  uint32_t topk_slot_count = static_cast<uint32_t>(
      token_pair_slot_topk_output_group_offsets.shape(2));
  uint32_t output_group_count = static_cast<uint32_t>(
      token_pair_slot_topk_output_group_offsets.shape(3));
  uint32_t token_pair_slot_topk_output_group_slot_count = static_cast<uint32_t>(
      token_pair_slot_topk_output_group_route_slot_ids.shape(0));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(token_pair_slot_topk_output_group_offsets, 9);
  encoder.set_input_array(token_pair_slot_topk_output_group_counts, 10);
  encoder.set_input_array(
      token_pair_slot_topk_output_group_route_slot_ids,
      11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_pair_count, 23);
  encoder.set_bytes(token_pair_slot_topk_output_group_slot_count, 24);
  encoder.set_bytes(pair_slot_count, 25);
  encoder.set_bytes(topk_slot_count, 26);
  encoder.set_bytes(output_group_count, 27);
  const char* token_pair_slot_topk_output_group_stream_dispatch_contract =
      "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups";
  (void)token_pair_slot_topk_output_group_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          pair_slot_count * topk_slot_count * output_group_count,
          k_blocks * codeword_group_count),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& token_pair_slot_topk_codeword_group_pipeline_offsets = inputs[9];
  auto& token_pair_slot_topk_codeword_group_pipeline_counts = inputs[10];
  auto& token_pair_slot_topk_codeword_group_pipeline_route_slot_ids =
      inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_group_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_pair_count = static_cast<uint32_t>(
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(0));
  uint32_t pair_slot_count = static_cast<uint32_t>(
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(1));
  uint32_t topk_slot_count = static_cast<uint32_t>(
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(2));
  uint32_t output_stripe_count = static_cast<uint32_t>(
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(4));
  uint32_t token_pair_slot_topk_codeword_group_pipeline_slot_count =
      static_cast<uint32_t>(
          token_pair_slot_topk_codeword_group_pipeline_route_slot_ids.shape(0));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(
      token_pair_slot_topk_codeword_group_pipeline_offsets,
      9);
  encoder.set_input_array(
      token_pair_slot_topk_codeword_group_pipeline_counts,
      10);
  encoder.set_input_array(
      token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
      11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_group_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_pair_count, 23);
  encoder.set_bytes(token_pair_slot_topk_codeword_group_pipeline_slot_count, 24);
  encoder.set_bytes(pair_slot_count, 25);
  encoder.set_bytes(topk_slot_count, 26);
  encoder.set_bytes(output_stripe_count, 27);
  const char* token_pair_slot_topk_codeword_group_pipeline_dispatch_contract =
      "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks";
  (void)token_pair_slot_topk_codeword_group_pipeline_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          pair_slot_count * topk_slot_count * codeword_group_count *
              output_stripe_count,
          k_blocks),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& scale_slot_broadcast_offsets = inputs[9];
  auto& scale_slot_broadcast_counts = inputs[10];
  auto& scale_slot_broadcast_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_count = static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(scale_slot_broadcast_offsets.shape(0));
  uint32_t pair_slot_count =
      static_cast<uint32_t>(scale_slot_broadcast_offsets.shape(1));
  uint32_t topk_slot_count =
      static_cast<uint32_t>(scale_slot_broadcast_offsets.shape(2));
  uint32_t scale_slot_count =
      static_cast<uint32_t>(scale_slot_broadcast_offsets.shape(3));
  uint32_t output_stripe_count =
      static_cast<uint32_t>(scale_slot_broadcast_offsets.shape(4));
  uint32_t scale_slot_broadcast_slot_count =
      static_cast<uint32_t>(scale_slot_broadcast_route_slot_ids.shape(0));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(scale_slot_broadcast_offsets, 9);
  encoder.set_input_array(scale_slot_broadcast_counts, 10);
  encoder.set_input_array(scale_slot_broadcast_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(token_pair_count, 23);
  encoder.set_bytes(scale_slot_broadcast_slot_count, 24);
  encoder.set_bytes(pair_slot_count, 25);
  encoder.set_bytes(topk_slot_count, 26);
  encoder.set_bytes(scale_slot_count, 27);
  encoder.set_bytes(output_stripe_count, 28);
  const char* token_pair_slot_topk_scale_slot_broadcast_stream_dispatch_contract =
      "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords";
  (void)token_pair_slot_topk_scale_slot_broadcast_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          pair_slot_count * topk_slot_count * scale_slot_count *
              output_stripe_count,
          k_blocks),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& route_bucket_offsets = inputs[9];
  auto& route_bucket_counts = inputs[10];
  auto& route_bucket_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_bucket_count =
      static_cast<uint32_t>(route_bucket_offsets.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(route_bucket_offsets.shape(1));
  uint32_t pair_slot_count =
      static_cast<uint32_t>(route_bucket_offsets.shape(2));
  uint32_t topk_slot_count =
      static_cast<uint32_t>(route_bucket_offsets.shape(3));
  uint32_t output_microtile_count =
      static_cast<uint32_t>(route_bucket_offsets.shape(5));
  uint32_t route_bucket_slot_count =
      static_cast<uint32_t>(route_bucket_route_slot_ids.shape(0));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(route_bucket_offsets, 9);
  encoder.set_input_array(route_bucket_counts, 10);
  encoder.set_input_array(route_bucket_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_tile_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(route_bucket_count, 23);
  encoder.set_bytes(route_bucket_slot_count, 24);
  encoder.set_bytes(token_pair_count, 25);
  encoder.set_bytes(pair_slot_count, 26);
  encoder.set_bytes(topk_slot_count, 27);
  encoder.set_bytes(output_microtile_count, 28);
  const char*
      token_pair_slot_topk_route_bucket_codeword_reduce_dispatch_contract =
          "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles";
  (void)token_pair_slot_topk_route_bucket_codeword_reduce_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          route_bucket_count,
          token_pair_count * pair_slot_count * topk_slot_count *
              output_microtile_count,
          k_blocks),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& kblock_microtile_offsets = inputs[9];
  auto& kblock_microtile_counts = inputs[10];
  auto& kblock_microtile_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t kblock_microtile_slot_count =
      static_cast<uint32_t>(kblock_microtile_route_slot_ids.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(kblock_microtile_offsets.shape(0));
  uint32_t pair_slot_count =
      static_cast<uint32_t>(kblock_microtile_offsets.shape(1));
  uint32_t topk_slot_count =
      static_cast<uint32_t>(kblock_microtile_offsets.shape(2));
  uint32_t output_microtile_count =
      static_cast<uint32_t>(kblock_microtile_offsets.shape(4));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(kblock_microtile_offsets, 9);
  encoder.set_input_array(kblock_microtile_counts, 10);
  encoder.set_input_array(kblock_microtile_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_tile_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(kblock_microtile_slot_count, 23);
  encoder.set_bytes(token_pair_count, 24);
  encoder.set_bytes(pair_slot_count, 25);
  encoder.set_bytes(topk_slot_count, 26);
  encoder.set_bytes(output_microtile_count, 27);
  const char*
      token_pair_slot_topk_kblock_microtile_stream_dispatch_contract =
          "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles";
  (void)token_pair_slot_topk_kblock_microtile_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          pair_slot_count * topk_slot_count * output_microtile_count,
          k_blocks),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codeword_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& output_tile_fused_offsets = inputs[9];
  auto& output_tile_fused_counts = inputs[10];
  auto& output_tile_fused_route_slot_ids = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(codeword_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(codeword_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(codeword_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(codeword_tiles.shape(3));
  uint32_t codeword_tile_count =
      static_cast<uint32_t>(codeword_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t output_tile_fused_slot_count =
      static_cast<uint32_t>(output_tile_fused_route_slot_ids.shape(0));
  uint32_t token_pair_count =
      static_cast<uint32_t>(output_tile_fused_offsets.shape(0));
  uint32_t pair_slot_count =
      static_cast<uint32_t>(output_tile_fused_offsets.shape(1));
  uint32_t topk_slot_count =
      static_cast<uint32_t>(output_tile_fused_offsets.shape(2));
  uint32_t output_tile_count =
      static_cast<uint32_t>(output_tile_fused_offsets.shape(3));
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codeword_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_input_array(output_tile_fused_offsets, 9);
  encoder.set_input_array(output_tile_fused_counts, 10);
  encoder.set_input_array(output_tile_fused_route_slot_ids, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codeword_tile_count, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.set_bytes(output_tile_fused_slot_count, 23);
  encoder.set_bytes(token_pair_count, 24);
  encoder.set_bytes(pair_slot_count, 25);
  encoder.set_bytes(topk_slot_count, 26);
  encoder.set_bytes(output_tile_count, 27);
  const char* token_pair_slot_topk_output_tile_fused_stream_dispatch_contract =
      "token_pairs_x_pair_slots_x_topk_slots_x_output_tiles";
  (void)token_pair_slot_topk_output_tile_fused_stream_dispatch_contract;
  encoder.dispatch_threadgroups(
      MTL::Size::Make(
          token_pair_count,
          pair_slot_count * topk_slot_count * output_tile_count,
          1),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PComponentStreamRHSSortedPartialMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_component_bits = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& component_scale_slots = inputs[6];
  auto& component_codeword_indices = inputs[7];
  auto& component_offsets = inputs[8];
  auto& codebook = inputs[9];
  auto& tile_experts = inputs[10];
  auto& tile_offsets = inputs[11];
  auto& tile_counts = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_component_bits.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_component_bits.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_component_bits.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_component_bits.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_component_bits.shape(4));
  uint32_t components = static_cast<uint32_t>(component_offsets.shape(1));
  uint32_t component_pairs = (components + 3u) / 4u;
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  int num_route_tiles_shape = static_cast<int>(num_route_tiles);
  int k_blocks_shape = static_cast<int>(k_blocks);
  int component_pairs_shape = static_cast<int>(component_pairs);
  int n_tiles_shape = static_cast<int>(n_tiles);
  int route_tile_size_shape = 64;
  int component_width_shape = 64;
  auto component_stream_partials = array(
      {num_route_tiles_shape, k_blocks_shape, component_pairs_shape,
       n_tiles_shape, route_tile_size_shape, component_width_shape},
      float32,
      nullptr,
      {});
  component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(partial_pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_component_bits, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(component_scale_slots, 6);
  encoder.set_input_array(component_codeword_indices, 7);
  encoder.set_input_array(component_offsets, 8);
  encoder.set_input_array(codebook, 9);
  encoder.set_input_array(tile_experts, 10);
  encoder.set_input_array(tile_offsets, 11);
  encoder.set_input_array(tile_counts, 12);
  encoder.set_output_array(component_stream_partials, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(components, 22);
  encoder.set_bytes(component_pairs, 23);
  encoder.set_bytes(scale_groups, 24);
  encoder.set_bytes(num_route_tiles, 25);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, component_pairs, k_blocks),
      MTL::Size::Make(64, 1, 1));

  auto* reduce_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_reduce");
  encoder.set_compute_pipeline_state(reduce_pso);
  encoder.set_input_array(component_stream_partials, 0);
  encoder.set_output_array(out, 1);
  encoder.set_input_array(tile_offsets, 2);
  encoder.set_input_array(tile_counts, 3);
  encoder.set_bytes(route_count, 4);
  encoder.set_bytes(output_dims, 5);
  encoder.set_bytes(n_tiles, 6);
  encoder.set_bytes(k_blocks, 7);
  encoder.set_bytes(component_pairs, 8);
  encoder.set_bytes(num_route_tiles, 9);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PComponentStreamRHSSortedTensorOpsMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_component_bits = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& component_scale_slots = inputs[6];
  auto& component_codeword_indices = inputs[7];
  auto& component_offsets = inputs[8];
  auto& codebook = inputs[9];
  auto& tile_experts = inputs[10];
  auto& tile_offsets = inputs[11];
  auto& tile_counts = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_component_bits.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_component_bits.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_component_bits.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_component_bits.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_component_bits.shape(4));
  uint32_t components = static_cast<uint32_t>(component_offsets.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_tensorops_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_component_bits, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(component_scale_slots, 6);
  encoder.set_input_array(component_codeword_indices, 7);
  encoder.set_input_array(component_offsets, 8);
  encoder.set_input_array(codebook, 9);
  encoder.set_input_array(tile_experts, 10);
  encoder.set_input_array(tile_offsets, 11);
  encoder.set_input_array(tile_counts, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(components, 22);
  encoder.set_bytes(scale_groups, 23);
  encoder.set_bytes(num_route_tiles, 24);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PComponentStreamRHSSortedSharedDecodeMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_component_bits = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& scale_tiles = inputs[3];
  auto& scale_group_indices = inputs[4];
  auto& codeword_scale_slots = inputs[5];
  auto& component_scale_slots = inputs[6];
  auto& component_codeword_indices = inputs[7];
  auto& component_offsets = inputs[8];
  auto& codebook = inputs[9];
  auto& tile_experts = inputs[10];
  auto& tile_offsets = inputs[11];
  auto& tile_counts = inputs[12];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_component_bits.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_component_bits.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_component_bits.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_component_bits.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_component_bits.shape(4));
  uint32_t components = static_cast<uint32_t>(component_offsets.shape(1));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_shared_decode_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_component_bits, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(scale_tiles, 3);
  encoder.set_input_array(scale_group_indices, 4);
  encoder.set_input_array(codeword_scale_slots, 5);
  encoder.set_input_array(component_scale_slots, 6);
  encoder.set_input_array(component_codeword_indices, 7);
  encoder.set_input_array(component_offsets, 8);
  encoder.set_input_array(codebook, 9);
  encoder.set_input_array(tile_experts, 10);
  encoder.set_input_array(tile_offsets, 11);
  encoder.set_input_array(tile_counts, 12);
  encoder.set_output_array(out, 13);
  encoder.set_bytes(route_count, 14);
  encoder.set_bytes(output_dims, 15);
  encoder.set_bytes(K, 16);
  encoder.set_bytes(experts, 17);
  encoder.set_bytes(n_tiles, 18);
  encoder.set_bytes(k_blocks, 19);
  encoder.set_bytes(bn, 20);
  encoder.set_bytes(codewords, 21);
  encoder.set_bytes(components, 22);
  encoder.set_bytes(scale_groups, 23);
  encoder.set_bytes(num_route_tiles, 24);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso =
      cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  int num_route_tiles_shape = static_cast<int>(num_route_tiles);
  int k_blocks_shape = static_cast<int>(k_blocks);
  int n_tiles_shape = static_cast<int>(n_tiles);
  int route_tile_size_shape = 64;
  int bn_shape = static_cast<int>(bn);
  array kblock_partials(
      {num_route_tiles_shape, k_blocks_shape, n_tiles_shape, route_tile_size_shape, bn_shape},
      float32,
      nullptr,
      {});
  kblock_partials.set_data(allocator::malloc(kblock_partials.nbytes()));
  auto* pso =
      cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(kblock_partials, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, k_blocks, num_route_tiles),
      MTL::Size::Make(128, 1, 1));

  auto* reduce_pso =
      cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce");
  encoder.set_compute_pipeline_state(reduce_pso);
  encoder.set_input_array(kblock_partials, 0);
  encoder.set_output_array(out, 1);
  encoder.set_input_array(tile_offsets, 2);
  encoder.set_input_array(tile_counts, 3);
  encoder.set_bytes(route_count, 4);
  encoder.set_bytes(output_dims, 5);
  encoder.set_bytes(n_tiles, 6);
  encoder.set_bytes(k_blocks, 7);
  encoder.set_bytes(bn, 8);
  encoder.set_bytes(num_route_tiles, 9);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PPackedRHSSortedTiledMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_packed_rhs_sorted_tiled_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSplitByteRHSSortedTiledMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_tiles = inputs[1];
  auto& abs_index_tiles = inputs[2];
  auto& parity_tiles = inputs[3];
  auto& scale_tiles = inputs[4];
  auto& scale_group_indices = inputs[5];
  auto& codeword_scale_slots = inputs[6];
  auto& codebook = inputs[7];
  auto& tile_experts = inputs[8];
  auto& tile_offsets = inputs[9];
  auto& tile_counts = inputs[10];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_split_byte_rhs_sorted_tiled_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_tiles, 1);
  encoder.set_input_array(abs_index_tiles, 2);
  encoder.set_input_array(parity_tiles, 3);
  encoder.set_input_array(scale_tiles, 4);
  encoder.set_input_array(scale_group_indices, 5);
  encoder.set_input_array(codeword_scale_slots, 6);
  encoder.set_input_array(codebook, 7);
  encoder.set_input_array(tile_experts, 8);
  encoder.set_input_array(tile_offsets, 9);
  encoder.set_input_array(tile_counts, 10);
  encoder.set_output_array(out, 11);
  encoder.set_bytes(route_count, 12);
  encoder.set_bytes(output_dims, 13);
  encoder.set_bytes(K, 14);
  encoder.set_bytes(experts, 15);
  encoder.set_bytes(n_tiles, 16);
  encoder.set_bytes(k_blocks, 17);
  encoder.set_bytes(bn, 18);
  encoder.set_bytes(codewords, 19);
  encoder.set_bytes(scale_groups, 20);
  encoder.set_bytes(num_route_tiles, 21);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& sign_byte_lut = inputs[1];
  auto& sign_byte_slots = inputs[2];
  auto& abs_index_lut = inputs[3];
  auto& abs_index_slots = inputs[4];
  auto& scale_tiles = inputs[5];
  auto& scale_group_indices = inputs[6];
  auto& codeword_scale_slots = inputs[7];
  auto& codebook = inputs[8];
  auto& tile_experts = inputs[9];
  auto& tile_offsets = inputs[10];
  auto& tile_counts = inputs[11];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(2));
  uint32_t bn = static_cast<uint32_t>(sign_byte_slots.shape(3));
  uint32_t codewords = static_cast<uint32_t>(sign_byte_slots.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get(
      "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(sign_byte_lut, 1);
  encoder.set_input_array(sign_byte_slots, 2);
  encoder.set_input_array(abs_index_lut, 3);
  encoder.set_input_array(abs_index_slots, 4);
  encoder.set_input_array(scale_tiles, 5);
  encoder.set_input_array(scale_group_indices, 6);
  encoder.set_input_array(codeword_scale_slots, 7);
  encoder.set_input_array(codebook, 8);
  encoder.set_input_array(tile_experts, 9);
  encoder.set_input_array(tile_offsets, 10);
  encoder.set_input_array(tile_counts, 11);
  encoder.set_output_array(out, 12);
  encoder.set_bytes(route_count, 13);
  encoder.set_bytes(output_dims, 14);
  encoder.set_bytes(K, 15);
  encoder.set_bytes(experts, 16);
  encoder.set_bytes(n_tiles, 17);
  encoder.set_bytes(k_blocks, 18);
  encoder.set_bytes(bn, 19);
  encoder.set_bytes(codewords, 20);
  encoder.set_bytes(scale_groups, 21);
  encoder.set_bytes(num_route_tiles, 22);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PPackedRHSSortedTiledM128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_packed_rhs_sorted_tiled_m128_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(256, 1, 1));
}

void NaxE8PPackedRHSSortedTiledK128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& code_tiles = inputs[1];
  auto& scale_tiles = inputs[2];
  auto& scale_group_indices = inputs[3];
  auto& codeword_scale_slots = inputs[4];
  auto& codebook = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t experts = static_cast<uint32_t>(code_tiles.shape(0));
  uint32_t n_tiles = static_cast<uint32_t>(code_tiles.shape(1));
  uint32_t k_blocks = static_cast<uint32_t>(code_tiles.shape(2));
  uint32_t bn = static_cast<uint32_t>(code_tiles.shape(3));
  uint32_t codewords = static_cast<uint32_t>(code_tiles.shape(4));
  uint32_t scale_groups = static_cast<uint32_t>(scale_tiles.shape(4));
  uint32_t output_dims = static_cast<uint32_t>(output_dims_);
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_packed_rhs_sorted_tiled_k128_matmul");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(code_tiles, 1);
  encoder.set_input_array(scale_tiles, 2);
  encoder.set_input_array(scale_group_indices, 3);
  encoder.set_input_array(codeword_scale_slots, 4);
  encoder.set_input_array(codebook, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(output_dims, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(experts, 13);
  encoder.set_bytes(n_tiles, 14);
  encoder.set_bytes(k_blocks, 15);
  encoder.set_bytes(bn, 16);
  encoder.set_bytes(codewords, 17);
  encoder.set_bytes(scale_groups, 18);
  encoder.set_bytes(num_route_tiles, 19);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedDirectReduceMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_direct_reduce");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.set_bytes(num_tiles, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make((N + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}

void NaxE8PFp16SortedInlineBMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_inline_b");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelGs352Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_gs352");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelLutMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& full_grid = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_lut");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(full_grid, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelTgcbMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_tgcb");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelTgscaleMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_tgscale");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelTgcbTgscaleMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_tgcb_tgscale");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelTgcbHoistMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_tgcb_hoist");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelBk128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_bk128");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelM128N32Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 31u) / 32u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m128n32");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelM64N128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 127u) / 128u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m64n128");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(256, 1, 1));
}

void NaxE8PFp16SortedSteelM32N64Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m32n64");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PFp16SortedSteelM64N64T64Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m64n64t64");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}

void NaxE8PFp16SortedSteelM32N64T128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m32n64t128");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8PFp16SortedSteelM32N128Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted_x = inputs[0];
  auto& codes = inputs[1];
  auto& scales = inputs[2];
  auto& codebook = inputs[3];
  auto& tile_experts = inputs[4];
  auto& tile_offsets = inputs[5];
  auto& tile_counts = inputs[6];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  uint32_t route_count = static_cast<uint32_t>(sorted_x.shape(0));
  uint32_t K = static_cast<uint32_t>(sorted_x.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 127u) / 128u;
  auto* pso = cache.get("nax_e8p_fp16_sorted_matmul_steel_m32n128");

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(sorted_x, 0);
  encoder.set_input_array(codes, 1);
  encoder.set_input_array(scales, 2);
  encoder.set_input_array(codebook, 3);
  encoder.set_input_array(tile_experts, 4);
  encoder.set_input_array(tile_offsets, 5);
  encoder.set_input_array(tile_counts, 6);
  encoder.set_output_array(out, 7);
  encoder.set_bytes(route_count, 8);
  encoder.set_bytes(N, 9);
  encoder.set_bytes(K, 10);
  encoder.set_bytes(group_size, 11);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

void NaxE8Int8RoutedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& x_q = inputs[0];
  auto& x_scales = inputs[1];
  auto& codes = inputs[2];
  auto& scales = inputs[3];
  auto& codebook = inputs[4];
  auto& lhs_indices = inputs[5];
  auto& tile_experts = inputs[6];
  auto& tile_offsets = inputs[7];
  auto& tile_counts = inputs[8];
  auto& out = outputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);
  auto* pso = cache.get("nax_e8_int8_routed_matmul");

  uint32_t route_count = static_cast<uint32_t>(lhs_indices.shape(0));
  uint32_t K = static_cast<uint32_t>(x_q.shape(1));
  uint32_t N = static_cast<uint32_t>(codes.shape(1));
  uint32_t num_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t group_size = static_cast<uint32_t>(group_size_);
  uint32_t tiles_n = (N + 63u) / 64u;

  auto& encoder = metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(pso);
  encoder.set_input_array(x_q, 0);
  encoder.set_input_array(x_scales, 1);
  encoder.set_input_array(codes, 2);
  encoder.set_input_array(scales, 3);
  encoder.set_input_array(codebook, 4);
  encoder.set_input_array(lhs_indices, 5);
  encoder.set_input_array(tile_experts, 6);
  encoder.set_input_array(tile_offsets, 7);
  encoder.set_input_array(tile_counts, 8);
  encoder.set_output_array(out, 9);
  encoder.set_bytes(route_count, 10);
  encoder.set_bytes(N, 11);
  encoder.set_bytes(K, 12);
  encoder.set_bytes(group_size, 13);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(tiles_n, num_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

array fp16_matmul_tile(
    const array& x,
    const array& weight_t,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || weight_t.ndim() != 2) {
    throw std::invalid_argument("fp16_matmul_tile expects x [32,16] and weight_t [16,16]");
  }
  if (x.shape(0) != 32 || x.shape(1) != 16 || weight_t.shape(0) != 16 ||
      weight_t.shape(1) != 16) {
    throw std::invalid_argument("fp16_matmul_tile only supports a 32x16 by 16x16 NAX smoke tile");
  }
  auto stream = to_stream(s);
  return array(
      {32, 16},
      float16,
      std::make_shared<NaxFp16MatmulTile>(stream, kernel_dir),
      {astype(x, float16, stream), astype(weight_t, float16, stream)});
}

array e8_fp16_matmul_tile(
    const array& x,
    const array& codes,
    const array& scales,
    const array& codebook,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || codes.ndim() != 2 || scales.ndim() != 2 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8_fp16_matmul_tile expects x [32,16], codes [16,2], scales [16,16/group], codebook [256]");
  }
  if (x.shape(0) != 32 || x.shape(1) != 16 || codes.shape(0) != 16 ||
      codes.shape(1) != 2 || scales.shape(0) != 16 || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8_fp16_matmul_tile only supports a 32x16 by fused-E8 16x16 NAX smoke tile");
  }
  if (group_size != 8 && group_size != 16) {
    throw std::invalid_argument("e8_fp16_matmul_tile smoke supports group_size 8 or 16");
  }
  if (scales.shape(1) != 16 / group_size) {
    throw std::invalid_argument("e8_fp16_matmul_tile scales shape does not match group_size");
  }
  auto stream = to_stream(s);
  return array(
      {32, 16},
      float16,
      std::make_shared<NaxE8Fp16MatmulTile>(stream, kernel_dir, group_size),
      {astype(x, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8_fp16_matmul(
    const array& x,
    const array& codes,
    const array& scales,
    const array& codebook,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || codes.ndim() != 2 || scales.ndim() != 2 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8_fp16_matmul expects x [M,K], codes [N,K/8], scales [N,K/group], codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int N = codes.shape(0);
  if (M <= 0 || N <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument("e8_fp16_matmul requires positive M/N/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument("e8_fp16_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(1) != K / 8 || scales.shape(0) != N ||
      scales.shape(1) != K / group_size || codebook.shape(0) != 256) {
    throw std::invalid_argument("e8_fp16_matmul input shapes do not match K/group_size");
  }
  auto stream = to_stream(s);
  return array(
      {M, N},
      float16,
      std::make_shared<NaxE8Fp16Matmul>(stream, kernel_dir, group_size),
      {astype(x, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8_fp16_routed_matmul(
    const array& x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& lhs_indices,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || lhs_indices.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8_fp16_routed_matmul expects x [T,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D route descriptors");
  }
  int K = x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  int route_count = lhs_indices.shape(0);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument("e8_fp16_routed_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument("e8_fp16_routed_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument("e8_fp16_routed_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8Fp16RoutedMatmul>(stream, kernel_dir, group_size),
      {astype(x, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(lhs_indices, int32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8_fp16_routed_steel_matmul(
    const array& x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& lhs_indices,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || lhs_indices.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8_fp16_routed_steel_matmul expects x [T,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D route descriptors");
  }
  int K = x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  int route_count = lhs_indices.shape(0);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8_fp16_routed_steel_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8_fp16_routed_steel_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8_fp16_routed_steel_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8Fp16RoutedSteelMatmul>(stream, kernel_dir, group_size),
      {astype(x, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(lhs_indices, int32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8_fp16_sorted_steel_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8_fp16_sorted_steel_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8_fp16_sorted_steel_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8_fp16_sorted_steel_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8_fp16_sorted_steel_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8Fp16SortedSteelMatmul>(stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
      astype(tile_offsets, int32, stream),
      astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelMatmul>(stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_packed_rhs_tile_matmul(
    const array& x,
    const array& code_tile,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || code_tile.ndim() != 2 || scale_tile.ndim() != 2 ||
      scale_group_indices.ndim() != 1 || codeword_scale_slots.ndim() != 1 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_packed_rhs_tile_matmul expects x [M,BK], code_tile [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int BN = code_tile.shape(0);
  int codewords = code_tile.shape(1);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_packed_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (codewords != K / 8 || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_packed_rhs_tile_matmul input shapes do not match BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PPackedRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(code_tile, mlx::core::uint16, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
	       astype(codebook, mlx::core::uint32, stream)});
	}

array e8p_split_byte_rhs_tile_matmul(
    const array& x,
    const array& sign_tile,
    const array& abs_index_tile,
    const array& parity_tile,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || sign_tile.ndim() != 2 ||
      abs_index_tile.ndim() != 2 || parity_tile.ndim() != 2 ||
      scale_tile.ndim() != 2 || scale_group_indices.ndim() != 1 ||
      codeword_scale_slots.ndim() != 1 || codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_tile_matmul expects x [M,BK], sign/abs/parity tiles [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int BN = sign_tile.shape(0);
  int codewords = sign_tile.shape(1);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (codewords != K / 8 || abs_index_tile.shape(0) != BN ||
      abs_index_tile.shape(1) != codewords || parity_tile.shape(0) != BN ||
      parity_tile.shape(1) != codewords || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_tile_matmul input shapes do not match BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PSplitByteRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(sign_tile, mlx::core::uint8, stream),
       astype(abs_index_tile, mlx::core::uint8, stream),
       astype(parity_tile, mlx::core::uint8, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8p_sign_nibble_abs_index_rhs_tile_matmul(
    const array& x,
    const array& sign_low_nibble_tile,
    const array& sign_high_nibble_tile,
    const array& abs_index_tile,
    const array& parity_tile,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || sign_low_nibble_tile.ndim() != 2 ||
      sign_high_nibble_tile.ndim() != 2 || abs_index_tile.ndim() != 2 ||
      parity_tile.ndim() != 2 || scale_tile.ndim() != 2 ||
      scale_group_indices.ndim() != 1 || codeword_scale_slots.ndim() != 1 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_tile_matmul expects x [M,BK], sign low/high nibble, abs, and parity tiles [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int BN = sign_low_nibble_tile.shape(0);
  int codewords = sign_low_nibble_tile.shape(1);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (codewords != K / 8 || sign_high_nibble_tile.shape(0) != BN ||
      sign_high_nibble_tile.shape(1) != codewords ||
      abs_index_tile.shape(0) != BN ||
      abs_index_tile.shape(1) != codewords || parity_tile.shape(0) != BN ||
      parity_tile.shape(1) != codewords || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_tile_matmul input shapes do not match BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PSignNibbleAbsIndexRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(sign_low_nibble_tile, mlx::core::uint8, stream),
       astype(sign_high_nibble_tile, mlx::core::uint8, stream),
       astype(abs_index_tile, mlx::core::uint8, stream),
       astype(parity_tile, mlx::core::uint8, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8p_sign_plane_abs_index_rhs_tile_matmul(
    const array& x,
    const array& sign_bit_planes,
    const array& abs_index_tile,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || sign_bit_planes.ndim() != 2 ||
      abs_index_tile.ndim() != 2 || scale_tile.ndim() != 2 ||
      scale_group_indices.ndim() != 1 || codeword_scale_slots.ndim() != 1 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_tile_matmul expects x [M,BK], sign_bit_planes [BK/8,8], abs_index_tile [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int codewords = sign_bit_planes.shape(0);
  int sign_bits = sign_bit_planes.shape(1);
  int BN = abs_index_tile.shape(0);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (codewords != K / 8 || sign_bits != 8 ||
      abs_index_tile.shape(1) != codewords || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_tile_matmul input shapes do not match sign-plane BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PSignPlaneAbsIndexRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(sign_bit_planes, mlx::core::uint64, stream),
       astype(abs_index_tile, mlx::core::uint8, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8p_sign_nibble_micro_lut_rhs_tile_matmul(
    const array& x,
    const array& sign_low_nibble_lut,
    const array& sign_low_nibble_slots,
    const array& sign_high_nibble_lut,
    const array& sign_high_nibble_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || sign_low_nibble_lut.ndim() != 1 ||
      sign_low_nibble_slots.ndim() != 2 ||
      sign_high_nibble_lut.ndim() != 1 ||
      sign_high_nibble_slots.ndim() != 2 || abs_index_lut.ndim() != 1 ||
      abs_index_slots.ndim() != 2 || scale_tile.ndim() != 2 ||
      scale_group_indices.ndim() != 1 || codeword_scale_slots.ndim() != 1 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_tile_matmul expects x [M,BK], low/high nibble LUTs [16], low/high slots [BN,BK/8], abs LUT [256], abs slots [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int BN = sign_low_nibble_slots.shape(0);
  int codewords = sign_low_nibble_slots.shape(1);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (sign_low_nibble_lut.shape(0) != 16 ||
      sign_high_nibble_lut.shape(0) != 16 ||
      abs_index_lut.shape(0) != 256 || codewords != K / 8 ||
      sign_high_nibble_slots.shape(0) != BN ||
      sign_high_nibble_slots.shape(1) != codewords ||
      abs_index_slots.shape(0) != BN ||
      abs_index_slots.shape(1) != codewords || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_tile_matmul input shapes do not match micro-LUT BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PSignNibbleMicroLUTRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(sign_low_nibble_lut, mlx::core::uint8, stream),
       astype(sign_low_nibble_slots, mlx::core::uint8, stream),
       astype(sign_high_nibble_lut, mlx::core::uint8, stream),
       astype(sign_high_nibble_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8p_split_byte_factor_reuse_rhs_tile_matmul(
    const array& x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tile,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    int output_count,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x.ndim() != 2 || sign_byte_lut.ndim() != 1 ||
      sign_byte_slots.ndim() != 2 || abs_index_lut.ndim() != 1 ||
      abs_index_slots.ndim() != 2 || scale_tile.ndim() != 2 ||
      scale_group_indices.ndim() != 1 || codeword_scale_slots.ndim() != 1 ||
      codebook.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_tile_matmul expects x [M,BK], sign/abs LUTs [256], sign/abs slots [BN,BK/8], scale_tile [BN,G], scale_group_indices [G], codeword_scale_slots [BK/8], and codebook [256]");
  }
  int M = x.shape(0);
  int K = x.shape(1);
  int BN = sign_byte_slots.shape(0);
  int codewords = sign_byte_slots.shape(1);
  int scale_groups = scale_tile.shape(1);
  if (M <= 0 || K <= 0 || BN <= 0 || output_count <= 0 ||
      output_count > BN || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_tile_matmul requires positive M/K/BN/output_count and K divisible by 8");
  }
  if (sign_byte_lut.shape(0) != 256 || abs_index_lut.shape(0) != 256 ||
      codewords != K / 8 || abs_index_slots.shape(0) != BN ||
      abs_index_slots.shape(1) != codewords || scale_tile.shape(0) != BN ||
      scale_group_indices.shape(0) != scale_groups ||
      codeword_scale_slots.shape(0) != codewords || codebook.shape(0) != 256) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_tile_matmul input shapes do not match factor-reuse BK/codeword/scale layout");
  }
  auto stream = to_stream(s);
  return array(
      {M, output_count},
      float16,
      std::make_shared<NaxE8PSplitByteFactorReuseRHSTileMatmul>(
          stream, kernel_dir, output_count),
      {astype(x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tile, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream)});
}

array e8p_packed_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_matmul expects sorted_x [routes,K], packed RHS tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PPackedRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
	       astype(tile_counts, int32, stream)});
	}

array e8p_split_byte_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_tiles,
    const array& abs_index_tiles,
    const array& parity_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_tiles.ndim() != 5 ||
      abs_index_tiles.ndim() != 5 || parity_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_matmul expects sorted_x [routes,K], split RHS tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_tiles.shape(0);
  int n_tiles = sign_tiles.shape(1);
  int k_blocks = sign_tiles.shape(2);
  int bn = sign_tiles.shape(3);
  int codewords = sign_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape() != sign_tiles.shape() ||
      parity_tiles.shape() != sign_tiles.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSplitByteRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_tiles, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(parity_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_packed_rhs_sorted_tiled_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_matmul expects sorted_x [routes,K], packed RHS tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_matmul requires q2-like bn64/bk64 packed RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PPackedRHSSortedTiledMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_nibble_abs_index_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_low_nibble_tiles,
    const array& sign_high_nibble_tiles,
    const array& abs_index_tiles,
    const array& parity_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_low_nibble_tiles.ndim() != 5 ||
      sign_high_nibble_tiles.ndim() != 5 || abs_index_tiles.ndim() != 5 ||
      parity_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_matmul expects sorted_x [routes,K], sign-nibble/abs/parity RHS tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_low_nibble_tiles.shape(0);
  int n_tiles = sign_low_nibble_tiles.shape(1);
  int k_blocks = sign_low_nibble_tiles.shape(2);
  int bn = sign_low_nibble_tiles.shape(3);
  int codewords = sign_low_nibble_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_high_nibble_tiles.shape() != sign_low_nibble_tiles.shape() ||
      abs_index_tiles.shape() != sign_low_nibble_tiles.shape() ||
      parity_tiles.shape() != sign_low_nibble_tiles.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignNibbleAbsIndexRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_low_nibble_tiles, mlx::core::uint8, stream),
       astype(sign_high_nibble_tiles, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(parity_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_plane_abs_index_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_bit_planes,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_bit_planes.ndim() != 5 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_matmul expects sorted_x [routes,K], sign_bit_planes [E,n_tiles,k_blocks,codewords,8], abs_index_tiles [E,n_tiles,k_blocks,bn,codewords], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_bit_planes.shape(0);
  int n_tiles = sign_bit_planes.shape(1);
  int k_blocks = sign_bit_planes.shape(2);
  int codewords = sign_bit_planes.shape(3);
  int sign_bits = sign_bit_planes.shape(4);
  int bn = abs_index_tiles.shape(3);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || sign_bits != 8 || route_count <= 0 || K <= 0 ||
      output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_matmul requires positive packed tile dimensions, 8 sign bit planes, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_matmul input shapes do not match sign-plane K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignPlaneAbsIndexRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_bit_planes, mlx::core::uint64, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_nibble_micro_lut_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_low_nibble_lut,
    const array& sign_low_nibble_slots,
    const array& sign_high_nibble_lut,
    const array& sign_high_nibble_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_low_nibble_lut.ndim() != 1 ||
      sign_low_nibble_slots.ndim() != 5 ||
      sign_high_nibble_lut.ndim() != 1 ||
      sign_high_nibble_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_matmul expects sorted_x [routes,K], nibble LUTs [16], nibble/abs slots [E,n_tiles,k_blocks,bn,*], abs LUT [E,n_tiles,k_blocks,256], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_low_nibble_slots.shape(0);
  int n_tiles = sign_low_nibble_slots.shape(1);
  int k_blocks = sign_low_nibble_slots.shape(2);
  int bn = sign_low_nibble_slots.shape(3);
  int codewords = sign_low_nibble_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 ||
      sign_low_nibble_lut.shape(0) != 16 ||
      sign_high_nibble_lut.shape(0) != 16 || codebook.shape(0) != 256 ||
      sign_high_nibble_slots.shape() != sign_low_nibble_slots.shape() ||
      abs_index_slots.shape() != sign_low_nibble_slots.shape() ||
      abs_index_lut.shape(0) != E || abs_index_lut.shape(1) != n_tiles ||
      abs_index_lut.shape(2) != k_blocks || abs_index_lut.shape(3) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_matmul input shapes do not match micro-LUT K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignNibbleMicroLUTRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_low_nibble_lut, mlx::core::uint8, stream),
       astype(sign_low_nibble_slots, mlx::core::uint8, stream),
       astype(sign_high_nibble_lut, mlx::core::uint8, stream),
       astype(sign_high_nibble_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
    const array& sorted_x,
    const array& sign_low_nibble_tiles,
    const array& sign_high_nibble_tiles,
    const array& abs_index_tiles,
    const array& parity_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_low_nibble_tiles.ndim() != 5 ||
      sign_high_nibble_tiles.ndim() != 5 || abs_index_tiles.ndim() != 5 ||
      parity_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul expects sorted_x [routes,K], sign-nibble/abs/parity RHS tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_low_nibble_tiles.shape(0);
  int n_tiles = sign_low_nibble_tiles.shape(1);
  int k_blocks = sign_low_nibble_tiles.shape(2);
  int bn = sign_low_nibble_tiles.shape(3);
  int codewords = sign_low_nibble_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul requires q2-like bn64/bk64 sign-nibble RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_high_nibble_tiles.shape() != sign_low_nibble_tiles.shape() ||
      abs_index_tiles.shape() != sign_low_nibble_tiles.shape() ||
      parity_tiles.shape() != sign_low_nibble_tiles.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul input shapes do not match sign-nibble K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_low_nibble_tiles, mlx::core::uint8, stream),
       astype(sign_high_nibble_tiles, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(parity_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
      astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
    const array& sorted_x,
    const array& sign_bit_planes,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_bit_planes.ndim() != 5 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul expects sorted_x [routes,K], sign_bit_planes [E,n_tiles,k_blocks,codewords,8], abs_index_tiles [E,n_tiles,k_blocks,bn,codewords], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_bit_planes.shape(0);
  int n_tiles = sign_bit_planes.shape(1);
  int k_blocks = sign_bit_planes.shape(2);
  int codewords = sign_bit_planes.shape(3);
  int sign_bits = sign_bit_planes.shape(4);
  int bn = abs_index_tiles.shape(3);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || sign_bits != 8 || route_count <= 0 || K <= 0 ||
      output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul requires positive packed tile dimensions, 8 sign bit planes, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul requires q2-like bn64/bk64 sign-plane RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul input shapes do not match sign-plane K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_bit_planes, mlx::core::uint64, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
    const array& sorted_x,
    const array& sign_low_nibble_lut,
    const array& sign_low_nibble_slots,
    const array& sign_high_nibble_lut,
    const array& sign_high_nibble_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_low_nibble_lut.ndim() != 1 ||
      sign_low_nibble_slots.ndim() != 5 ||
      sign_high_nibble_lut.ndim() != 1 ||
      sign_high_nibble_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul expects sorted_x [routes,K], nibble LUTs [16], nibble/abs slots [E,n_tiles,k_blocks,64,8], abs LUT [E,n_tiles,k_blocks,256], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_low_nibble_slots.shape(0);
  int n_tiles = sign_low_nibble_slots.shape(1);
  int k_blocks = sign_low_nibble_slots.shape(2);
  int bn = sign_low_nibble_slots.shape(3);
  int codewords = sign_low_nibble_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul requires q2-like bn64/bk64 sign-nibble micro-LUT RHS tiles");
  }
  if (K != k_blocks * codewords * 8 ||
      sign_low_nibble_lut.shape(0) != 16 ||
      sign_high_nibble_lut.shape(0) != 16 || codebook.shape(0) != 256 ||
      sign_high_nibble_slots.shape() != sign_low_nibble_slots.shape() ||
      abs_index_slots.shape() != sign_low_nibble_slots.shape() ||
      abs_index_lut.shape(0) != E || abs_index_lut.shape(1) != n_tiles ||
      abs_index_lut.shape(2) != k_blocks || abs_index_lut.shape(3) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul input shapes do not match micro-LUT K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_low_nibble_lut, mlx::core::uint8, stream),
       astype(sign_low_nibble_slots, mlx::core::uint8, stream),
       astype(sign_high_nibble_lut, mlx::core::uint8, stream),
       astype(sign_high_nibble_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_split_byte_factor_reuse_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 4 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_matmul expects sorted_x [routes,K], factor LUTs [E,n_tiles,k_blocks,256], slot tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int n_tiles = sign_byte_lut.shape(1);
  int k_blocks = sign_byte_lut.shape(2);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(3) != 256 || abs_index_lut.shape() != sign_byte_lut.shape() ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(1) != n_tiles ||
      sign_byte_slots.shape(2) != k_blocks || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn || scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_matmul input shapes do not match factor-reuse K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSplitByteFactorReuseRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 3 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 3 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_matmul expects sorted_x [routes,K], factor LUTs [E,k_blocks,256], slot tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int k_blocks = sign_byte_lut.shape(1);
  int n_tiles = sign_byte_slots.shape(1);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(2) != 256 || abs_index_lut.shape() != sign_byte_lut.shape() ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(2) != k_blocks ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_matmul input shapes do not match expert/K-block factor-reuse layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PExpertKBlockFactorReuseRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_component_stream_rhs_sorted_scalar_matmul(
    const array& sorted_x,
    const array& sign_component_bits,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& component_scale_slots,
    const array& component_codeword_indices,
    const array& component_offsets,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_component_bits.ndim() != 6 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      component_scale_slots.ndim() != 2 ||
      component_codeword_indices.ndim() != 2 || component_offsets.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_scalar_matmul expects sorted_x [routes,K], sign component bits [E,n_tiles,k_blocks,bn,codewords,8], abs/scale tiles, component maps, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_component_bits.shape(0);
  int n_tiles = sign_component_bits.shape(1);
  int k_blocks = sign_component_bits.shape(2);
  int bn = sign_component_bits.shape(3);
  int codewords = sign_component_bits.shape(4);
  int sign_bits = sign_component_bits.shape(5);
  int components = component_offsets.shape(1);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || sign_bits != 8 || components != codewords * 8 ||
      route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_scalar_matmul requires positive component-stream dimensions, 8 sign components per codeword, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks || abs_index_tiles.shape(3) != bn ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      component_scale_slots.shape(0) != k_blocks ||
      component_codeword_indices.shape() != component_scale_slots.shape() ||
      component_offsets.shape() != component_scale_slots.shape() ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_scalar_matmul input shapes do not match component-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PComponentStreamRHSSortedScalarMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_component_bits, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(component_scale_slots, int32, stream),
       astype(component_codeword_indices, int32, stream),
       astype(component_offsets, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_route_slot_codeword_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_slot_codeword_stream_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_slot_codeword_stream_rhs_sorted_matmul requires positive route-slot codeword-stream dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_route_slot_codeword_stream_rhs_sorted_matmul input shapes do not match route-slot codeword-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PRouteSlotCodewordStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul requires positive route-slot MMA codeword-tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul input shapes do not match route-slot MMA codeword-tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& active_route_tiles,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || active_route_tiles.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, 1D tile descriptors, and active route tiles");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int active_route_tile_count = active_route_tiles.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      active_route_tile_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul requires positive active route-tile codeword outer-product dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul input shapes do not match active route-tile codeword outer-product layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(active_route_tiles, int32, stream)});
}

array e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& expert_cohort_offsets,
    const array& expert_cohort_counts,
    const array& route_cohort_offsets,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || expert_cohort_offsets.ndim() != 1 ||
      expert_cohort_counts.ndim() != 1 || route_cohort_offsets.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and expert cohort descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int expert_cohort_count = route_cohort_offsets.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      expert_cohort_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul requires positive expert-cohort codeword-broadcast dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      expert_cohort_offsets.shape(0) != E ||
      expert_cohort_counts.shape() != expert_cohort_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul input shapes do not match expert-cohort codeword-broadcast layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(expert_cohort_offsets, int32, stream),
       astype(expert_cohort_counts, int32, stream),
       astype(route_cohort_offsets, int32, stream)});
}

array e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_batch_segment_offsets,
    const array& route_batch_segment_counts,
    const array& route_batch_route_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || route_batch_segment_offsets.ndim() != 1 ||
      route_batch_segment_counts.ndim() != 1 ||
      route_batch_route_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and route batch segment descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_batch_count = route_batch_segment_offsets.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_batch_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul requires positive route-batch codeword-reduce dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_batch_segment_counts.shape() != route_batch_segment_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul input shapes do not match route-batch segmented codeword-reduce layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_batch_segment_offsets, int32, stream),
       astype(route_batch_segment_counts, int32, stream),
       astype(route_batch_route_ids, int32, stream)});
}

array e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_cohort_offsets,
    const array& token_cohort_counts,
    const array& token_cohort_active_expert_ids,
    const array& token_cohort_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_cohort_offsets.ndim() != 1 ||
      token_cohort_counts.ndim() != 1 ||
      token_cohort_active_expert_ids.ndim() != 1 ||
      token_cohort_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_cohort_codeword_stream_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and token cohort descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_cohort_count = token_cohort_offsets.shape(0);
  int active_expert_count = token_cohort_active_expert_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      token_cohort_count <= 0 || active_expert_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_cohort_codeword_stream_rhs_sorted_matmul requires positive token-cohort codeword-stream dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_cohort_counts.shape() != token_cohort_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_cohort_codeword_stream_rhs_sorted_matmul input shapes do not match token-cohort codeword-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PTokenCohortCodewordStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_cohort_offsets, int32, stream),
       astype(token_cohort_counts, int32, stream),
       astype(token_cohort_active_expert_ids, int32, stream),
       astype(token_cohort_route_slot_ids, int32, stream)});
}

array e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_cohort_offsets,
    const array& token_cohort_counts,
    const array& token_cohort_active_expert_ids,
    const array& token_cohort_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_cohort_offsets.ndim() != 1 ||
      token_cohort_counts.ndim() != 1 ||
      token_cohort_active_expert_ids.ndim() != 1 ||
      token_cohort_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and token-cohort MMA codeword-tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_cohort_count = token_cohort_offsets.shape(0);
  int active_expert_count = token_cohort_active_expert_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      token_cohort_count <= 0 || active_expert_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul requires positive token-cohort MMA codeword-tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_cohort_counts.shape() != token_cohort_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul input shapes do not match token-cohort MMA codeword-tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_cohort_offsets, int32, stream),
       astype(token_cohort_counts, int32, stream),
       astype(token_cohort_active_expert_ids, int32, stream),
       astype(token_cohort_route_slot_ids, int32, stream)});
}

array e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& output_stationary_route_batch_offsets,
    const array& output_stationary_route_batch_counts,
    const array& output_stationary_route_batch_active_expert_ids,
    const array& output_stationary_route_batch_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      output_stationary_route_batch_offsets.ndim() != 1 ||
      output_stationary_route_batch_counts.ndim() != 1 ||
      output_stationary_route_batch_active_expert_ids.ndim() != 1 ||
      output_stationary_route_batch_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_output_stationary_codeword_tile_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and output-stationary route-batch descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_batch_count = output_stationary_route_batch_offsets.shape(0);
  int active_expert_count =
      output_stationary_route_batch_active_expert_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_batch_count <= 0 || active_expert_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_output_stationary_codeword_tile_rhs_sorted_matmul requires positive output-stationary codeword-tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      output_stationary_route_batch_counts.shape() !=
          output_stationary_route_batch_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_output_stationary_codeword_tile_rhs_sorted_matmul input shapes do not match output-stationary codeword-tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8POutputStationaryCodewordTileRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(output_stationary_route_batch_offsets, int32, stream),
       astype(output_stationary_route_batch_counts, int32, stream),
       astype(output_stationary_route_batch_active_expert_ids, int32, stream),
       astype(output_stationary_route_batch_route_slot_ids, int32, stream)});
}

array e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& input_stationary_route_batch_offsets,
    const array& input_stationary_route_batch_counts,
    const array& input_stationary_route_batch_active_expert_ids,
    const array& input_stationary_route_batch_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      input_stationary_route_batch_offsets.ndim() != 1 ||
      input_stationary_route_batch_counts.ndim() != 1 ||
      input_stationary_route_batch_active_expert_ids.ndim() != 1 ||
      input_stationary_route_batch_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_input_stationary_codeword_tile_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and input-stationary route-batch descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_batch_count = input_stationary_route_batch_offsets.shape(0);
  int active_expert_count =
      input_stationary_route_batch_active_expert_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_batch_count <= 0 || active_expert_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_input_stationary_codeword_tile_rhs_sorted_matmul requires positive input-stationary codeword-tile dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      input_stationary_route_batch_counts.shape() !=
          input_stationary_route_batch_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_input_stationary_codeword_tile_rhs_sorted_matmul input shapes do not match input-stationary codeword-tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PInputStationaryCodewordTileRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(input_stationary_route_batch_offsets, int32, stream),
       astype(input_stationary_route_batch_counts, int32, stream),
       astype(input_stationary_route_batch_active_expert_ids, int32, stream),
       astype(input_stationary_route_batch_route_slot_ids, int32, stream)});
}

array e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_factor_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_factor_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul expects sorted_x [routes,K], codeword factor tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale tiles, codeword scale slots, codebook, and route-tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int expert_count = codeword_factor_tiles.shape(0);
  int n_tiles = codeword_factor_tiles.shape(1);
  int k_block_count = codeword_factor_tiles.shape(2);
  int bn = codeword_factor_tiles.shape(3);
  int codeword_tile_count = codeword_factor_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (expert_count <= 0 || n_tiles <= 0 || k_block_count <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || route_count <= 0 || K <= 0 ||
      output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul requires positive expert/K-block codeword factor dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_block_count * codeword_tile_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != expert_count ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_block_count ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_block_count ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_block_count ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul input shapes do not match expert/K-block codeword factor layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_factor_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale tiles, codeword scale slots, codebook, and route-tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int expert_count = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_block_count = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_tile_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (expert_count <= 0 || n_tiles <= 0 || k_block_count <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul requires positive expert/K-block/scale-slot dimensions, routes, K, and in-range output_dims");
  }
  if (K != k_block_count * codeword_tile_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != expert_count ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_block_count ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_block_count ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_block_count ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul input shapes do not match expert/K-block scale-slot stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
    const array& route_local_codeword_dot_lut,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_codeword_lut_route_slots,
    const array& route_codeword_lut_offsets,
    const array& route_codeword_lut_counts,
    const array& route_codeword_lut_codeword_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (route_local_codeword_dot_lut.ndim() != 3 ||
      code_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      route_codeword_lut_route_slots.ndim() != 1 ||
      route_codeword_lut_offsets.ndim() != 1 ||
      route_codeword_lut_counts.ndim() != 1 ||
      route_codeword_lut_codeword_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_codeword_lut_accumulate_rhs_sorted_matmul expects route-local codeword dot LUT [routes,k_blocks,codewords], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, route-tile descriptors, and route-codeword LUT descriptors");
  }
  int route_count = route_local_codeword_dot_lut.shape(0);
  int k_block_count = route_local_codeword_dot_lut.shape(1);
  int route_codeword_lut_id_count = route_local_codeword_dot_lut.shape(2);
  int expert_count = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_block_count_from_code_tiles = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codeword_count = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_slot_count = route_codeword_lut_route_slots.shape(0);
  if (route_count <= 0 || k_block_count <= 0 || codeword_count <= 0 ||
      expert_count <= 0 || n_tiles <= 0 || bn <= 0 || scale_groups <= 0 ||
      route_slot_count <= 0 || route_codeword_lut_id_count <= 0 ||
      output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_codeword_lut_accumulate_rhs_sorted_matmul requires positive route-codeword LUT dimensions and in-range output_dims");
  }
  if (k_block_count_from_code_tiles != k_block_count ||
      scale_tiles.shape(0) != expert_count ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_block_count ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_block_count ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_block_count ||
      codeword_scale_slots.shape(1) != codeword_count ||
      route_codeword_lut_codeword_ids.shape(0) !=
          route_codeword_lut_id_count ||
      codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_codeword_lut_offsets.shape() !=
          route_codeword_lut_route_slots.shape() ||
      route_codeword_lut_counts.shape() !=
          route_codeword_lut_route_slots.shape()) {
    throw std::invalid_argument(
        "e8p_route_codeword_lut_accumulate_rhs_sorted_matmul input shapes do not match route-codeword LUT layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(route_local_codeword_dot_lut, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_codeword_lut_route_slots, int32, stream),
       astype(route_codeword_lut_offsets, int32, stream),
       astype(route_codeword_lut_counts, int32, stream),
       astype(route_codeword_lut_codeword_ids, int32, stream)});
}

array e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& rowwise_route_microtile_offsets,
    const array& rowwise_route_microtile_counts,
    const array& rowwise_route_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      rowwise_route_microtile_offsets.ndim() != 1 ||
      rowwise_route_microtile_counts.ndim() != 1 ||
      rowwise_route_microtile_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and rowwise route-microtile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_microtile_count = rowwise_route_microtile_offsets.shape(0);
  int route_microtile_slot_count = rowwise_route_microtile_route_slot_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_microtile_count <= 0 || route_microtile_slot_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul requires positive rowwise codeword-tile dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      rowwise_route_microtile_counts.shape() !=
          rowwise_route_microtile_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul input shapes do not match rowwise codeword-tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(rowwise_route_microtile_offsets, int32, stream),
       astype(rowwise_route_microtile_counts, int32, stream),
       astype(rowwise_route_microtile_route_slot_ids, int32, stream)});
}

array e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& output_tile_local_route_microtile_offsets,
    const array& output_tile_local_route_microtile_counts,
    const array& output_tile_local_route_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      output_tile_local_route_microtile_offsets.ndim() != 1 ||
      output_tile_local_route_microtile_counts.ndim() != 1 ||
      output_tile_local_route_microtile_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_output_tile_local_codeword_lut_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and output-tile-local route-microtile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_microtile_count =
      output_tile_local_route_microtile_offsets.shape(0);
  int route_microtile_slot_count =
      output_tile_local_route_microtile_route_slot_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_microtile_count <= 0 || route_microtile_slot_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_output_tile_local_codeword_lut_rhs_sorted_matmul requires positive output-tile-local codeword LUT dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      output_tile_local_route_microtile_counts.shape() !=
          output_tile_local_route_microtile_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_output_tile_local_codeword_lut_rhs_sorted_matmul input shapes do not match output-tile-local codeword LUT layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8POutputTileLocalCodewordLutRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(output_tile_local_route_microtile_offsets, int32, stream),
       astype(output_tile_local_route_microtile_counts, int32, stream),
       astype(output_tile_local_route_microtile_route_slot_ids, int32, stream)});
}

array e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_microtile_codeword_block_reduce_offsets,
    const array& route_microtile_codeword_block_reduce_counts,
    const array& route_microtile_codeword_block_reduce_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      route_microtile_codeword_block_reduce_offsets.ndim() != 1 ||
      route_microtile_codeword_block_reduce_counts.ndim() != 1 ||
      route_microtile_codeword_block_reduce_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and route-microtile codeword-block reduce descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_microtile_count =
      route_microtile_codeword_block_reduce_offsets.shape(0);
  int route_microtile_slot_count =
      route_microtile_codeword_block_reduce_route_slot_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      route_microtile_count <= 0 || route_microtile_slot_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul requires positive route-microtile codeword-block reduce dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_microtile_codeword_block_reduce_counts.shape() !=
          route_microtile_codeword_block_reduce_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul input shapes do not match route-microtile codeword-block reduce layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_microtile_codeword_block_reduce_offsets, int32, stream),
       astype(route_microtile_codeword_block_reduce_counts, int32, stream),
       astype(
           route_microtile_codeword_block_reduce_route_slot_ids,
           int32,
           stream)});
}

array e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& kblock_wavefront_codeword_scan_offsets,
    const array& kblock_wavefront_codeword_scan_counts,
    const array& kblock_wavefront_codeword_scan_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      kblock_wavefront_codeword_scan_offsets.ndim() != 1 ||
      kblock_wavefront_codeword_scan_counts.ndim() != 1 ||
      kblock_wavefront_codeword_scan_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and k-block wavefront descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codeword_groups = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_microtile_count = kblock_wavefront_codeword_scan_offsets.shape(0);
  int route_microtile_slot_count =
      kblock_wavefront_codeword_scan_route_slot_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_groups <= 0 || route_count <= 0 || K <= 0 ||
      output_dims <= 0 || route_microtile_count <= 0 ||
      route_microtile_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul requires positive k-block wavefront dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_groups * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_groups ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      kblock_wavefront_codeword_scan_counts.shape() !=
          kblock_wavefront_codeword_scan_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul input shapes do not match k-block wavefront layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(kblock_wavefront_codeword_scan_offsets, int32, stream),
       astype(kblock_wavefront_codeword_scan_counts, int32, stream),
       astype(
           kblock_wavefront_codeword_scan_route_slot_ids,
           int32,
           stream)});
}

array e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_route_output_stripe_offsets,
    const array& token_route_output_stripe_counts,
    const array& token_route_output_stripe_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      token_route_output_stripe_offsets.ndim() != 1 ||
      token_route_output_stripe_counts.ndim() != 1 ||
      token_route_output_stripe_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul expects sorted_x [routes,K], code/scale tiles [E,n_tiles,k_blocks,bn,*], codeword scale slots, codebook, tile descriptors, and token-route output-stripe descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codeword_stages = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_count = token_route_output_stripe_offsets.shape(0);
  int route_slot_count = token_route_output_stripe_route_slot_ids.shape(0);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_stages <= 0 || route_count <= 0 || K <= 0 ||
      output_dims <= 0 || token_count <= 0 || route_slot_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul requires positive token-route output-stripe dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_stages * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_stages ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_route_output_stripe_counts.shape() !=
          token_route_output_stripe_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul input shapes do not match token-route output-stripe layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_route_output_stripe_offsets, int32, stream),
       astype(token_route_output_stripe_counts, int32, stream),
       astype(
           token_route_output_stripe_route_slot_ids,
           int32,
           stream)});
}

array e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& scale_group_route_block_offsets,
    const array& scale_group_route_block_counts,
    const array& scale_group_route_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      scale_group_route_block_offsets.ndim() != 1 ||
      scale_group_route_block_counts.ndim() != 1 ||
      scale_group_route_block_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_scale_group_route_block_reduce_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale tiles, codeword scale slots, codebook, tile descriptors, and scale-group route-block descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_tile_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_block_count = scale_group_route_block_offsets.shape(0);
  int route_block_slot_count =
      scale_group_route_block_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_block_count <= 0 ||
      route_block_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_scale_group_route_block_reduce_rhs_sorted_matmul requires positive scale-group route-block dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_tile_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      scale_group_route_block_counts.shape() !=
          scale_group_route_block_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_scale_group_route_block_reduce_rhs_sorted_matmul input shapes do not match scale-group route-block reduce layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(scale_group_route_block_offsets, int32, stream),
       astype(scale_group_route_block_counts, int32, stream),
       astype(
           scale_group_route_block_route_slot_ids,
           int32,
           stream)});
}

array e8p_route_block_output_group_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_block_output_group_offsets,
    const array& route_block_output_group_counts,
    const array& route_block_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      route_block_output_group_offsets.ndim() != 1 ||
      route_block_output_group_counts.ndim() != 1 ||
      route_block_output_group_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_block_output_group_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and route-block output-group descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_block_count = route_block_output_group_offsets.shape(0);
  int route_block_slot_count =
      route_block_output_group_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_block_count <= 0 ||
      route_block_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_block_output_group_stream_rhs_sorted_matmul requires positive route-block output-group dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_block_output_group_counts.shape() !=
          route_block_output_group_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_route_block_output_group_stream_rhs_sorted_matmul input shapes do not match route-block output-group stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_block_output_group_offsets, int32, stream),
       astype(route_block_output_group_counts, int32, stream),
       astype(
           route_block_output_group_route_slot_ids,
           int32,
           stream)});
}

array e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& output_group_pretransposed_route_offsets,
    const array& output_group_pretransposed_route_counts,
    const array& output_group_pretransposed_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      output_group_pretransposed_route_offsets.ndim() != 1 ||
      output_group_pretransposed_route_counts.ndim() != 1 ||
      output_group_pretransposed_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and output-group-pretransposed route descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_block_count = output_group_pretransposed_route_offsets.shape(0);
  int route_slot_count = output_group_pretransposed_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_block_count <= 0 ||
      route_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul requires positive output-group-pretransposed dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      output_group_pretransposed_route_counts.shape() !=
          output_group_pretransposed_route_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul input shapes do not match output-group-pretransposed stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(output_group_pretransposed_route_offsets, int32, stream),
       astype(output_group_pretransposed_route_counts, int32, stream),
       astype(
           output_group_pretransposed_route_slot_ids,
           int32,
           stream)});
}

array e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& kblock_route_fused_offsets,
    const array& kblock_route_fused_counts,
    const array& kblock_route_fused_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || kblock_route_fused_offsets.ndim() != 1 ||
      kblock_route_fused_counts.ndim() != 1 ||
      kblock_route_fused_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and K-block route-fused descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_block_count = kblock_route_fused_offsets.shape(0);
  int route_slot_count = kblock_route_fused_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_block_count <= 0 ||
      route_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul requires positive K-block route-fused dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      kblock_route_fused_counts.shape() !=
          kblock_route_fused_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul input shapes do not match K-block route-fused stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(kblock_route_fused_offsets, int32, stream),
       astype(kblock_route_fused_counts, int32, stream),
       astype(kblock_route_fused_route_slot_ids, int32, stream)});
}

array e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_tile_output_swizzle_offsets,
    const array& route_tile_output_swizzle_counts,
    const array& route_tile_output_swizzle_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      route_tile_output_swizzle_offsets.ndim() != 1 ||
      route_tile_output_swizzle_counts.ndim() != 1 ||
      route_tile_output_swizzle_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and route-tile output-swizzle descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_tile_count = route_tile_output_swizzle_offsets.shape(0);
  int route_slot_count = route_tile_output_swizzle_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_tile_count <= 0 ||
      route_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul requires positive route-tile output-swizzle dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_tile_output_swizzle_counts.shape() !=
          route_tile_output_swizzle_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul input shapes do not match route-tile output-swizzle stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_tile_output_swizzle_offsets, int32, stream),
       astype(route_tile_output_swizzle_counts, int32, stream),
       astype(route_tile_output_swizzle_route_slot_ids, int32, stream)});
}

array e8p_token_topk_output_tile_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_topk_offsets,
    const array& token_topk_counts,
    const array& token_topk_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_topk_offsets.ndim() != 1 ||
      token_topk_counts.ndim() != 1 ||
      token_topk_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_topk_output_tile_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and token top-k output-tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_count = token_topk_offsets.shape(0);
  int token_topk_slot_count = token_topk_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_count <= 0 ||
      token_topk_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_topk_output_tile_stream_rhs_sorted_matmul requires positive token top-k output-tile dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_topk_counts.shape() != token_topk_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_topk_output_tile_stream_rhs_sorted_matmul input shapes do not match token top-k output-tile stream layout");
  }
  int topk_slot_capacity =
      (token_topk_slot_count + token_count - 1) / token_count;
  auto stream = to_stream(s);
  return array(
      {token_count, topk_slot_capacity, output_dims},
      float16,
      std::make_shared<NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_topk_offsets, int32, stream),
       astype(token_topk_counts, int32, stream),
       astype(token_topk_route_slot_ids, int32, stream)});
}

array e8p_token_block_output_group_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_block_offsets,
    const array& token_block_counts,
    const array& token_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_block_offsets.ndim() != 1 ||
      token_block_counts.ndim() != 1 ||
      token_block_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_block_output_group_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and token-block output-group descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_block_count = token_block_offsets.shape(0);
  int token_block_topk_slot_count = token_block_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_block_count <= 0 ||
      token_block_topk_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_block_output_group_stream_rhs_sorted_matmul requires positive token-block output-group dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_block_counts.shape() != token_block_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_block_output_group_stream_rhs_sorted_matmul input shapes do not match token-block output-group stream layout");
  }
  int topk_slot_capacity =
      (token_block_topk_slot_count + token_block_count - 1) / token_block_count;
  auto stream = to_stream(s);
  return array(
      {token_block_count, topk_slot_capacity, output_dims},
      float16,
      std::make_shared<NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_block_offsets, int32, stream),
       astype(token_block_counts, int32, stream),
       astype(token_block_route_slot_ids, int32, stream)});
}

array e8p_token_output_stripe_group_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_output_stripe_offsets,
    const array& token_output_stripe_counts,
    const array& token_output_stripe_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_output_stripe_offsets.ndim() != 1 ||
      token_output_stripe_counts.ndim() != 1 ||
      token_output_stripe_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_output_stripe_group_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, and token output-stripe group descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_count = token_output_stripe_offsets.shape(0);
  int token_output_stripe_slot_count =
      token_output_stripe_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_count <= 0 ||
      token_output_stripe_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_output_stripe_group_stream_rhs_sorted_matmul requires positive token output-stripe group dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_output_stripe_counts.shape() !=
          token_output_stripe_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_output_stripe_group_stream_rhs_sorted_matmul input shapes do not match token output-stripe group stream layout");
  }
  int topk_group_capacity =
      (token_output_stripe_slot_count + token_count - 1) / token_count;
  auto stream = to_stream(s);
  return array(
      {token_count, topk_group_capacity, output_dims},
      float16,
      std::make_shared<NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_output_stripe_offsets, int32, stream),
       astype(token_output_stripe_counts, int32, stream),
       astype(token_output_stripe_route_slot_ids, int32, stream)});
}

array e8p_token_expert_output_block_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_expert_output_block_offsets,
    const array& token_expert_output_block_counts,
    const array& token_expert_output_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      token_expert_output_block_offsets.ndim() != 2 ||
      token_expert_output_block_counts.ndim() != 2 ||
      token_expert_output_block_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_expert_output_block_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, token-expert offsets/counts [tokens,E], and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_count = token_expert_output_block_offsets.shape(0);
  int token_expert_output_block_slot_count =
      token_expert_output_block_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_count <= 0 ||
      token_expert_output_block_slot_count <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_expert_output_block_stream_rhs_sorted_matmul requires positive token-expert output-block dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_expert_output_block_offsets.shape(1) != experts ||
      token_expert_output_block_counts.shape() !=
          token_expert_output_block_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_expert_output_block_stream_rhs_sorted_matmul input shapes do not match token-expert output-block stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_count, experts, output_dims},
      float16,
      std::make_shared<NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_expert_output_block_offsets, int32, stream),
       astype(token_expert_output_block_counts, int32, stream),
       astype(token_expert_output_block_route_slot_ids, int32, stream)});
}

array e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_pair_kblock_offsets,
    const array& token_pair_kblock_counts,
    const array& token_pair_kblock_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || token_pair_kblock_offsets.ndim() != 2 ||
      token_pair_kblock_counts.ndim() != 2 ||
      token_pair_kblock_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, token-pair offsets/counts [token_pairs,E], and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = token_pair_kblock_offsets.shape(0);
  int token_pair_kblock_slot_count =
      token_pair_kblock_route_slot_ids.shape(0);
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      token_pair_kblock_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul requires positive token-pair K-block dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_pair_kblock_offsets.shape(1) != experts ||
      token_pair_kblock_counts.shape() != token_pair_kblock_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul input shapes do not match token-pair K-block accumulator stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, experts, output_dims},
      float16,
      std::make_shared<NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_pair_kblock_offsets, int32, stream),
       astype(token_pair_kblock_counts, int32, stream),
       astype(token_pair_kblock_route_slot_ids, int32, stream)});
}

array e8p_token_pair_output_group_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_pair_output_group_offsets,
    const array& token_pair_output_group_counts,
    const array& token_pair_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      token_pair_output_group_offsets.ndim() != 3 ||
      token_pair_output_group_counts.ndim() != 3 ||
      token_pair_output_group_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_output_group_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, token-pair output-group offsets/counts [token_pairs,output_groups,E], and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = token_pair_output_group_offsets.shape(0);
  int output_group_count = token_pair_output_group_offsets.shape(1);
  int token_pair_output_group_slot_count =
      token_pair_output_group_route_slot_ids.shape(0);
  int expected_output_group_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      output_group_count <= 0 || token_pair_output_group_slot_count <= 0 ||
      output_dims > n_tiles * bn ||
      output_group_count != expected_output_group_count) {
    throw std::invalid_argument(
        "e8p_token_pair_output_group_stream_rhs_sorted_matmul requires positive token-pair output-group dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_pair_output_group_offsets.shape(2) != experts ||
      token_pair_output_group_counts.shape() !=
          token_pair_output_group_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_output_group_stream_rhs_sorted_matmul input shapes do not match token-pair output-group stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, experts, output_dims},
      float16,
      std::make_shared<NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_pair_output_group_offsets, int32, stream),
       astype(token_pair_output_group_counts, int32, stream),
       astype(token_pair_output_group_route_slot_ids, int32, stream)});
}

array e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_pair_slot_topk_output_group_offsets,
    const array& token_pair_slot_topk_output_group_counts,
    const array& token_pair_slot_topk_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      token_pair_slot_topk_output_group_offsets.ndim() != 4 ||
      token_pair_slot_topk_output_group_counts.ndim() != 4 ||
      token_pair_slot_topk_output_group_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale tiles, codeword scale slots, codebook, tile descriptors, token-pair pair-slot top-k-slot output-group offsets/counts [token_pairs,pair_slots,topk_slots,output_groups], and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = token_pair_slot_topk_output_group_offsets.shape(0);
  int pair_slot_count = token_pair_slot_topk_output_group_offsets.shape(1);
  int topk_slot_count = token_pair_slot_topk_output_group_offsets.shape(2);
  int output_group_count = token_pair_slot_topk_output_group_offsets.shape(3);
  int token_pair_slot_topk_output_group_slot_count =
      token_pair_slot_topk_output_group_route_slot_ids.shape(0);
  int expected_output_group_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      pair_slot_count <= 0 || topk_slot_count <= 0 ||
      output_group_count <= 0 ||
      token_pair_slot_topk_output_group_slot_count <= 0 ||
      output_dims > n_tiles * bn ||
      output_group_count != expected_output_group_count) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul requires positive token-pair slot/top-k output-group dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_pair_slot_topk_output_group_counts.shape() !=
          token_pair_slot_topk_output_group_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul input shapes do not match token-pair slot/top-k output-group stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(token_pair_slot_topk_output_group_offsets, int32, stream),
       astype(token_pair_slot_topk_output_group_counts, int32, stream),
       astype(
           token_pair_slot_topk_output_group_route_slot_ids,
           int32,
           stream)});
}

array e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& token_pair_slot_topk_codeword_group_pipeline_offsets,
    const array& token_pair_slot_topk_codeword_group_pipeline_counts,
    const array& token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 ||
      token_pair_slot_topk_codeword_group_pipeline_offsets.ndim() != 5 ||
      token_pair_slot_topk_codeword_group_pipeline_counts.ndim() != 5 ||
      token_pair_slot_topk_codeword_group_pipeline_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul expects sorted_x [routes,K], codeword_tiles [E,n_tiles,k_blocks,bn,codeword_groups], scale_tiles, codeword_scale_slots, codebook, tile descriptors, token-pair pair-slot top-k-slot codeword-group output-stripe offsets/counts, and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_group_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count =
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(0);
  int pair_slot_count =
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(1);
  int topk_slot_count =
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(2);
  int codeword_pipeline_group_count =
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(3);
  int output_stripe_count =
      token_pair_slot_topk_codeword_group_pipeline_offsets.shape(4);
  int token_pair_slot_topk_codeword_group_pipeline_slot_count =
      token_pair_slot_topk_codeword_group_pipeline_route_slot_ids.shape(0);
  int expected_output_stripe_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_group_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      pair_slot_count <= 0 || topk_slot_count <= 0 ||
      codeword_pipeline_group_count <= 0 || output_stripe_count <= 0 ||
      token_pair_slot_topk_codeword_group_pipeline_slot_count <= 0 ||
      output_dims > n_tiles * bn ||
      output_stripe_count != expected_output_stripe_count ||
      codeword_pipeline_group_count != codeword_group_count) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul requires positive token-pair slot/top-k codeword-group output-stripe dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_group_count * 8 ||
      codebook.shape(0) != 256 || scale_tiles.shape(0) != experts ||
      scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_group_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      token_pair_slot_topk_codeword_group_pipeline_counts.shape() !=
          token_pair_slot_topk_codeword_group_pipeline_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul input shapes do not match token-pair slot/top-k codeword-group output-stripe layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(
           token_pair_slot_topk_codeword_group_pipeline_offsets,
           int32,
           stream),
       astype(
           token_pair_slot_topk_codeword_group_pipeline_counts,
           int32,
           stream),
       astype(
           token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
           int32,
           stream)});
}

array e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& scale_slot_broadcast_offsets,
    const array& scale_slot_broadcast_counts,
    const array& scale_slot_broadcast_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || scale_slot_broadcast_offsets.ndim() != 6 ||
      scale_slot_broadcast_counts.ndim() != 6 ||
      scale_slot_broadcast_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword_tiles [E,n_tiles,k_blocks,bn,codewords], scale_tiles, codeword_scale_slots, codebook, tile descriptors, token-pair pair-slot top-k-slot scale-slot output-stripe K-block offsets/counts, and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = scale_slot_broadcast_offsets.shape(0);
  int pair_slot_count = scale_slot_broadcast_offsets.shape(1);
  int topk_slot_count = scale_slot_broadcast_offsets.shape(2);
  int scale_slot_count = scale_slot_broadcast_offsets.shape(3);
  int output_stripe_count = scale_slot_broadcast_offsets.shape(4);
  int offset_k_blocks = scale_slot_broadcast_offsets.shape(5);
  int scale_slot_broadcast_slot_count =
      scale_slot_broadcast_route_slot_ids.shape(0);
  int expected_output_stripe_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      pair_slot_count <= 0 || topk_slot_count <= 0 ||
      scale_slot_count <= 0 || output_stripe_count <= 0 ||
      offset_k_blocks != k_blocks ||
      scale_slot_broadcast_slot_count <= 0 || output_dims > n_tiles * bn ||
      output_stripe_count != expected_output_stripe_count ||
      scale_slot_count > scale_groups) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul requires positive token-pair slot/top-k scale-slot output-stripe K-block dimensions, routes, K, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_count * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != experts || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      scale_slot_broadcast_counts.shape() !=
          scale_slot_broadcast_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul input shapes do not match token-pair slot/top-k scale-slot broadcast stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(scale_slot_broadcast_offsets, int32, stream),
       astype(scale_slot_broadcast_counts, int32, stream),
       astype(scale_slot_broadcast_route_slot_ids, int32, stream)});
}

array e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& route_bucket_offsets,
    const array& route_bucket_counts,
    const array& route_bucket_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || route_bucket_offsets.ndim() != 7 ||
      route_bucket_counts.ndim() != 7 ||
      route_bucket_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul expects sorted_x [routes,K], codeword_tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale_tiles, codeword_scale_slots, codebook, tile descriptors, route-bucket token-pair pair-slot top-k-slot K-block output-microtile codeword-tile offsets/counts, and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_tile_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int route_bucket_count = route_bucket_offsets.shape(0);
  int token_pair_count = route_bucket_offsets.shape(1);
  int pair_slot_count = route_bucket_offsets.shape(2);
  int topk_slot_count = route_bucket_offsets.shape(3);
  int offset_k_blocks = route_bucket_offsets.shape(4);
  int output_microtile_count = route_bucket_offsets.shape(5);
  int route_bucket_codeword_tile_count = route_bucket_offsets.shape(6);
  int route_bucket_slot_count = route_bucket_route_slot_ids.shape(0);
  int expected_output_microtile_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || route_bucket_count <= 0 ||
      token_pair_count <= 0 || pair_slot_count <= 0 ||
      topk_slot_count <= 0 || offset_k_blocks != k_blocks ||
      output_microtile_count <= 0 ||
      output_microtile_count != expected_output_microtile_count ||
      route_bucket_codeword_tile_count != codeword_tile_count ||
      route_bucket_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul requires positive route-bucket/token-pair slot/top-k K-block output-microtile codeword-tile dimensions, routes, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_tile_count * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != experts || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      route_bucket_counts.shape() != route_bucket_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul input shapes do not match route-bucket token-pair slot/top-k codeword-reduce layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(route_bucket_offsets, int32, stream),
       astype(route_bucket_counts, int32, stream),
       astype(route_bucket_route_slot_ids, int32, stream)});
}

array e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& kblock_microtile_offsets,
    const array& kblock_microtile_counts,
    const array& kblock_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || kblock_microtile_offsets.ndim() != 5 ||
      kblock_microtile_counts.ndim() != 5 ||
      kblock_microtile_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword_tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale_tiles, codeword_scale_slots, codebook, tile descriptors, token-pair pair-slot top-k-slot K-block output-microtile offsets/counts, and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_tile_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = kblock_microtile_offsets.shape(0);
  int pair_slot_count = kblock_microtile_offsets.shape(1);
  int topk_slot_count = kblock_microtile_offsets.shape(2);
  int offset_k_blocks = kblock_microtile_offsets.shape(3);
  int output_microtile_count = kblock_microtile_offsets.shape(4);
  int kblock_microtile_slot_count = kblock_microtile_route_slot_ids.shape(0);
  int expected_output_microtile_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      pair_slot_count <= 0 || topk_slot_count <= 0 ||
      offset_k_blocks != k_blocks || output_microtile_count <= 0 ||
      output_microtile_count != expected_output_microtile_count ||
      kblock_microtile_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul requires positive token-pair slot/top-k K-block output-microtile dimensions, routes, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_tile_count * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != experts || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      kblock_microtile_counts.shape() != kblock_microtile_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul input shapes do not match token-pair slot/top-k K-block microtile stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(kblock_microtile_offsets, int32, stream),
       astype(kblock_microtile_counts, int32, stream),
       astype(kblock_microtile_route_slot_ids, int32, stream)});
}

array e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul(
    const array& sorted_x,
    const array& codeword_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    const array& output_tile_fused_offsets,
    const array& output_tile_fused_counts,
    const array& output_tile_fused_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codeword_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1 || output_tile_fused_offsets.ndim() != 4 ||
      output_tile_fused_counts.ndim() != 4 ||
      output_tile_fused_route_slot_ids.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul expects sorted_x [routes,K], codeword_tiles [E,n_tiles,k_blocks,bn,codeword_tiles], scale_tiles, codeword_scale_slots, codebook, tile descriptors, token-pair pair-slot top-k-slot output-tile offsets/counts, and route slot ids");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int experts = codeword_tiles.shape(0);
  int n_tiles = codeword_tiles.shape(1);
  int k_blocks = codeword_tiles.shape(2);
  int bn = codeword_tiles.shape(3);
  int codeword_tile_count = codeword_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  int token_pair_count = output_tile_fused_offsets.shape(0);
  int pair_slot_count = output_tile_fused_offsets.shape(1);
  int topk_slot_count = output_tile_fused_offsets.shape(2);
  int output_tile_count = output_tile_fused_offsets.shape(3);
  int output_tile_fused_slot_count =
      output_tile_fused_route_slot_ids.shape(0);
  int expected_output_tile_count = (output_dims + 63) / 64;
  if (experts <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codeword_tile_count <= 0 || scale_groups <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || token_pair_count <= 0 ||
      pair_slot_count <= 0 || topk_slot_count <= 0 ||
      output_tile_count <= 0 ||
      output_tile_count != expected_output_tile_count ||
      output_tile_fused_slot_count <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul requires positive token-pair slot/top-k output-tile dimensions, routes, descriptors, and in-range output_dims");
  }
  if (K != k_blocks * codeword_tile_count * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != experts || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codeword_tile_count ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape() ||
      output_tile_fused_counts.shape() !=
          output_tile_fused_offsets.shape()) {
    throw std::invalid_argument(
        "e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul input shapes do not match token-pair slot/top-k output-tile fused stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {token_pair_count, pair_slot_count, topk_slot_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(codeword_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream),
       astype(output_tile_fused_offsets, int32, stream),
       astype(output_tile_fused_counts, int32, stream),
       astype(output_tile_fused_route_slot_ids, int32, stream)});
}

array e8p_component_stream_rhs_sorted_partial_matmul(
    const array& sorted_x,
    const array& sign_component_bits,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& component_scale_slots,
    const array& component_codeword_indices,
    const array& component_offsets,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_component_bits.ndim() != 6 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      component_scale_slots.ndim() != 2 ||
      component_codeword_indices.ndim() != 2 || component_offsets.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_partial_matmul expects sorted_x [routes,K], sign component bits [E,n_tiles,k_blocks,bn,codewords,8], abs/scale tiles, component maps, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_component_bits.shape(0);
  int n_tiles = sign_component_bits.shape(1);
  int k_blocks = sign_component_bits.shape(2);
  int bn = sign_component_bits.shape(3);
  int codewords = sign_component_bits.shape(4);
  int sign_bits = sign_component_bits.shape(5);
  int components = component_offsets.shape(1);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn != 64 ||
      codewords <= 0 || sign_bits != 8 || components != codewords * 8 ||
      route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_partial_matmul requires positive 64-wide component-stream dimensions, 8 sign components per codeword, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks || abs_index_tiles.shape(3) != bn ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      component_scale_slots.shape(0) != k_blocks ||
      component_codeword_indices.shape() != component_scale_slots.shape() ||
      component_offsets.shape() != component_scale_slots.shape() ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_partial_matmul input shapes do not match component-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PComponentStreamRHSSortedPartialMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_component_bits, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(component_scale_slots, int32, stream),
       astype(component_codeword_indices, int32, stream),
       astype(component_offsets, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_component_stream_rhs_sorted_tensorops_matmul(
    const array& sorted_x,
    const array& sign_component_bits,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& component_scale_slots,
    const array& component_codeword_indices,
    const array& component_offsets,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_component_bits.ndim() != 6 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      component_scale_slots.ndim() != 2 ||
      component_codeword_indices.ndim() != 2 || component_offsets.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_tensorops_matmul expects sorted_x [routes,K], sign component bits [E,n_tiles,k_blocks,bn,codewords,8], abs/scale tiles, component maps, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_component_bits.shape(0);
  int n_tiles = sign_component_bits.shape(1);
  int k_blocks = sign_component_bits.shape(2);
  int bn = sign_component_bits.shape(3);
  int codewords = sign_component_bits.shape(4);
  int sign_bits = sign_component_bits.shape(5);
  int components = component_offsets.shape(1);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn != 64 ||
      codewords <= 0 || sign_bits != 8 || components != codewords * 8 ||
      route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_tensorops_matmul requires positive 64-wide component-stream dimensions, 8 sign components per codeword, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks || abs_index_tiles.shape(3) != bn ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      component_scale_slots.shape(0) != k_blocks ||
      component_codeword_indices.shape() != component_scale_slots.shape() ||
      component_offsets.shape() != component_scale_slots.shape() ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_tensorops_matmul input shapes do not match component-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PComponentStreamRHSSortedTensorOpsMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_component_bits, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(component_scale_slots, int32, stream),
       astype(component_codeword_indices, int32, stream),
       astype(component_offsets, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_component_stream_rhs_sorted_shared_decode_matmul(
    const array& sorted_x,
    const array& sign_component_bits,
    const array& abs_index_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& component_scale_slots,
    const array& component_codeword_indices,
    const array& component_offsets,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_component_bits.ndim() != 6 ||
      abs_index_tiles.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      component_scale_slots.ndim() != 2 ||
      component_codeword_indices.ndim() != 2 || component_offsets.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_shared_decode_matmul expects sorted_x [routes,K], sign component bits [E,n_tiles,k_blocks,bn,codewords,8], abs/scale tiles, component maps, codebook, and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_component_bits.shape(0);
  int n_tiles = sign_component_bits.shape(1);
  int k_blocks = sign_component_bits.shape(2);
  int bn = sign_component_bits.shape(3);
  int codewords = sign_component_bits.shape(4);
  int sign_bits = sign_component_bits.shape(5);
  int components = component_offsets.shape(1);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn != 64 ||
      codewords <= 0 || sign_bits != 8 || components != codewords * 8 ||
      route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_shared_decode_matmul requires positive 64-wide component-stream dimensions, 8 sign components per codeword, routes, K, and in-range output_dims");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape(0) != E || abs_index_tiles.shape(1) != n_tiles ||
      abs_index_tiles.shape(2) != k_blocks || abs_index_tiles.shape(3) != bn ||
      abs_index_tiles.shape(4) != codewords || scale_tiles.shape(0) != E ||
      scale_tiles.shape(1) != n_tiles || scale_tiles.shape(2) != k_blocks ||
      scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      component_scale_slots.shape(0) != k_blocks ||
      component_codeword_indices.shape() != component_scale_slots.shape() ||
      component_offsets.shape() != component_scale_slots.shape() ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_component_stream_rhs_sorted_shared_decode_matmul input shapes do not match component-stream layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PComponentStreamRHSSortedSharedDecodeMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_component_bits, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(component_scale_slots, int32, stream),
       astype(component_codeword_indices, int32, stream),
       astype(component_offsets, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 3 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 3 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul expects sorted_x [routes,K], factor LUTs [E,k_blocks,256], slot tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int k_blocks = sign_byte_lut.shape(1);
  int n_tiles = sign_byte_slots.shape(1);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(2) != 256 || abs_index_lut.shape() != sign_byte_lut.shape() ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(2) != k_blocks ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul input shapes do not match expert/K-block factor-reuse layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 3 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 3 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul expects sorted_x [routes,K], factor LUTs [E,k_blocks,256], slot tiles [E,n_tiles,k_blocks,bn,*], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int k_blocks = sign_byte_lut.shape(1);
  int n_tiles = sign_byte_slots.shape(1);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(2) != 256 || abs_index_lut.shape() != sign_byte_lut.shape() ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(2) != k_blocks ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul input shapes do not match expert/K-block factor-reuse layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_split_byte_rhs_sorted_tiled_matmul(
    const array& sorted_x,
    const array& sign_tiles,
    const array& abs_index_tiles,
    const array& parity_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_tiles.ndim() != 5 ||
      abs_index_tiles.ndim() != 5 || parity_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_tiled_matmul expects sorted_x [routes,K], split RHS tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_tiles.shape(0);
  int n_tiles = sign_tiles.shape(1);
  int k_blocks = sign_tiles.shape(2);
  int bn = sign_tiles.shape(3);
  int codewords = sign_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_tiled_matmul requires positive split-byte tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_tiled_matmul requires q2-like bn64/bk64 split-byte RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      abs_index_tiles.shape() != sign_tiles.shape() ||
      parity_tiles.shape() != sign_tiles.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_rhs_sorted_tiled_matmul input shapes do not match split-byte K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSplitByteRHSSortedTiledMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_tiles, mlx::core::uint8, stream),
       astype(abs_index_tiles, mlx::core::uint8, stream),
       astype(parity_tiles, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 4 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul expects sorted_x [routes,K], factor LUTs [E,n_tiles,k_blocks,256], slot tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int n_tiles = sign_byte_lut.shape(1);
  int k_blocks = sign_byte_lut.shape(2);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul requires q2-like bn64/bk64 factor-reuse RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(3) != 256 || abs_index_lut.shape() != sign_byte_lut.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(1) != n_tiles ||
      sign_byte_slots.shape(2) != k_blocks || abs_index_slots.shape() != sign_byte_slots.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul input shapes do not match factor-reuse K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 4 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul expects sorted_x [routes,K], factor LUTs [E,n_tiles,k_blocks,256], slot tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int n_tiles = sign_byte_lut.shape(1);
  int k_blocks = sign_byte_lut.shape(2);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul requires q2-like bn64/bk64 factor-reuse RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(3) != 256 ||
      abs_index_lut.shape() != sign_byte_lut.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(1) != n_tiles ||
      sign_byte_slots.shape(2) != k_blocks ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul input shapes do not match factor-reuse K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
    const array& sorted_x,
    const array& sign_byte_lut,
    const array& sign_byte_slots,
    const array& abs_index_lut,
    const array& abs_index_slots,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || sign_byte_lut.ndim() != 4 ||
      sign_byte_slots.ndim() != 5 || abs_index_lut.ndim() != 4 ||
      abs_index_slots.ndim() != 5 || scale_tiles.ndim() != 5 ||
      scale_group_indices.ndim() != 2 || codeword_scale_slots.ndim() != 2 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul expects sorted_x [routes,K], factor LUTs [E,n_tiles,k_blocks,256], slot tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = sign_byte_lut.shape(0);
  int n_tiles = sign_byte_lut.shape(1);
  int k_blocks = sign_byte_lut.shape(2);
  int bn = sign_byte_slots.shape(3);
  int codewords = sign_byte_slots.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || bn <= 0 ||
      codewords <= 0 || route_count <= 0 || K <= 0 || output_dims <= 0 ||
      output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul requires q2-like bn64/bk64 factor-reuse RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      sign_byte_lut.shape(3) != 256 ||
      abs_index_lut.shape() != sign_byte_lut.shape() ||
      sign_byte_slots.shape(0) != E || sign_byte_slots.shape(1) != n_tiles ||
      sign_byte_slots.shape(2) != k_blocks ||
      abs_index_slots.shape() != sign_byte_slots.shape() ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul input shapes do not match factor-reuse K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<
          NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(sign_byte_lut, mlx::core::uint8, stream),
       astype(sign_byte_slots, mlx::core::uint8, stream),
       astype(abs_index_lut, mlx::core::uint8, stream),
       astype(abs_index_slots, mlx::core::uint8, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_packed_rhs_sorted_tiled_m128_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_m128_matmul expects sorted_x [routes,K], packed RHS tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_m128_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_m128_matmul requires q2-like bn64/bk64 packed RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_m128_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PPackedRHSSortedTiledM128Matmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_packed_rhs_sorted_tiled_k128_matmul(
    const array& sorted_x,
    const array& code_tiles,
    const array& scale_tiles,
    const array& scale_group_indices,
    const array& codeword_scale_slots,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || code_tiles.ndim() != 5 ||
      scale_tiles.ndim() != 5 || scale_group_indices.ndim() != 2 ||
      codeword_scale_slots.ndim() != 2 || codebook.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_k128_matmul expects sorted_x [routes,K], packed RHS tiles [E,n_tiles,k_blocks,64,8], scale maps [k_blocks,*], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = code_tiles.shape(0);
  int n_tiles = code_tiles.shape(1);
  int k_blocks = code_tiles.shape(2);
  int bn = code_tiles.shape(3);
  int codewords = code_tiles.shape(4);
  int scale_groups = scale_tiles.shape(4);
  if (E <= 0 || n_tiles <= 0 || k_blocks <= 0 || route_count <= 0 ||
      K <= 0 || output_dims <= 0 || output_dims > n_tiles * bn) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_k128_matmul requires positive packed tile dimensions, routes, K, and in-range output_dims");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_k128_matmul requires q2-like bn64/bk64 packed RHS tiles");
  }
  if (K != k_blocks * codewords * 8 || codebook.shape(0) != 256 ||
      scale_tiles.shape(0) != E || scale_tiles.shape(1) != n_tiles ||
      scale_tiles.shape(2) != k_blocks || scale_tiles.shape(3) != bn ||
      scale_group_indices.shape(0) != k_blocks ||
      scale_group_indices.shape(1) != scale_groups ||
      codeword_scale_slots.shape(0) != k_blocks ||
      codeword_scale_slots.shape(1) != codewords ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_packed_rhs_sorted_tiled_k128_matmul input shapes do not match packed K/N tile layout");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, output_dims},
      float16,
      std::make_shared<NaxE8PPackedRHSSortedTiledK128Matmul>(
          stream, kernel_dir, output_dims),
      {astype(sorted_x, float16, stream),
       astype(code_tiles, mlx::core::uint16, stream),
       astype(scale_tiles, float16, stream),
       astype(scale_group_indices, int32, stream),
       astype(codeword_scale_slots, int32, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_direct_reduce_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_direct_reduce_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_direct_reduce_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_direct_reduce_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_direct_reduce_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedDirectReduceMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_inline_b_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_inline_b_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_inline_b_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_inline_b_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_inline_b_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedInlineBMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_gs352_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_gs352_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,4], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K != 1408) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_gs352_matmul requires positive E/N/routes and Air down K=1408");
  }
  if (group_size != 352) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_gs352_matmul requires group_size=352");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != 4 || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_gs352_matmul input shapes do not match Air down K/group/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelGs352Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_lut_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& full_grid,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      full_grid.ndim() != 2 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_lut_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], full_grid [65536,8], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_lut_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_lut_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || full_grid.shape(0) != 65536 ||
      full_grid.shape(1) != 8 || tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_lut_matmul input shapes do not match K/group_size/full_grid/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelLutMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(full_grid, float16, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_tgcb_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelTgcbMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
	       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_tgscale_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgscale_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgscale_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgscale_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgscale_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelTgscaleMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_tgscale_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_tgscale_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_tgscale_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_tgscale_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelTgcbTgscaleMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_tgcb_hoist_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_hoist_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_hoist_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_hoist_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_tgcb_hoist_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelTgcbHoistMatmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_bk128_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_bk128_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_bk128_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_bk128_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_bk128_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelBk128Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
      astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m128n32_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m128n32_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m128n32_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m128n32_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m128n32_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM128N32Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m64n128_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n128_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n128_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n128_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n128_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM64N128Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m32n64_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM32N64Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m64n64t64_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n64t64_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n64t64_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n64t64_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m64n64t64_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM64N64T64Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m32n64t128_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64t128_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64t128_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64t128_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n64t128_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM32N64T128Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8p_fp16_sorted_steel_m32n128_matmul(
    const array& sorted_x,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (sorted_x.ndim() != 2 || codes.ndim() != 3 || scales.ndim() != 3 ||
      codebook.ndim() != 1 || tile_experts.ndim() != 1 ||
      tile_offsets.ndim() != 1 || tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n128_matmul expects sorted_x [routes,K], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D tile descriptors");
  }
  int route_count = sorted_x.shape(0);
  int K = sorted_x.shape(1);
  int E = codes.shape(0);
  int N = codes.shape(1);
  if (E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n128_matmul requires positive E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n128_matmul requires group_size divisible by 8 and K");
  }
  if (codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument(
        "e8p_fp16_sorted_steel_m32n128_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8PFp16SortedSteelM32N128Matmul>(
          stream, kernel_dir, group_size),
      {astype(sorted_x, float16, stream),
       astype(codes, mlx::core::uint16, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

array e8_int8_routed_matmul(
    const array& x_q,
    const array& x_scales,
    const array& codes,
    const array& scales,
    const array& codebook,
    const array& lhs_indices,
    const array& tile_experts,
    const array& tile_offsets,
    const array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    StreamOrDevice s) {
  if (x_q.ndim() != 2 || x_scales.ndim() != 2 || codes.ndim() != 3 ||
      scales.ndim() != 3 || codebook.ndim() != 1 || lhs_indices.ndim() != 1 ||
      tile_experts.ndim() != 1 || tile_offsets.ndim() != 1 ||
      tile_counts.ndim() != 1) {
    throw std::invalid_argument(
        "e8_int8_routed_matmul expects x_q [T,K], x_scales [T,K/group], codes [E,N,K/8], scales [E,N,K/group], codebook [256], and 1D route descriptors");
  }
  int K = x_q.shape(1);
  int T = x_q.shape(0);
  int E = codes.shape(0);
  int N = codes.shape(1);
  int route_count = lhs_indices.shape(0);
  if (T <= 0 || E <= 0 || N <= 0 || route_count <= 0 || K <= 0 || K % 8 != 0) {
    throw std::invalid_argument("e8_int8_routed_matmul requires positive T/E/N/routes/K and K divisible by 8");
  }
  if (group_size <= 0 || group_size % 8 != 0 || K % group_size != 0) {
    throw std::invalid_argument("e8_int8_routed_matmul requires group_size divisible by 8 and K");
  }
  if (x_scales.shape(0) != T || x_scales.shape(1) != K / group_size ||
      codes.shape(2) != K / 8 || scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || codebook.shape(0) != 256 ||
      tile_offsets.shape() != tile_experts.shape() ||
      tile_counts.shape() != tile_experts.shape()) {
    throw std::invalid_argument("e8_int8_routed_matmul input shapes do not match K/group_size/descriptors");
  }
  auto stream = to_stream(s);
  return array(
      {route_count, N},
      float16,
      std::make_shared<NaxE8Int8RoutedMatmul>(stream, kernel_dir, group_size),
      {astype(x_q, int8, stream),
       astype(x_scales, float16, stream),
       astype(codes, mlx::core::uint8, stream),
       astype(scales, float16, stream),
       astype(codebook, mlx::core::uint32, stream),
       astype(lhs_indices, int32, stream),
       astype(tile_experts, int32, stream),
       astype(tile_offsets, int32, stream),
       astype(tile_counts, int32, stream)});
}

}  // namespace vqnax
