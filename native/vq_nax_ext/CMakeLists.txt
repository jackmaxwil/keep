cmake_minimum_required(VERSION 3.27)

project(vq_nax_ext LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.12
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")
find_package(MLX CONFIG REQUIRED)

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pathlib, mlx.core as mx; p=pathlib.Path(mx.__file__).resolve(); print(next(parent / 'include' for parent in p.parents if (parent / 'include' / 'mlx' / 'backend' / 'metal' / 'kernels' / 'steel' / 'gemm' / 'nax.h').exists()))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_METAL_INCLUDE_DIR)

add_library(vqnax_prim_lib)

target_sources(
  vqnax_prim_lib
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/csrc/nax_fp16_primitive.mm)

target_include_directories(
  vqnax_prim_lib
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc)

target_compile_definitions(
  vqnax_prim_lib
  PRIVATE VQNAX_MLX_INCLUDE_DIR="${MLX_METAL_INCLUDE_DIR}")

target_link_libraries(
  vqnax_prim_lib
  PUBLIC
    mlx
    "-framework Metal"
    "-framework Foundation")

nanobind_add_module(
  _vqnax
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/csrc/bindings.cpp
)

target_link_libraries(_vqnax PRIVATE vqnax_prim_lib)

if(BUILD_SHARED_LIBS)
  target_link_options(_vqnax PRIVATE -Wl,-rpath,@loader_path)
endif()
