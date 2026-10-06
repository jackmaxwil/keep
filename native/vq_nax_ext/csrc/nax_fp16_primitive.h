#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <string>

namespace vqnax {

namespace mx = mlx::core;

class NaxFp16MatmulTile : public mx::Primitive {
 public:
  explicit NaxFp16MatmulTile(mx::Stream stream, std::string kernel_dir)
      : mx::Primitive(stream), kernel_dir_(std::move(kernel_dir)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxFp16MatmulTile: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxFp16MatmulTile";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxFp16MatmulTile&>(other);
    return kernel_dir_ == rhs.kernel_dir_;
  }

 private:
  std::string kernel_dir_;
};

class NaxE8Fp16MatmulTile : public mx::Primitive {
 public:
  explicit NaxE8Fp16MatmulTile(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Fp16MatmulTile: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Fp16MatmulTile";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Fp16MatmulTile&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8Fp16Matmul : public mx::Primitive {
 public:
  explicit NaxE8Fp16Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Fp16Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Fp16Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Fp16Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8Fp16RoutedMatmul : public mx::Primitive {
 public:
  explicit NaxE8Fp16RoutedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Fp16RoutedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Fp16RoutedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Fp16RoutedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8Fp16RoutedSteelMatmul : public mx::Primitive {
 public:
  explicit NaxE8Fp16RoutedSteelMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Fp16RoutedSteelMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Fp16RoutedSteelMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Fp16RoutedSteelMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8Fp16SortedSteelMatmul : public mx::Primitive {
 public:
  explicit NaxE8Fp16SortedSteelMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Fp16SortedSteelMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Fp16SortedSteelMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Fp16SortedSteelMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PPackedRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PPackedRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PPackedRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PPackedRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PPackedRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PSplitByteRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PSplitByteRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PSignNibbleAbsIndexRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleAbsIndexRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleAbsIndexRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleAbsIndexRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleAbsIndexRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PSignPlaneAbsIndexRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignPlaneAbsIndexRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignPlaneAbsIndexRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignPlaneAbsIndexRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignPlaneAbsIndexRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PSignNibbleMicroLUTRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleMicroLUTRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleMicroLUTRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleMicroLUTRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleMicroLUTRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PSplitByteFactorReuseRHSTileMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteFactorReuseRHSTileMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_count)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_count_(output_count) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteFactorReuseRHSTileMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteFactorReuseRHSTileMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteFactorReuseRHSTileMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_count_ == rhs.output_count_;
  }

 private:
  std::string kernel_dir_;
  int output_count_;
};

class NaxE8PPackedRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PPackedRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PPackedRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PPackedRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PPackedRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PSplitByteRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignNibbleAbsIndexRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleAbsIndexRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleAbsIndexRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleAbsIndexRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleAbsIndexRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignPlaneAbsIndexRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignPlaneAbsIndexRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignPlaneAbsIndexRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignPlaneAbsIndexRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignPlaneAbsIndexRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignNibbleMicroLUTRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleMicroLUTRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleMicroLUTRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleMicroLUTRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleMicroLUTRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteFactorReuseRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteFactorReuseRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteFactorReuseRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteFactorReuseRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteFactorReuseRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertKBlockFactorReuseRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PExpertKBlockFactorReuseRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertKBlockFactorReuseRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertKBlockFactorReuseRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PExpertKBlockFactorReuseRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PComponentStreamRHSSortedScalarMatmul : public mx::Primitive {
 public:
  explicit NaxE8PComponentStreamRHSSortedScalarMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PComponentStreamRHSSortedScalarMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PComponentStreamRHSSortedScalarMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PComponentStreamRHSSortedScalarMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteSlotCodewordStreamRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PRouteSlotCodewordStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteSlotCodewordStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteSlotCodewordStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PRouteSlotCodewordStreamRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul : public mx::Primitive {
 public:
  explicit NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PRouteSlotMMACodewordTileRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PActiveRouteTileCodewordOuterProductRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PExpertCohortCodewordBroadcastRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRouteBatchSegmentedCodewordReduceRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenCohortCodewordStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenCohortCodewordStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenCohortCodewordStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenCohortCodewordStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenCohortCodewordStreamRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenCohortMMACodewordTileRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8POutputStationaryCodewordTileRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8POutputStationaryCodewordTileRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8POutputStationaryCodewordTileRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8POutputStationaryCodewordTileRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8POutputStationaryCodewordTileRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PInputStationaryCodewordTileRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PInputStationaryCodewordTileRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PInputStationaryCodewordTileRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PInputStationaryCodewordTileRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PInputStationaryCodewordTileRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PExpertKBlockCodewordFactorReuseRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PExpertKBlockScaleSlotStreamRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRouteCodewordLutAccumulateRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRowwiseCodewordTileAccumulateRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8POutputTileLocalCodewordLutRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8POutputTileLocalCodewordLutRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8POutputTileLocalCodewordLutRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8POutputTileLocalCodewordLutRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8POutputTileLocalCodewordLutRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRouteMicrotileCodewordBlockReduceRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PKBlockWavefrontCodewordScanRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenRouteOutputStripePipelineRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PScaleGroupRouteBlockReduceRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRouteBlockOutputGroupStreamRHSSortedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8POutputGroupPretransposedCodewordStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PKBlockOutputGroupRouteFusedStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PRouteTileOutputSwizzleStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenTopKOutputTileStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenBlockOutputGroupStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenOutputStripeGroupStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenExpertOutputBlockStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairKBlockAccumulatorStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairOutputGroupStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkOutputGroupStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkCodewordGroupPipelineRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkScaleSlotBroadcastStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkRouteBucketCodewordReduceRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkKblockMicrotileStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<
        const NaxE8PTokenPairSlotTopkOutputTileFusedStreamRHSSortedMatmul&>(
        other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PComponentStreamRHSSortedPartialMatmul : public mx::Primitive {
 public:
  explicit NaxE8PComponentStreamRHSSortedPartialMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PComponentStreamRHSSortedPartialMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PComponentStreamRHSSortedPartialMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PComponentStreamRHSSortedPartialMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PComponentStreamRHSSortedTensorOpsMatmul : public mx::Primitive {
 public:
  explicit NaxE8PComponentStreamRHSSortedTensorOpsMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PComponentStreamRHSSortedTensorOpsMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PComponentStreamRHSSortedTensorOpsMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PComponentStreamRHSSortedTensorOpsMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PComponentStreamRHSSortedSharedDecodeMatmul : public mx::Primitive {
 public:
  explicit NaxE8PComponentStreamRHSSortedSharedDecodeMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PComponentStreamRHSSortedSharedDecodeMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PComponentStreamRHSSortedSharedDecodeMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PComponentStreamRHSSortedSharedDecodeMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul
    : public mx::Primitive {
 public:
  explicit NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PPackedRHSSortedTiledMatmul : public mx::Primitive {
 public:
  explicit NaxE8PPackedRHSSortedTiledMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PPackedRHSSortedTiledMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PPackedRHSSortedTiledMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PPackedRHSSortedTiledMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteRHSSortedTiledMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteRHSSortedTiledMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteRHSSortedTiledMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteRHSSortedTiledMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteRHSSortedTiledMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteFactorReuseRHSSortedTiledMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteFactorReuseRHSSortedSharedDecodeMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleAbsIndexRHSSortedTensorOpsMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignPlaneAbsIndexRHSSortedTensorOpsMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSignNibbleMicroLUTRHSSortedTensorOpsMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul
    : public mx::Primitive {
 public:
  explicit NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PSplitByteFactorReuseRHSSortedSharedNDecodeMatmul&>(
            other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PPackedRHSSortedTiledM128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PPackedRHSSortedTiledM128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PPackedRHSSortedTiledM128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PPackedRHSSortedTiledM128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PPackedRHSSortedTiledM128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PPackedRHSSortedTiledK128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PPackedRHSSortedTiledK128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int output_dims)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        output_dims_(output_dims) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PPackedRHSSortedTiledK128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PPackedRHSSortedTiledK128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PPackedRHSSortedTiledK128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && output_dims_ == rhs.output_dims_;
  }

 private:
  std::string kernel_dir_;
  int output_dims_;
};

class NaxE8PFp16SortedDirectReduceMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedDirectReduceMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedDirectReduceMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedDirectReduceMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedDirectReduceMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedInlineBMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedInlineBMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedInlineBMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedInlineBMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedInlineBMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelGs352Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelGs352Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedSteelGs352Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelGs352Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedSteelGs352Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelTgcbMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelTgcbMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelTgcbMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelTgcbMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelTgcbMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelTgscaleMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelTgscaleMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedSteelTgscaleMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelTgscaleMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedSteelTgscaleMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelTgcbTgscaleMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelTgcbTgscaleMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedSteelTgcbTgscaleMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelTgcbTgscaleMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedSteelTgcbTgscaleMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelTgcbHoistMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelTgcbHoistMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error(
        "NaxE8PFp16SortedSteelTgcbHoistMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelTgcbHoistMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs =
        static_cast<const NaxE8PFp16SortedSteelTgcbHoistMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelLutMatmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelLutMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelLutMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelLutMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelLutMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelBk128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelBk128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelBk128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelBk128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelBk128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM128N32Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM128N32Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM128N32Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM128N32Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM128N32Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM64N128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM64N128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM64N128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM64N128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM64N128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM64N64T64Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM64N64T64Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM64N64T64Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM64N64T64Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM64N64T64Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM32N64Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM32N64Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM32N64Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM32N64Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM32N64Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM32N64T128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM32N64T128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM32N64T128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM32N64T128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM32N64T128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8PFp16SortedSteelM32N128Matmul : public mx::Primitive {
 public:
  explicit NaxE8PFp16SortedSteelM32N128Matmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8PFp16SortedSteelM32N128Matmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8PFp16SortedSteelM32N128Matmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8PFp16SortedSteelM32N128Matmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

class NaxE8Int8RoutedMatmul : public mx::Primitive {
 public:
  explicit NaxE8Int8RoutedMatmul(
      mx::Stream stream,
      std::string kernel_dir,
      int group_size)
      : mx::Primitive(stream),
        kernel_dir_(std::move(kernel_dir)),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NaxE8Int8RoutedMatmul: CPU is not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "NaxE8Int8RoutedMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& rhs = static_cast<const NaxE8Int8RoutedMatmul&>(other);
    return kernel_dir_ == rhs.kernel_dir_ && group_size_ == rhs.group_size_;
  }

 private:
  std::string kernel_dir_;
  int group_size_;
};

mx::array fp16_matmul_tile(
    const mx::array& x,
    const mx::array& weight_t,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_fp16_matmul_tile(
    const mx::array& x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_fp16_matmul(
    const mx::array& x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_fp16_routed_matmul(
    const mx::array& x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& lhs_indices,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_fp16_routed_steel_matmul(
    const mx::array& x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& lhs_indices,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_fp16_sorted_steel_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_packed_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& code_tile,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& sign_tile,
    const mx::array& abs_index_tile,
    const mx::array& parity_tile,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_abs_index_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& sign_low_nibble_tile,
    const mx::array& sign_high_nibble_tile,
    const mx::array& abs_index_tile,
    const mx::array& parity_tile,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_plane_abs_index_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& sign_bit_planes,
    const mx::array& abs_index_tile,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_micro_lut_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& sign_low_nibble_lut,
    const mx::array& sign_low_nibble_slots,
    const mx::array& sign_high_nibble_lut,
    const mx::array& sign_high_nibble_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_factor_reuse_rhs_tile_matmul(
    const mx::array& x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tile,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    int output_count,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_packed_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_tiles,
    const mx::array& abs_index_tiles,
    const mx::array& parity_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_abs_index_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_low_nibble_tiles,
    const mx::array& sign_high_nibble_tiles,
    const mx::array& abs_index_tiles,
    const mx::array& parity_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_plane_abs_index_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_bit_planes,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_bit_planes,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_micro_lut_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_low_nibble_lut,
    const mx::array& sign_low_nibble_slots,
    const mx::array& sign_high_nibble_lut,
    const mx::array& sign_high_nibble_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_low_nibble_tiles,
    const mx::array& sign_high_nibble_tiles,
    const mx::array& abs_index_tiles,
    const mx::array& parity_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_low_nibble_lut,
    const mx::array& sign_low_nibble_slots,
    const mx::array& sign_high_nibble_lut,
    const mx::array& sign_high_nibble_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_factor_reuse_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_kblock_factor_reuse_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_component_stream_rhs_sorted_scalar_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_component_bits,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& component_scale_slots,
    const mx::array& component_codeword_indices,
    const mx::array& component_offsets,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_slot_codeword_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& active_route_tiles,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& expert_cohort_offsets,
    const mx::array& expert_cohort_counts,
    const mx::array& route_cohort_offsets,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_batch_segment_offsets,
    const mx::array& route_batch_segment_counts,
    const mx::array& route_batch_route_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_cohort_offsets,
    const mx::array& token_cohort_counts,
    const mx::array& token_cohort_active_expert_ids,
    const mx::array& token_cohort_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_cohort_offsets,
    const mx::array& token_cohort_counts,
    const mx::array& token_cohort_active_expert_ids,
    const mx::array& token_cohort_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& output_stationary_route_batch_offsets,
    const mx::array& output_stationary_route_batch_counts,
    const mx::array& output_stationary_route_batch_active_expert_ids,
    const mx::array& output_stationary_route_batch_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& input_stationary_route_batch_offsets,
    const mx::array& input_stationary_route_batch_counts,
    const mx::array& input_stationary_route_batch_active_expert_ids,
    const mx::array& input_stationary_route_batch_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_factor_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
    const mx::array& route_local_codeword_dot_lut,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_codeword_lut_route_slots,
    const mx::array& route_codeword_lut_offsets,
    const mx::array& route_codeword_lut_counts,
    const mx::array& route_codeword_lut_codeword_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& rowwise_route_microtile_offsets,
    const mx::array& rowwise_route_microtile_counts,
    const mx::array& rowwise_route_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& output_tile_local_route_microtile_offsets,
    const mx::array& output_tile_local_route_microtile_counts,
    const mx::array& output_tile_local_route_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_microtile_codeword_block_reduce_offsets,
    const mx::array& route_microtile_codeword_block_reduce_counts,
    const mx::array& route_microtile_codeword_block_reduce_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& kblock_wavefront_codeword_scan_offsets,
    const mx::array& kblock_wavefront_codeword_scan_counts,
    const mx::array& kblock_wavefront_codeword_scan_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_route_output_stripe_offsets,
    const mx::array& token_route_output_stripe_counts,
    const mx::array& token_route_output_stripe_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& scale_group_route_block_offsets,
    const mx::array& scale_group_route_block_counts,
    const mx::array& scale_group_route_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_block_output_group_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_block_output_group_offsets,
    const mx::array& route_block_output_group_counts,
    const mx::array& route_block_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& output_group_pretransposed_route_offsets,
    const mx::array& output_group_pretransposed_route_counts,
    const mx::array& output_group_pretransposed_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& kblock_route_fused_offsets,
    const mx::array& kblock_route_fused_counts,
    const mx::array& kblock_route_fused_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_tile_output_swizzle_offsets,
    const mx::array& route_tile_output_swizzle_counts,
    const mx::array& route_tile_output_swizzle_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_topk_output_tile_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_topk_offsets,
    const mx::array& token_topk_counts,
    const mx::array& token_topk_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_block_output_group_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_block_offsets,
    const mx::array& token_block_counts,
    const mx::array& token_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_output_stripe_group_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_output_stripe_offsets,
    const mx::array& token_output_stripe_counts,
    const mx::array& token_output_stripe_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_expert_output_block_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_expert_output_block_offsets,
    const mx::array& token_expert_output_block_counts,
    const mx::array& token_expert_output_block_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_pair_kblock_offsets,
    const mx::array& token_pair_kblock_counts,
    const mx::array& token_pair_kblock_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_output_group_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_pair_output_group_offsets,
    const mx::array& token_pair_output_group_counts,
    const mx::array& token_pair_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_pair_slot_topk_output_group_offsets,
    const mx::array& token_pair_slot_topk_output_group_counts,
    const mx::array& token_pair_slot_topk_output_group_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& token_pair_slot_topk_codeword_group_pipeline_offsets,
    const mx::array& token_pair_slot_topk_codeword_group_pipeline_counts,
    const mx::array& token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& scale_slot_broadcast_offsets,
    const mx::array& scale_slot_broadcast_counts,
    const mx::array& scale_slot_broadcast_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& route_bucket_offsets,
    const mx::array& route_bucket_counts,
    const mx::array& route_bucket_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& kblock_microtile_offsets,
    const mx::array& kblock_microtile_counts,
    const mx::array& kblock_microtile_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    const mx::array& output_tile_fused_offsets,
    const mx::array& output_tile_fused_counts,
    const mx::array& output_tile_fused_route_slot_ids,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
    const mx::array& sorted_x,
    const mx::array& codeword_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_component_stream_rhs_sorted_partial_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_component_bits,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& component_scale_slots,
    const mx::array& component_codeword_indices,
    const mx::array& component_offsets,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_component_stream_rhs_sorted_tensorops_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_component_bits,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& component_scale_slots,
    const mx::array& component_codeword_indices,
    const mx::array& component_offsets,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_component_stream_rhs_sorted_shared_decode_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_component_bits,
    const mx::array& abs_index_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& component_scale_slots,
    const mx::array& component_codeword_indices,
    const mx::array& component_offsets,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_packed_rhs_sorted_tiled_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_rhs_sorted_tiled_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_tiles,
    const mx::array& abs_index_tiles,
    const mx::array& parity_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
    const mx::array& sorted_x,
    const mx::array& sign_byte_lut,
    const mx::array& sign_byte_slots,
    const mx::array& abs_index_lut,
    const mx::array& abs_index_slots,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_packed_rhs_sorted_tiled_m128_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_packed_rhs_sorted_tiled_k128_matmul(
    const mx::array& sorted_x,
    const mx::array& code_tiles,
    const mx::array& scale_tiles,
    const mx::array& scale_group_indices,
    const mx::array& codeword_scale_slots,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int output_dims,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_direct_reduce_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_inline_b_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_gs352_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_lut_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& full_grid,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_tgcb_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_tgscale_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_tgcb_hoist_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_bk128_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m128n32_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m64n128_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m64n64t64_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m32n64_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m32n64t128_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8p_fp16_sorted_steel_m32n128_matmul(
    const mx::array& sorted_x,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

mx::array e8_int8_routed_matmul(
    const mx::array& x_q,
    const mx::array& x_scales,
    const mx::array& codes,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& lhs_indices,
    const mx::array& tile_experts,
    const mx::array& tile_offsets,
    const mx::array& tile_counts,
    int group_size,
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

}  // namespace vqnax
