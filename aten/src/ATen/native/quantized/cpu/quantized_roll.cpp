#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/TensorTransformations.h> // for roll_common
#include <ATen/NativeFunctions.h> // Need that for the `native_functions.yaml`
// #include <ATen/core/Type.h>
#include <ATen/native/TensorIterator.h>
// #include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {

// namespace {

inline void check_inputs(const Tensor& qa) {
  TORCH_CHECK(
      qa.scalar_type() == c10::kQInt8 || qa.scalar_type() == c10::kQUInt8,
      "quantized_roll operands should use QInt8 or QUInt8 data types.");
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine || qa.qscheme() == kPerTensorSymmetric,
      "Only per-tensor quantization is suported in quantized_roll.");
}

// Tensor quantized_roll(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
Tensor quantized_roll(const Tensor& self,  c10::ArrayRef<long> shifts, c10::ArrayRef<long> dims) {
  check_inputs(self);
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }
  // avoid a div zero error below.
  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Preserve);
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + size;
  }
  auto t0 = self.narrow(dim, start, size-start);
  auto t1 = self.narrow(dim, 0, start);
  return at::cat({t0, t1}, dim);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::quantized_roll"), TORCH_FN(quantized_roll));
}

// } // namespace

} // namespace native
} // namespace at
