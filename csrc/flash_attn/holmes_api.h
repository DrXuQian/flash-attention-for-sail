#ifdef FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
// Holmes compatibility entry points. Included after the v2.8.2 API definitions.
namespace holmes_mha {

size_t get_workspace_size(int b, int s_q, int s_k, int h, int h_kv, int d, int d_v,
                          bool is_generation_phase = false, int num_splits = 0,
                          bool compile_tune_mode = false) {
    TORCH_CHECK(b > 0 && s_q >= 0 && s_k >= 0 && h > 0 && h_kv > 0 && h % h_kv == 0,
                "Invalid Holmes FA2 workspace shape");
    TORCH_CHECK(d > 0 && d_v > 0, "Holmes FA2 requires positive head dimensions");
    // Holmes queries all backend budgets before selecting FA2/FA3/CUTLASS.
    // Unsupported FA2 dimensions must not abort a valid fallback compilation.
    // The raw forward entry still rejects unsupported execution shapes.
    if (d > 256 || d % 8 != 0 || d != d_v) return 0;
    TORCH_CHECK(num_splits >= 0 && num_splits <= 128, "Holmes FA2 splits must be in [0, 128]");
    (void)is_generation_phase;
    const size_t rows = holmes_checked_mul(holmes_checked_mul(size_t(b), size_t(h)), size_t(s_q));
    const size_t head_rounded = ((size_t(d) + (d <= 128 ? 31 : 63)) / (d <= 128 ? 32 : 64)) * (d <= 128 ? 32 : 64);
    int splits;
    if (compile_tune_mode) {
        // Compile-time shapes may be bounds: a smaller runtime shape can choose
        // more splits. Retain an upper bound rather than the bound shape's heuristic.
        splits = 128;
    } else {
        const bool swapped = s_q == 1 && h > h_kv;
        splits = holmes_select_splits(b, swapped ? h_kv : h,
            swapped ? h / h_kv : s_q, s_k, d, swapped,
            get_num_sm(get_current_device()), num_splits);
    }
    const size_t limit = holmes_workspace_limit();
    // A capped dynamic budget must cover any smaller shape; do not round it
    // down to the bound shape's last fitting split layout.
    TORCH_CHECK(holmes_workspace_bytes(rows, head_rounded, 1) <= limit,
                "Holmes FA2 workspace limit cannot hold LSE and RNG state");
    return std::min(holmes_workspace_bytes(rows, head_rounded, splits), limit);
}

inline void check_raw_layout(const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& strides, const void* ptr) {
    TORCH_CHECK(shape.size() == 4 && strides.size() == 4, "Holmes FA2 expects rank-4 shape and strides");
    size_t span = 0;
    bool empty = false;
    for (size_t i = 0; i < 4; ++i) {
        TORCH_CHECK(shape[i] >= 0 && shape[i] <= std::numeric_limits<int>::max(),
                    "Holmes FA2 dimension is outside the supported int range");
        TORCH_CHECK(strides[i] >= 0, "Holmes FA2 requires nonnegative strides");
        empty |= shape[i] == 0;
        if (shape[i] > 0) span = holmes_checked_add(span,
            holmes_checked_mul(size_t(shape[i] - 1), size_t(strides[i])));
    }
    TORCH_CHECK(shape[0] > 0 && shape[2] > 0 && shape[3] > 0,
                "Holmes FA2 requires positive batch, heads and head dimension");
    TORCH_CHECK(strides[3] == 1, "Holmes FA2 requires contiguous last dimension");
    TORCH_CHECK(empty || ptr != nullptr, "Holmes FA2 received a null tensor pointer");
    if (!empty) TORCH_CHECK(holmes_checked_mul(holmes_checked_add(span, 1), 2)
                           <= size_t(std::numeric_limits<int64_t>::max()),
                           "Holmes FA2 tensor span is too large");
    // Allocation capacity and lifetime remain the caller's responsibility.
}

template <typename T>
void mha_fwd_raw_impl(hggcStream_t hggcStream, T *devPtrQ,
                      const std::vector<int64_t> &q_shape,
                      const std::vector<int64_t> &q_strides, T *devPtrK,
                      const std::vector<int64_t> &k_shape,
                      const std::vector<int64_t> &k_strides, T *devPtrV,
                      const std::vector<int64_t> &v_shape,
                      const std::vector<int64_t> &v_strides, T *devPtrO,
                      const std::vector<int64_t> &out_shape,
                      const std::vector<int64_t> &out_strides, float scale,
                      bool is_causal, void *workspace_ptr,
                      size_t workspace_size) {
  HolmesWorkspaceScope workspace_scope(workspace_ptr, workspace_size);
  check_raw_layout(q_shape, q_strides, devPtrQ);
  check_raw_layout(k_shape, k_strides, devPtrK);
  check_raw_layout(v_shape, v_strides, devPtrV);
  check_raw_layout(out_shape, out_strides, devPtrO);
  int device_id = -1;
  auto err = hggcGetDevice(&device_id);
  TORCH_CHECK(err == hggcSuccess, "hggcGetDevice failed: ", int(err), " ", hggcGetErrorString(err));

  c10::Device device(c10::DeviceType::CUDA, device_id);
  c10::DeviceIndex device_index = device.index();

  // ensure fa2 run on the stream
  // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
  // CUDAStreamGuard switches to the stream's device and makes it the current
  // stream on that device. CUDAStreamGuard will also restore the current device
  // and stream when it's destroyed
  at::cuda::CUDAStreamGuard g(at::cuda::getStreamFromExternal(
      static_cast<hggcStream_t>(hggcStream), device_index));
  c10::optional<at::Tensor> alibi_slopes_ = c10::nullopt;

  auto data_type = at::kHalf;
  if constexpr (std::is_same<T, cutlass::bfloat16_t>::value)
    data_type = at::kBFloat16;
  at::Tensor q, k, v;
  // TensorMaker queries even empty CUDA pointers. A null pointer is permitted
  // only for a zero-element layout (checked above); borrow the workspace address
  // for its non-owning metadata wrapper without reserving or accessing bytes.
  if (!devPtrQ) devPtrQ = static_cast<T*>(workspace_ptr);
  if (!devPtrK) devPtrK = static_cast<T*>(workspace_ptr);
  if (!devPtrV) devPtrV = static_cast<T*>(workspace_ptr);
  if (!devPtrO) devPtrO = static_cast<T*>(workspace_ptr);

  q = at::from_blob(devPtrQ, q_shape, q_strides,
                    at::dtype(data_type).device(device));
  k = at::from_blob(devPtrK, k_shape, k_strides,
                    at::dtype(data_type).device(device));
  v = at::from_blob(devPtrV, v_shape, v_strides,
                    at::dtype(data_type).device(device));
  c10::optional<at::Tensor> out_ = at::from_blob(
      devPtrO, out_shape, out_strides, at::dtype(data_type).device(device));

  mha_fwd(q, k, v, out_, alibi_slopes_, 0.0f, scale, is_causal, -1, -1,
          0.0f, false, c10::nullopt);
  return;
}

template void mha_fwd_raw_impl(
    hggcStream_t hggcStream, cutlass::half_t *devPtrQ,
    const std::vector<int64_t> &q_shape, const std::vector<int64_t> &q_strides,
    cutlass::half_t *devPtrK, const std::vector<int64_t> &k_shape,
    const std::vector<int64_t> &k_strides, cutlass::half_t *devPtrV,
    const std::vector<int64_t> &v_shape, const std::vector<int64_t> &v_strides,
    cutlass::half_t *devPtrO, const std::vector<int64_t> &out_shape,
    const std::vector<int64_t> &out_strides, float scale, bool is_causal,
    void *workspace_ptr, size_t workspace_size);

template void mha_fwd_raw_impl(
    hggcStream_t hggcStream, cutlass::bfloat16_t *devPtrQ,
    const std::vector<int64_t> &q_shape, const std::vector<int64_t> &q_strides,
    cutlass::bfloat16_t *devPtrK, const std::vector<int64_t> &k_shape,
    const std::vector<int64_t> &k_strides, cutlass::bfloat16_t *devPtrV,
    const std::vector<int64_t> &v_shape, const std::vector<int64_t> &v_strides,
    cutlass::bfloat16_t *devPtrO, const std::vector<int64_t> &out_shape,
    const std::vector<int64_t> &out_strides, float scale, bool is_causal,
    void *workspace_ptr, size_t workspace_size);

void mha_fwd_raw(void *stream, void *devPtrQ,
                 const std::vector<int64_t> &q_shape,
                 const std::vector<int64_t> &q_strides, void *devPtrK,
                 const std::vector<int64_t> &k_shape,
                 const std::vector<int64_t> &k_strides, void *devPtrV,
                 const std::vector<int64_t> &v_shape,
                 const std::vector<int64_t> &v_strides, void *devPtrO,
                 const std::vector<int64_t> &out_shape,
                 const std::vector<int64_t> &out_strides, float scale,
                 bool is_causal, void *workspace_ptr, uint64_t workspace_size,
                 int32_t data_type) {
  hggcStream_t hggcStream = reinterpret_cast<hggcStream_t>(stream);
  constexpr int32_t kFloat16 = 0;
  constexpr int32_t kBFloat16 = 1;
  if (data_type == kFloat16) {
    auto devQ = reinterpret_cast<cutlass::half_t *>(devPtrQ);
    auto devK = reinterpret_cast<cutlass::half_t *>(devPtrK);
    auto devV = reinterpret_cast<cutlass::half_t *>(devPtrV);
    auto devO = reinterpret_cast<cutlass::half_t *>(devPtrO);
    mha_fwd_raw_impl(hggcStream, devQ, q_shape, q_strides, devK, k_shape,
                     k_strides, devV, v_shape, v_strides, devO, out_shape,
                     out_strides, scale, is_causal, workspace_ptr,
                     workspace_size);
  } else if (data_type == kBFloat16) {
    auto devQ = reinterpret_cast<cutlass::bfloat16_t *>(devPtrQ);
    auto devK = reinterpret_cast<cutlass::bfloat16_t *>(devPtrK);
    auto devV = reinterpret_cast<cutlass::bfloat16_t *>(devPtrV);
    auto devO = reinterpret_cast<cutlass::bfloat16_t *>(devPtrO);
    mha_fwd_raw_impl(hggcStream, devQ, q_shape, q_strides, devK, k_shape,
                     k_strides, devV, v_shape, v_strides, devO, out_shape,
                     out_strides, scale, is_causal, workspace_ptr,
                     workspace_size);
  } else {
    TORCH_CHECK(false, "Holmes FA2 unsupported data type: ", data_type);
  }
  return;
}
}  // namespace holmes_mha

#endif // FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
