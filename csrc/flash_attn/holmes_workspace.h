#ifdef FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
// Included inside FLASH_NAMESPACE. Scratch belongs to the caller, not the library.
inline constexpr size_t holmes_workspace_limit() {
    return size_t(2) << 30;
}
inline constexpr size_t kHolmesWorkspaceAlignment = 256;
inline size_t holmes_checked_mul(size_t a, size_t b) {
    TORCH_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                "Holmes FA2 workspace multiplication overflow");
    return a * b;
}
inline size_t holmes_checked_add(size_t a, size_t b) {
    TORCH_CHECK(a <= std::numeric_limits<size_t>::max() - b,
                "Holmes FA2 workspace addition overflow");
    return a + b;
}
inline size_t holmes_workspace_append(size_t offset, size_t bytes) {
    if (bytes == 0) return offset;
    offset = holmes_checked_add(offset, kHolmesWorkspaceAlignment - 1)
             & ~(kHolmesWorkspaceAlignment - 1);
    return holmes_checked_add(offset, bytes);
}
// Matches forward allocation order: LSE, split LSE, split output, RNG state.
inline size_t holmes_workspace_bytes(size_t rows, size_t head_rounded, int splits) {
    size_t offset = holmes_workspace_append(0, holmes_checked_mul(rows, sizeof(float)));
    if (splits > 1) {
        const size_t split_lse = holmes_checked_mul(
            holmes_checked_mul(rows, size_t(splits)), sizeof(float));
        offset = holmes_workspace_append(offset, split_lse);
        offset = holmes_workspace_append(offset, holmes_checked_mul(split_lse, head_rounded));
    }
    return holmes_workspace_append(offset, 2 * sizeof(int64_t));
}
inline int holmes_max_splits(size_t rows, size_t head_rounded, size_t capacity) {
    TORCH_CHECK(holmes_workspace_bytes(rows, head_rounded, 1) <= capacity,
                "Holmes FA2 workspace cannot hold LSE and RNG state");
    int splits = 128;
    while (splits > 1 && holmes_workspace_bytes(rows, head_rounded, splits) > capacity) --splits;
    return splits;
}
struct HolmesWorkspace {
    char* data;
    size_t size;
    size_t offset = 0;
};
inline thread_local HolmesWorkspace* holmes_workspace = nullptr;

struct HolmesWorkspaceScope {
    HolmesWorkspace storage;
    HolmesWorkspace* previous;
    HolmesWorkspaceScope(void* ptr, size_t bytes)
        : storage{static_cast<char*>(ptr), bytes}, previous(holmes_workspace) {
        TORCH_CHECK(ptr != nullptr && bytes >= 2 * sizeof(int64_t) && bytes != std::numeric_limits<size_t>::max(),
                    "Holmes FA2 requires a caller-owned workspace with an explicit capacity");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(ptr) % 16 == 0,
                    "Holmes FA2 workspace must be 16-byte aligned");
        // The caller must retain this allocation through every graph replay and
        // must not reuse it concurrently on streams without synchronization.
        holmes_workspace = &storage;
    }
    ~HolmesWorkspaceScope() { holmes_workspace = previous; }
    HolmesWorkspaceScope(const HolmesWorkspaceScope&) = delete;
};

inline at::Tensor holmes_empty(at::IntArrayRef shape, at::TensorOptions options) {
    TORCH_CHECK(holmes_workspace != nullptr, "Holmes FA2 requires an active workspace scope");
    size_t elements = 1;
    for (int64_t dim : shape) {
        TORCH_CHECK(dim >= 0, "Negative workspace tensor dimension");
        TORCH_CHECK(dim == 0 || elements <= std::numeric_limits<size_t>::max() / size_t(dim),
                    "Workspace tensor size overflow");
        elements *= size_t(dim);
    }
    // Empty tensors have no kernel storage and consume no arena capacity.
    if (elements == 0) return at::from_blob(holmes_workspace->data, shape, options);
    const size_t item_size = c10::elementSize(c10::typeMetaToScalarType(options.dtype()));
    TORCH_CHECK(elements <= std::numeric_limits<size_t>::max() / item_size,
                "Workspace byte size overflow");
    const size_t bytes = elements * item_size;
    auto& arena = *holmes_workspace;
    const size_t offset = holmes_checked_add(arena.offset, kHolmesWorkspaceAlignment - 1)
                          & ~(kHolmesWorkspaceAlignment - 1);
    TORCH_CHECK(offset <= arena.size && bytes <= arena.size - offset,
                "Holmes FA2 workspace too small: offset ", offset, " bytes ", bytes, " have ", arena.size);
    auto tensor = at::from_blob(arena.data + offset, shape, options);
    arena.offset = offset + bytes;
    return tensor;
}

#endif // FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
