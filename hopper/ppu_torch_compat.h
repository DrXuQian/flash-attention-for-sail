#pragma once

// CUDAContextLight pulls cusparse.h, which in turn defines NVIDIA __half and
// __nv_bfloat16.  PPU CUTLASS deliberately uses HGGC's scalar definitions, so
// the two complete header families cannot coexist in this host translation
// unit.  FA3 only needs this one ATen declaration from CUDAContextLight.
#if defined(USE_PPU)
#include <cuda_runtime_api.h>
namespace at::cuda {
cudaDeviceProp* getCurrentDeviceProperties();
}  // namespace at::cuda
#else
#include <ATen/cuda/CUDAContextLight.h>
#endif
