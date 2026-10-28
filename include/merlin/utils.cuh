#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * Modified by Hygon Information Technology Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <hip/hip_cooperative_groups.h>
#include <stdarg.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include "hip/hip_fp16.h"
#include "debug.hpp"

using namespace cooperative_groups;
namespace cg = cooperative_groups;


namespace nv {
namespace merlin {

template <class S>
static __forceinline__ __device__ S device_nano() {
  S mclk;
  asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(mclk));
  return mclk;
}

inline void __cudaCheckError(const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            hipGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = hipDeviceSynchronize();
  if (hipSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, hipGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}
#define CudaCheckError() nv::merlin::__cudaCheckError(__FILE__, __LINE__)

static inline size_t SAFE_GET_GRID_SIZE(size_t N, int block_size) {
  return ((N) > std::numeric_limits<int>::max())
             ? (((1 << 30) - 1) / block_size + 1)
             : (((N)-1) / block_size + 1);
}

static inline int SAFE_GET_BLOCK_SIZE(int block_size, int device = -1) {
  hipDeviceProp_t_v2 prop;
  int current_device = device;
  if (current_device == -1) {
    CUDA_CHECK(hipGetDevice(&current_device));
  }
  CUDA_CHECK(hipGetDeviceProperties_v2(&prop, current_device));
  if (block_size > prop.maxThreadsPerBlock) {
    fprintf(stdout,
            "The requested block_size=%d exceeds the device limit, "
            "the maxThreadsPerBlock=%d will be applied.\n",
            block_size, prop.maxThreadsPerBlock);
  }
  return std::min(prop.maxThreadsPerBlock, block_size);
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__inline__ __device__ uint64_t Murmur3HashDevice(uint64_t const& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__inline__ __device__ int64_t Murmur3HashDevice(int64_t const& key) {
  uint64_t k = uint64_t(key);
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return int64_t(k);
}

__inline__ __device__ uint32_t Murmur3HashDevice(uint32_t const& key) {
  uint32_t k = key;
  k ^= k >> 16;
  k *= UINT32_C(0x85ebca6b);
  k ^= k >> 13;
  k *= UINT32_C(0xc2b2ae35);
  k ^= k >> 16;

  return k;
}

__inline__ __device__ int32_t Murmur3HashDevice(int32_t const& key) {
  uint32_t k = uint32_t(key);
  k ^= k >> 16;
  k *= UINT32_C(0x85ebca6b);
  k ^= k >> 13;
  k *= UINT32_C(0xc2b2ae35);
  k ^= k >> 16;

  return int32_t(k);
}

class CudaDeviceRestorer {
 public:
  CudaDeviceRestorer() { CUDA_CHECK(hipGetDevice(&dev_)); }
  ~CudaDeviceRestorer() { CUDA_CHECK(hipSetDevice(dev_)); }

 private:
  int dev_;
};

static inline int get_dev(const void* ptr) {
  hipPointerAttribute_t attr;
  CUDA_CHECK(hipPointerGetAttributes(&attr, ptr));
  int dev = -1;

#if DTKRT_VERSION >= 10000
  if (attr.type == hipMemoryTypeDevice)
#else
  if (attr.memoryType == hipMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}

static inline void switch_to_dev(const void* ptr) {
  int dev = get_dev(ptr);
  if (dev >= 0) {
    CUDA_CHECK(hipSetDevice(dev));
  }
}

static inline bool is_on_device(const void* ptr) {
  hipPointerAttribute_t attr;
  CUDA_CHECK(hipPointerGetAttributes(&attr, ptr));

#if DTKRT_VERSION >= 10000
  return (attr.type == hipMemoryTypeDevice);
#else
  return (attr.memoryType == hipMemoryTypeDevice);
#endif
}

template <typename TOUT, typename TIN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<__half, float> {
  static __forceinline__ __device__ __half convert(float val) {
    return __float2half(val);
  }
};

template <>
struct TypeConvertFunc<float, __half> {
  static __forceinline__ __device__ float convert(__half val) {
    return __half2float(val);
  }
};

template <>
struct TypeConvertFunc<float, float> {
  static __forceinline__ __device__ float convert(float val) { return val; }
};

template <>
struct TypeConvertFunc<float, long long> {
  static __forceinline__ __device__ float convert(long long val) {
    return static_cast<float>(val);
  }
};

template <>
struct TypeConvertFunc<float, unsigned int> {
  static __forceinline__ __device__ float convert(unsigned int val) {
    return static_cast<float>(val);
  }
};

template <>
struct TypeConvertFunc<int, long long> {
  static __forceinline__ __device__ int convert(long long val) {
    return static_cast<int>(val);
  }
};

template <>
struct TypeConvertFunc<int, unsigned int> {
  static __forceinline__ __device__ int convert(unsigned int val) {
    return static_cast<int>(val);
  }
};

template <typename mutex, uint32_t TILE_SIZE, bool THREAD_SAFE = true>
__forceinline__ __device__ void lock(
    const cg::thread_block_tile<TILE_SIZE>& tile, mutex& set_mutex,
    unsigned long long lane = 0) {
  if (THREAD_SAFE) {
    set_mutex.acquire(tile, lane);
  }
}

template <typename mutex, uint32_t TILE_SIZE, bool THREAD_SAFE = true>
__forceinline__ __device__ void unlock(
    const cg::thread_block_tile<TILE_SIZE>& tile, mutex& set_mutex,
    unsigned long long lane = 0) {
  if (THREAD_SAFE) {
    set_mutex.release(tile, lane);
  }
}

inline void free_pointers(hipStream_t stream, int n, ...) {
  va_list args;
  va_start(args, n);
  void* ptr = nullptr;
  for (int i = 0; i < n; i++) {
    ptr = va_arg(args, void*);
    if (ptr) {
      hipPointerAttribute_t attr;
      memset(&attr, 0, sizeof(hipPointerAttribute_t));
      try {
        CUDA_CHECK(hipPointerGetAttributes(&attr, ptr));
        if (attr.devicePointer && (!attr.hostPointer)) {
          CUDA_CHECK(hipFreeAsync(ptr, stream));
        } else if (attr.devicePointer && attr.hostPointer) {
          CUDA_CHECK(hipHostFree(ptr));
        } else {
          free(ptr);
        }
      } catch (const nv::merlin::CudaException& e) {
        va_end(args);
        throw e;
      }
    }
  }
  va_end(args);
}

static __global__ void memset64bitKernel(void* devPtr, uint64_t value,
                                         size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    static_cast<uint64_t*>(devPtr)[idx] = value;
  }
}

__forceinline__ __host__ hipError_t memset64Async(void* devPtr, uint64_t value,
                                                   size_t count,
                                                   hipStream_t stream = 0) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
 hipLaunchKernelGGL(( memset64bitKernel), dim3(numBlocks), dim3(blockSize), 0, stream, devPtr, value, count);
  return hipGetLastError();
}

#define CUDA_FREE_POINTERS(stream, ...) \
  nv::merlin::free_pointers(            \
      stream, (sizeof((void*[]){__VA_ARGS__}) / sizeof(void*)), __VA_ARGS__);

static inline size_t GB(size_t n) { return n << 30; }

static inline size_t MB(size_t n) { return n << 20; }

static inline size_t KB(size_t n) { return n << 10; }

constexpr inline bool ispow2(unsigned x) { return x && (!(x & (x - 1))); }

}  // namespace merlin
}  // namespace nv
