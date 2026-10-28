#ifndef __UTIL_H__
#define __UTIL_H__
#include <hip/hip_cooperative_groups.h>
// 接口补充
template <uint32_t TILE_SIZE>
__device__ __forceinline__ int tile_ffs(unsigned long long mask) {
  if constexpr (TILE_SIZE <= 32) {
    return __ffs(static_cast<unsigned>(mask));
  } else {
    return __ffsll(mask);
  }
}

#define SIMD_BYTE_MASK 0xff
#define SIMD_COMBINE_FOUR_DATA(b1, b2, b3, b4)                                 \
  (((unsigned int)((unsigned char)b4) << 24) |                                 \
   ((unsigned int)((unsigned char)b3) << 16) |                                 \
   ((unsigned int)((unsigned char)b2) << 8) |                                  \
   (unsigned int)((unsigned char)b1))


__device__ static inline unsigned int __vcmpeq4(unsigned int a,
                                                unsigned int b) {
  char abyte1 = a & SIMD_BYTE_MASK;
  char abyte2 = a >> 8 & SIMD_BYTE_MASK;
  char abyte3 = a >> 16 & SIMD_BYTE_MASK;
  char abyte4 = a >> 24 & SIMD_BYTE_MASK;
  char bbyte1 = b & SIMD_BYTE_MASK;
  char bbyte2 = b >> 8 & SIMD_BYTE_MASK;
  char bbyte3 = b >> 16 & SIMD_BYTE_MASK;
  char bbyte4 = b >> 24 & SIMD_BYTE_MASK;
  char byte1 = abyte1 == bbyte1 ? SIMD_BYTE_MASK : 0;
  char byte2 = abyte2 == bbyte2 ? SIMD_BYTE_MASK : 0;
  char byte3 = abyte3 == bbyte3 ? SIMD_BYTE_MASK : 0;
  char byte4 = abyte4 == bbyte4 ? SIMD_BYTE_MASK : 0;
  return SIMD_COMBINE_FOUR_DATA(byte1, byte2, byte3, byte4);
}

// pipeline 接口补充
#  define _CUDA_PIPELINE_STATIC_ASSERT(c, m) static_assert(c, m)
#  define _CUDA_PIPELINE_ASSERT(x) assert((x));
#  define _CUDA_PIPELINE_ABORT() assert(0);

template<size_t CopySize, size_t SourceSize>
inline __device__
void cp_async(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

    char* const d = reinterpret_cast<char*>(dst);
    const char* const s = reinterpret_cast<const char*>(src);

    size_t copy_step_size;

    if (SourceSize == 0) {
        copy_step_size = CopySize;
    } else if (SourceSize == 2 || SourceSize == 4 || SourceSize == 8 || SourceSize == 16) {
        copy_step_size = SourceSize;
    } else {
        copy_step_size = 1;
    }
    for (size_t i = 0; i < CopySize; i += copy_step_size) {
        const bool copy_source = SourceSize && (i < SourceSize);

        switch (copy_step_size) {
        case 1:
            d[i] = copy_source ? s[i] : char();
            break;
        case 2:
            *reinterpret_cast<short*>(d + i) = copy_source ? *reinterpret_cast<const short*>(s + i) : short();
            break;
        case 4:
            *reinterpret_cast<int*>(d + i) = copy_source ? *reinterpret_cast<const int*>(s + i) : int();
            break;
        case 8:
            *reinterpret_cast<int2*>(d + i) = copy_source ? *reinterpret_cast<const int2*>(s + i) : int2();
            break;
        case 16:
            *reinterpret_cast<int4*>(d + i) = copy_source ? *reinterpret_cast<const int4*>(s + i) : int4();
            break;
        }
    }
}

template<size_t CopySize, size_t SourceSize>
inline __device__
void pipeline_memcpy_async(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
    // _CUDA_PIPELINE_ASSERT(__isShared(dst));
    // _CUDA_PIPELINE_ASSERT(__isGlobal(src));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

    cp_async<CopySize, SourceSize>(dst, src);
}

static inline __device__
void __pipeline_memcpy_async(void* __restrict__ dst_shared, const void* __restrict__ src_global, size_t size_and_align,
                             size_t zfill = 0)
{
    _CUDA_PIPELINE_ASSERT(size_and_align == 4 || size_and_align == 8 || size_and_align == 16);
    _CUDA_PIPELINE_ASSERT(zfill <= size_and_align);
    // _CUDA_PIPELINE_ASSERT(__isShared(dst_shared));
    // _CUDA_PIPELINE_ASSERT(__isGlobal(src_global));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst_shared) & (size_and_align - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src_global) & (size_and_align - 1)));

    switch (size_and_align) {
    case 16:
        switch (zfill) {
        case  0: pipeline_memcpy_async<16, 16>(dst_shared, src_global); return;
        case  1: pipeline_memcpy_async<16, 15>(dst_shared, src_global); return;
        case  2: pipeline_memcpy_async<16, 14>(dst_shared, src_global); return;
        case  3: pipeline_memcpy_async<16, 13>(dst_shared, src_global); return;
        case  4: pipeline_memcpy_async<16, 12>(dst_shared, src_global); return;
        case  5: pipeline_memcpy_async<16, 11>(dst_shared, src_global); return;
        case  6: pipeline_memcpy_async<16, 10>(dst_shared, src_global); return;
        case  7: pipeline_memcpy_async<16,  9>(dst_shared, src_global); return;
        case  8: pipeline_memcpy_async<16,  8>(dst_shared, src_global); return;
        case  9: pipeline_memcpy_async<16,  7>(dst_shared, src_global); return;
        case 10: pipeline_memcpy_async<16,  6>(dst_shared, src_global); return;
        case 11: pipeline_memcpy_async<16,  5>(dst_shared, src_global); return;
        case 12: pipeline_memcpy_async<16,  4>(dst_shared, src_global); return;
        case 13: pipeline_memcpy_async<16,  3>(dst_shared, src_global); return;
        case 14: pipeline_memcpy_async<16,  2>(dst_shared, src_global); return;
        case 15: pipeline_memcpy_async<16,  1>(dst_shared, src_global); return;
        case 16: pipeline_memcpy_async<16,  0>(dst_shared, src_global); return;
        default: _CUDA_PIPELINE_ABORT();                                                                   return;
        }
    case 8:
        switch (zfill) {
        case  0: pipeline_memcpy_async< 8,  8>(dst_shared, src_global); return;
        case  1: pipeline_memcpy_async< 8,  7>(dst_shared, src_global); return;
        case  2: pipeline_memcpy_async< 8,  6>(dst_shared, src_global); return;
        case  3: pipeline_memcpy_async< 8,  5>(dst_shared, src_global); return;
        case  4: pipeline_memcpy_async< 8,  4>(dst_shared, src_global); return;
        case  5: pipeline_memcpy_async< 8,  3>(dst_shared, src_global); return;
        case  6: pipeline_memcpy_async< 8,  2>(dst_shared, src_global); return;
        case  7: pipeline_memcpy_async< 8,  1>(dst_shared, src_global); return;
        case  8: pipeline_memcpy_async< 8,  0>(dst_shared, src_global); return;
        default: _CUDA_PIPELINE_ABORT();                                                                   return;
        }
    case 4:
        switch (zfill) {
        case  0: pipeline_memcpy_async< 4,  4>(dst_shared, src_global); return;
        case  1: pipeline_memcpy_async< 4,  3>(dst_shared, src_global); return;
        case  2: pipeline_memcpy_async< 4,  2>(dst_shared, src_global); return;
        case  3: pipeline_memcpy_async< 4,  1>(dst_shared, src_global); return;
        case  4: pipeline_memcpy_async< 4,  0>(dst_shared, src_global); return;
        default: _CUDA_PIPELINE_ABORT();                                                                   return;
        }
    default:
        _CUDA_PIPELINE_ABORT();
        return;
    }
}

static inline __device__
void __pipeline_commit()
{

}

template<unsigned N>
static inline __device__
void pipeline_wait_prior()
{
    
}

static inline __device__
void __pipeline_wait_prior(size_t prior)
{
    switch (prior) {
    case  0 : pipeline_wait_prior<0>(); return;
    case  1 : pipeline_wait_prior<1>(); return;
    case  2 : pipeline_wait_prior<2>(); return;
    case  3 : pipeline_wait_prior<3>(); return;
    case  4 : pipeline_wait_prior<4>(); return;
    case  5 : pipeline_wait_prior<5>(); return;
    case  6 : pipeline_wait_prior<6>(); return;
    case  7 : pipeline_wait_prior<7>(); return;
    default : pipeline_wait_prior<8>(); return;
    }
}

#endif