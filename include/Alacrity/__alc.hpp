#pragma once

#define __ALC_EXIT_FAILURE 1

// Auxiliary
#ifndef SIZE_TYPE 
  #include <cstddef>
  #define SIZE_TYPE size_t
#endif
#ifndef SSIZE_TYPE
  #define SSIZE_TYPE long long // POSIX systems have ssize_t and should be checked for
#endif

// Convience Macros
#ifdef __CUDACC__ // being parsed by nvcc
  #define __ALC_CUDA
#elif defined(__HIP_PLATFORM_AMD__)
  #define __ALC_HIP
#else
  #define __ALC_HOST
#endif

// Device Specific
#include <stdio.h> // Wrap This

// Includes
#ifdef __ALC_CUDA
  #include <cuda_runtime.h>
#elif defined(__ALC_HIP)
  #include <hip_runtime.h>
#else
  #include <cstdlib> // malloc, free, realloc
  #include <cstring> // memcpy
#endif

// Function APIs
#ifdef __ALC_CUDA
  #define __ALC_GLOBAL_IDX blockIdx.x * blockDim.x + threadIdx.x
  #define __ALC_LOCAL_IDX threadIdx.x
  #define __ALC_BLOCK_IDX blockIdx.x
  #define __ALC_NEXT_IDX(i) (i) += blockDim.x * gridDim.x

  #define __ALC_KERNEL_API __global__
  #define __ALC_DEVICE_API __device__
  #define __ALC_HOST_API __host__
#elif defined(__ALC_HIP)
  #define __ALC_GLOBAL_IDX blockIdx.x * blockDim.x + threadIdx.x
  #define __ALC_LOCAL_IDX threadIdx.x
  #define __ALC_BLOCK_IDX blockIdx.x
  #define __ALC_NEXT_IDX(i) (i) += blockDim.x * gridDim.x

  #define __ALC_KERNEL_API __global__
  #define __ALC_DEVICE_API __device__
  #define __ALC_HOST_API __host__
#else
  namespace alc { extern SIZE_TYPE __impl_hidx; }
  #define __ALC_GLOBAL_IDX alc::__impl_hidx
  #define __ALC_LOCAL_IDX 0
  #define __ALC_BLOCK_IDX 0 // don't change anything for host
  #define __ALC_NEXT_IDX(i) ++(i)

  #define __ALC_KERNEL_API
  #define __ALC_DEVICE_API
  #define __ALC_HOST_API
#endif

#ifdef __ALC_CUDA
  #define __ALC_DEV_SYNC() __syncthreads()
#elif defined(__ALC_HIP)
  #define __ALC_DEV_SYNC() __syncthreads()
#else
  #define __ALC_DEV_SYNC() 
#endif


// Error-checking macros for CUDA/HIP
#ifdef __ALC_CUDA
  #define __ALC_TEST_DEVICE_ERROR()                               \
    {                                                             \
      cudaError_t err = cudaGetLastError();                       \
      if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                cudaGetErrorString(err), __LINE__, __FILE__);     \
        exit(__ALC_EXIT_FAILURE);                                 \
      }                                                           \
      cudaGetLastError();                                         \
    }
  #define __ALC_CHECK_DEVICE_ERROR(call)                          \
    {                                                             \
      cudaError_t err = (call);                                   \
      if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                cudaGetErrorString(err), __LINE__, __FILE__);     \
        exit(__ALC_EXIT_FAILURE);                                 \
      }                                                           \
      cudaGetLastError();                                         \
    }
#elif __ALC_HIP
  #define __ALC_TEST_DEVICE_ERROR()                              \
    {                                                            \
      hipError_t err = hipGetLastError();                        \
      if (err != hipSuccess) {                                   \
        fprintf(stderr, "HIP Error: %s at line %d in file %s\n", \
                hipGetErrorString(err), __LINE__, __FILE__);     \
        exit(__ALC_EXIT_FAILURE);                                \
      }                                                          \
      hipGetLastError();                                         \
    }
  #define __ALC_CHECK_DEVICE_ERROR(call)                         \
    {                                                            \
      hipError_t err = (call);                                   \
      if (err != hipSuccess) {                                   \
        fprintf(stderr, "HIP Error: %s at line %d in file %s\n", \
                hipGetErrorString(err), __LINE__, __FILE__);     \
        exit(__ALC_EXIT_FAILURE);                                \
      }                                                          \
      hipGetLastError();                                         \
    }
#else
  #define __ALC_TEST_DEVICE_ERROR() {}
  #define __ALC_CHECK_DEVICE_ERROR(call) {} // does nothing, assume host
#endif

// DEBUG MACROS
#ifdef __ALC_NOSAFETY // Don't perform any checks on anything
  #undef __ALC_CHECK_DEVICE_ERROR
  #define __ALC_CHECK_DEVICE_ERROR(call) {/* Do Nothing */}
  #define __ALC_NO_ASSERT
  #define __ALC_NO_EXIT
  #define __ALC_NO_USR_CHECKS
#elif defined(__ALC_RELEASE) // Don't perform any actions that could cause abnormal program termination
  #define __ALC_NO_ASSERT
  #define __ALC_NO_EXIT
  #define __ALC_NO_USR_CHECKS
#elif defined(__ALC_MID) // asserts
  #define __ALC_NO_EXIT
  #define __ALC_NO_USR_CHECKS
#elif defined(__ALC_DEBUG) // exits, asserts
  #define __ALC_NO_USR_CHECKS
#endif


namespace alc {
// Algorithm Auxiliary
  extern int __device;
  extern int __device_count;
  extern bool *__device_p2p;
  extern SIZE_TYPE* __device_threads;
  bool __p2p_capable(int device1, int device2);

#ifndef __ALC_CUSTOM_BLOCK_COUNT
  SIZE_TYPE block_count(SIZE_TYPE deviceId, SIZE_TYPE arrlen);
#endif

// Auxiliary
#ifdef __ALC_CUDA
  void __setup__() {
    // Get the number of available GPUs
    __ALC_CHECK_DEVICE_ERROR(cudaGetDeviceCount(&alc::__device_count));

    // Allocate memory for P2P availability between GPUs if more than one device
    if (alc::__device_count > 1) {
      alc::__device_p2p = new bool[alc::__device_count * alc::__device_count];
      alc::__device_threads = new SIZE_TYPE[alc::__device_count];

      for (int i = 0; i < alc::__device_count; i++) {
        // Set reasonable number of threads based on runtime information
        int max_threads_per_block;
        __ALC_CHECK_DEVICE_ERROR(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0));

        // Assign the maximum number of threads per block based on the device's capabilities
        alc::__device_threads[i] = max_threads_per_block > 0 ? max_threads_per_block : 1024;  // Fallback to 1024 if an error occurs

        // get p2p capability
        for (int j = 0; j < alc::__device_count; j++) {
          // Disable P2P by default if both devices are the same
          if (i == j) {
            alc::__device_p2p[i * alc::__device_count + j] = false;
          } else {
            int p2pCapable = 0;
            __ALC_CHECK_DEVICE_ERROR(cudaDeviceCanAccessPeer(&p2pCapable, i, j));
            alc::__device_p2p[i * alc::__device_count + j] = (p2pCapable == 1);

            if (p2pCapable) {
              __ALC_CHECK_DEVICE_ERROR(cudaSetDevice(i));
              __ALC_CHECK_DEVICE_ERROR(cudaDeviceEnablePeerAccess(j, 0));
            }
          }
        }
      }
    } else {
      alc::__device_threads = new SIZE_TYPE[1];
      // Set reasonable number of threads based on runtime information
      int max_threads_per_block;
      __ALC_CHECK_DEVICE_ERROR(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0));

      // Assign the maximum number of threads per block based on the device's capabilities
      alc::__device_threads[0] = max_threads_per_block > 0 ? max_threads_per_block : 1024;  // Fallback to 1024 if an error occurs

      alc::__device_p2p = nullptr;  // Only one device, no P2P needed
    }
  }
#elif __ALC_HIP
  void __setup__() {
    // Get the number of available GPUs
    __ALC_CHECK_DEVICE_ERROR(hipGetDeviceCount(&alc::__device_count));

    // Allocate memory for P2P availability between GPUs if more than one device
    if (alc::__device_count > 1) {
      alc::__device_p2p = new bool[alc::__device_count * alc::__device_count];
      alc::__device_threads = new SIZE_TYPE[alc::__device_count];

      for (int i = 0; i < alc::__device_count; i++) {
        // Set reasonable number of threads based on runtime information
        int max_threads_per_block;
        __ALC_CHECK_DEVICE_ERROR(hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, i));

        // Assign the maximum number of threads per block based on the device's capabilities
        alc::__device_threads[i] = max_threads_per_block > 0 ? max_threads_per_block : 1024;  // Fallback to 1024 if an error occurs

        // Get P2P capability
        for (int j = 0; j < alc::__device_count; j++) {
          // Disable P2P by default if both devices are the same
          if (i == j) {
            alc::__device_p2p[i * alc::__device_count + j] = false;
          } else {
            int p2pCapable = 0;
            __ALC_CHECK_DEVICE_ERROR(hipDeviceCanAccessPeer(&p2pCapable, i, j));
            alc::__device_p2p[i * alc::__device_count + j] = (p2pCapable == 1);
            if (p2pCapable) {
              __ALC_CHECK_DEVICE_ERROR(hipSetDevice(i));
              __ALC_CHECK_DEVICE_ERROR(hipDeviceEnablePeerAccess(j, 0));
            }
          }
        }
      }
    } else {
      alc::__device_threads = new SIZE_TYPE[1];
      // Set reasonable number of threads based on runtime information
      int max_threads_per_block;
      __ALC_CHECK_DEVICE_ERROR(hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, 0));

      // Assign the maximum number of threads per block based on the device's capabilities
      alc::__device_threads[0] = max_threads_per_block > 0 ? max_threads_per_block : 1024;  // Fallback to 1024 if an error occurs

      alc::__device_p2p = nullptr;  // Only one device, no P2P needed
    }
  }
#else
  // Fallback setup when neither CUDA nor HIP is used
  void __setup__();
#endif
  void __clean__();
// END
};
