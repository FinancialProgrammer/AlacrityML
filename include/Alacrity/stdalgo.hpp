#pragma once
#include <Alacrity/__alc.hpp>

namespace alc {
// Memory
#ifdef __ALC_CUDA
  void *malloc(SIZE_TYPE bytes) { void *dptr; cudaMalloc(&dptr,bytes); return dptr; }
  void realloc(void **ptr, SIZE_TYPE bytes) { cudaFree(*ptr); cudaMalloc(ptr,bytes); }
  void free(void *src) { cudaFree(src); }

  void sync() { cudaDeviceSynchronize(); }

  void memcpyDevice(void *dest, const void *src, SIZE_TYPE bytes) { cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToDevice); }
  void memcpyToHost(void *dest, const void *src, SIZE_TYPE bytes) { cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToHost); }
  void memcpyFromHost(void *dest, const void *src, SIZE_TYPE bytes) { cudaMemcpy(dest,src,bytes,cudaMemcpyHostToDevice); }
  void set_device(int devI) { cudaSetDevice(devI); alc::__device = devI; }
#elif defined(__ALC_HIP)
  void *malloc(SIZE_TYPE bytes) { void *dptr; hipMalloc(&dptr,bytes); return dptr; }
  void realloc(void **ptr, SIZE_TYPE bytes) { hipFree(*ptr); hipMalloc(ptr,bytes); }
  void free(void *src) { hipFree(src); }

  void sync() { hipDeviceSynchronize(); }

  void memcpyDevice(void *dest, const void *src, SIZE_TYPE bytes) { hipMemcpy(dest,src,bytes,hipMemcpyDeviceToDevice); }
  void memcpyToHost(void *dest, const void *src, SIZE_TYPE bytes) { hipMemcpy(dest,src,bytes,hipMemcpyDeviceToHost); }
  void memcpyFromHost(void *dest, const void *src, SIZE_TYPE bytes) { hipMemcpy(dest,src,bytes,hipMemcpyHostToDevice); }
  void set_device(int devI) { hipSetDevice(devI); alc::__device = devI; }
#else
  void *malloc(SIZE_TYPE bytes) { return std::malloc(bytes); }
  void realloc(void **ptr, SIZE_TYPE bytes) { *ptr = std::realloc(ptr,bytes); }
  void free(void *src) { std::free(src); }

  void sync() {}

  void memcpyDevice(void *dest, const void *src, SIZE_TYPE bytes) { std::memcpy(dest,src,bytes); }
  void memcpyToHost(void *dest, const void *src, SIZE_TYPE bytes) { std::memcpy(dest,src,bytes); }
  void memcpyFromHost(void *dest, const void *src, SIZE_TYPE bytes) { std::memcpy(dest,src,bytes); }
  void set_device(int devI) { alc::__device = devI; }
#endif
// END

#ifdef __ALC_CUDA
  namespace __cuda {
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __call_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      dfunc(args...);
    }
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __callS_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      dfunc(args...);
    }
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __calllm_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      extern char __shared__ local_memory[];
      dfunc(local_memory,args...);
    }
  };
  struct algo {
    template <class F, class... ARGS>
    static void call(F func, SIZE_TYPE arrlen, ARGS... args) { 
      alc::__cuda::__call_impl<<<alc::block_count(alc::__device,arrlen),alc::__device_threads[alc::__device]>>>(func,arrlen,args...); 
    }
    template <class F, class... ARGS>
    static void callS(F func, SIZE_TYPE arrlen, ARGS... args) { // this may be optimizable
      alc::__cuda::__callS_impl<<<1,1>>>(func,arrlen,args...); 
    }
    template <class F, class... ARGS>
    static void calllm(SIZE_TYPE localMemory, F func, SIZE_TYPE arrlen, ARGS... args) { // this may be optimizable
      alc::__cuda::__calllm_impl<<<
        alc::block_count(alc::__device,arrlen),
        alc::__device_threads[alc::__device],
        localMemory
      >>>(func,arrlen,args...);
    }
  };
#elif defined(__ALC_HIP)
  namespace __hip {
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __call_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      dfunc(args...);
    }
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __callS_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      dfunc(args...);
    }
    template <class F, class... ARGS> __ALC_KERNEL_API
    static void __calllm_impl(F dfunc, SIZE_TYPE arrlen, ARGS... args) {
      // this should be triviably inlinable
      // if not try force inlining
      extern char __shared__ local_memory[];
      dfunc(local_memory,args...);
    }
  };
  struct algo {
    template <class F, class... ARGS>
    static void call(F func, SIZE_TYPE arrlen, ARGS... args) { 
      alc::__hip::__call_impl<<<alc::block_count(alc::__device,arrlen),alc::__device_threads[alc::__device]>>>(func,arrlen,args...); 
    }
    template <class F, class... ARGS>
    static void callS(F func, SIZE_TYPE arrlen, ARGS... args) { // this may be optimizable
      alc::__hip::__callS_impl<<<1,1>>>(func,arrlen,args...); 
    }
    template <class F, class... ARGS>
    static void calllm(SIZE_TYPE localMemory, F func, SIZE_TYPE arrlen, ARGS... args) { // this may be optimizable
      alc::__hip::__calllm_impl<<<
        alc::block_count(alc::__device,arrlen),
        alc::__device_threads[alc::__device],
        localMemory
      >>>(func,arrlen,args...);
    }
  };
#else
  extern void *__host_local_memory;
  struct algo {
    template <class F, class... ARGS>
    static void call(F func, SIZE_TYPE arrlen, ARGS... args) { // a-synchronous
      alc::__impl_hidx = 0;
      func(args...);
    }
    template <class F, class... ARGS>
    static void callS(F func, SIZE_TYPE arrlen, ARGS... args) { // synchronous
      alc::__impl_hidx = 0;
      func(args...);
    }
    template <class F, class... ARGS>
    static void calllm(SIZE_TYPE localMemory, F func, SIZE_TYPE arrlen, ARGS... args) { // this may be optimizable
      alc::__host_local_memory = alc::malloc(localMemory);
      alc::algo::call(func,arrlen,args...);
      alc::free(alc::__host_local_memory);
    }
  };
#endif

// stdalgos
  template <class T>
  void fill(T *arr, T val, SIZE_TYPE len) {
    alc::algo::call(
      [len] __ALC_DEVICE_API (T *arr, T _val) {
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
        for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) {
          arr[gIdx] = _val;
        }
      }, len, arr, val
    );
  }
  template <class T, class F>
  void ffill(T *arr, F func, SIZE_TYPE len) {
    alc::algo::call(
      [len,func] __ALC_DEVICE_API (T *arr) {
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
        for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) {
          arr[gIdx] = func(gIdx);
        }
      }, len, arr
    );
  }
};