#pragma once
#include <Alacrity/stdalgo.hpp>

namespace alc {
struct ealgo {
  template <class F, class T, class... ARGS>
  static void sum(F func, SIZE_TYPE arrlen, T *output, T *pinput) {
#ifdef __ALC_HOST
    SIZE_TYPE numBlocks = arrlen;
    T *input = pinput;
#else
    SIZE_TYPE numBlocks = alc::block_count(alc::__device,arrlen);
    T *input = (T*)alc::malloc(sizeof(T) * numBlocks);
    // blockSum
    // call with local/shared memory
    alc::algo::calllm(
      sizeof(T) * alc::__device_threads[alc::__device],
      [arrlen,func] __ALC_DEVICE_API (char *lmemory, T *output, T *input) {
        SIZE_TYPE tIdx = __ALC_LOCAL_IDX;
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
        T *shared_memory = reinterpret_cast<T*>(lmemory);

        // Load data into shared memory
        if (gIdx < arrlen)
        shared_memory[tIdx] = input[gIdx];
        else
        shared_memory[tIdx] = 0.0f;
        __ALC_DEV_SYNC();

        // Perform reduction within the block
        for (SSIZE_TYPE stride = blockDim.x / 2; stride > 0; stride >>= 1) {
          if (tIdx < stride && stride < arrlen) {
            shared_memory[tIdx] = func(shared_memory[tIdx], shared_memory[tIdx + stride]);
          }
          __ALC_DEV_SYNC();
        }

        // Store the block sum in global memory
        if (tIdx == 0) { output[__ALC_BLOCK_IDX] = shared_memory[0]; }
      }, arrlen, input, pinput
    );
#endif
    alc::algo::callS(
      [numBlocks,func] __ALC_DEVICE_API (T *output, T *input) {
        if (__ALC_LOCAL_IDX == 0) {
          for (SIZE_TYPE i = 0; i < numBlocks; ++i) { *output = func(*output, input[i]); }
        }
      }, numBlocks, output, input
    ); // finalSum
    alc::sync();
#ifndef __ALC_HOST
  alc::free(input);
#endif
  }
};
};
