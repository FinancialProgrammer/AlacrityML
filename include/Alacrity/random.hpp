#pragma once

#if defined(__ALC_CUDA)
  #include <curand_kernel.h>
namespace alc {
  template <class T>
  __ALC_DEVICE_API T randn(SIZE_TYPE index = 0) {
    // Initialize the seed and state for CURAND
    curandState state;
    curand_init(/*seed*/ clock64(), /*subsequence*/ index, /*offset*/ 0, &state);

    // Generate a normally distributed random number
    return static_cast<T>(curand_normal(&state));
  }
};
#elif defined(__ALC_HIP)
  #include <hiprand_kernel.h>
namespace alc {
  template <class T>
  __ALC_DEVICE_API T randn(SIZE_TYPE index = 0) {
    // Initialize the seed and state for HIPRAND
    hiprandState state;
    hiprand_init(/*seed*/ clock64(), /*subsequence*/ index, /*offset*/ 0, &state);

    // Generate a normally distributed random number
    return static_cast<T>(hiprand_normal(&state));
  }
};
#else
  #include <random>
namespace alc {
  template <class T>
  __ALC_DEVICE_API T randn(SIZE_TYPE index = 0) {
    // Use the standard library random generation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0.0, 1.0); // mean=0, stddev=1
    return dist(gen);
  }
};
#endif

