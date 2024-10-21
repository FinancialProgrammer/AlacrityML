#include "unittest.hpp"
#include <Alacrity/__alc.hpp>
#include <Alacrity/math.hpp>

#include <stdio.h>

int main() {
  alc::__setup__();
  alc::set_device(0);
  printf("[ INFO ] device set to %d\n",alc::__device);

  // create arrays
  SIZE_TYPE len = 1024;
  float *x = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  float *y = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();

  // activation functions
  TEST_SETUP(Activation Functions)
  alc::fill(x,2.f,len);
  alc::fill(y,0.f,len); alc::sync();

  TEST_CASE(float,x,2.f,len)
  TEST_CASE(float,y,0.f,len)

  alc::algo::call(
    [len] __ALC_DEVICE_API (float *x, float *y) {
      SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
      for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) {
        y[gIdx] = alc::binary_step(x[gIdx]);
      }
    }, len, x,y
  ); alc::sync();
  TEST_CASE(float,y,1.f,len);

  TEST_CASE(float,x,2.f,len);
  alc::fill(y,0.f,len); alc::sync(); TEST_CASE(float,y,0.f,len);

  alc::algo::call(
    [len] __ALC_DEVICE_API (float *x, float *y) {
      SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
      for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) {
        y[gIdx] = alc::sigmoid(x[gIdx]);
      }
    }, len, x,y
  ); alc::sync();
  TEST_CASE(float,y,0.880797f,len);

  // derivative activation functions

  // backward activation functions

  alc::free(x);
  alc::free(y);

  alc::__clean__();
}
