#include "unittest.hpp"
#include <Alacrity/__alc.hpp>
#include <Alacrity/extalgo.hpp>

#include <stdio.h>

int main() {
  alc::__setup__();
  alc::set_device(0);
  printf("[ INFO ] device set to %d\n",alc::__device);

  // create arrays
  SIZE_TYPE len = 1024;
  SIZE_TYPE outputLen = 10;
  float *x = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  float *y = (float*)alc::malloc(outputLen*sizeof(float)); __ALC_TEST_DEVICE_ERROR();

  // sum
  TEST_SETUP(sum)
  alc::fill(x,1.f,len);
  alc::fill(y,0.f,outputLen); alc::sync(); __ALC_TEST_DEVICE_ERROR();
  
  TEST_CASE(float,x,1.f,len) __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float,y,0.f,outputLen) __ALC_TEST_DEVICE_ERROR();

  for (SIZE_TYPE i = 0; i < outputLen; ++i) {
    alc::ealgo::sum( // synchronous
      [] __ALC_DEVICE_API (float x, float y) { return x + y; }, 
      len, &y[i], x
    ); __ALC_TEST_DEVICE_ERROR();
  }

  TEST_CASE(float,y,1024.f,outputLen)
  __ALC_TEST_DEVICE_ERROR();

  alc::free(x);
  alc::free(y);

  alc::__clean__();
}
