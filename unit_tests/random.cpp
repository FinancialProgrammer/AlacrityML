#include "unittest.hpp"
#include <Alacrity/__alc.hpp>
#include <Alacrity/stdalgo.hpp>
#include <Alacrity/random.hpp>

#include <stdio.h>

int main() {
  alc::__setup__();
  alc::set_device(0);
  printf("[ INFO ] device set to %d\n",alc::__device);

  SIZE_TYPE len = 1024;

  // Check Memory Copying
  TEST_SETUP(Test Random Numbers)
  float *x = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  float *y = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();

  auto f = [] __ALC_DEVICE_API (SIZE_TYPE index) { return alc::randn<float>(index); };

  alc::ffill(x,f,len);
  alc::ffill(y,f,len);
  alc::sync(); __ALC_TEST_DEVICE_ERROR();

  TEST_CASE_ADV(float,x,!= 0,len); // this is not a good test
  TEST_CASE_ADV(float,y,!= 0,len); // with 2 it should be slightly better

  alc::free(x);
  alc::free(y);

  alc::__clean__();
}
