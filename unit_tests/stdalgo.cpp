#include "unittest.hpp"
#include <Alacrity/__alc.hpp>
#include <Alacrity/stdalgo.hpp>

#include <stdio.h>

int main() {
  alc::__setup__();
  alc::set_device(0);
  printf("[ INFO ] device set to %d\n",alc::__device);

  SIZE_TYPE len = 1024;

  // Check Memory
  TEST_SETUP(Memory Malloc)
  void *good_memory = alc::malloc(1); __ALC_TEST_DEVICE_ERROR();
  if (good_memory == NULL) {
    BAD_CASE(good_memory,not be NULL);
  }
  GOOD_CASE(good_memory,is not NULL);

  // Check Memory Copying
  TEST_SETUP(Test Memory Manipulation)
  float *x = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  float *y = (float*)alc::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();

  alc::fill(x,1.f,len);
  alc::ffill(y, [] __ALC_DEVICE_API (SIZE_TYPE _) { return 0.f; }, len);
  alc::sync(); __ALC_TEST_DEVICE_ERROR();
  
  TEST_CASE(float,x,1,len); __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float,y,0,len); __ALC_TEST_DEVICE_ERROR();

  TEST_SETUP(Host-Device Memory Copying)
  float *hx = (float*)std::malloc(len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();

  alc::memcpyDevice(y,x,len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float,y,1,len);

  alc::memcpyToHost(hx,y,len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  // HOST_TEST_CASE(float,hx,1,len)

  alc::fill(y,0.f,len);
  alc::sync();

  alc::memcpyFromHost(y,hx,len*sizeof(float)); __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float,y,1,len);

  // check bad memory
  TEST_SETUP(Bad Memory Test)
  void *bad_memory = alc::malloc(-1);
  if (bad_memory != NULL) {
    BAD_CASE(bad_memory,NULL);
  }
  GOOD_CASE(bad_memory,NULL);

  alc::__clean__();
}
