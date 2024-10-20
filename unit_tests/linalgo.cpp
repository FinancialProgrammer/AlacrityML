#include "unittest.hpp"
#include <Alacrity/__alc.hpp>
#include <Alacrity/linalg.hpp>

#include <stdio.h>

#define SET_DATA(matVal,bufferVal,vrVal,vcVal) \
  alc::fill(mat,matVal,len); \
  alc::fill(matBuffer,bufferVal,len); \
  alc::fill(vecRow,vrVal,rows); \
  alc::fill(vecCol,vcVal,cols); alc::sync(); __ALC_TEST_DEVICE_ERROR();

int main() {
  alc::__setup__();
  alc::set_device(0);
  printf("[ INFO ] device set to %d\n",alc::__device);

  SIZE_TYPE cols = 1024;
  SIZE_TYPE rows = 128;
  SIZE_TYPE len  = cols * rows;

  float *mat       = (float*)alc::malloc(len*sizeof(float));
  float *matBuffer = (float*)alc::malloc(len*sizeof(float));
  float *vecRow    = (float*)alc::malloc(rows*sizeof(float));
  float *vecCol    = (float*)alc::malloc(cols*sizeof(float));

  // Vector Operations
  TEST_SETUP(Trivial Vector Operations)
  SET_DATA(1.f,1.f,0.f,0.f);
  alc::addv( mat, matBuffer, vecCol, cols ); alc::sync(); __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float, vecCol, 2, cols); __ALC_TEST_DEVICE_ERROR();

  // Dot
  SET_DATA(1.f,0.f,0.f,1.f);
  alc::dot( mat, matBuffer, vecCol, vecRow, rows, cols ); alc::sync(); __ALC_TEST_DEVICE_ERROR();
  TEST_CASE(float, vecRow, cols, rows); __ALC_TEST_DEVICE_ERROR();

  alc::__clean__();
}
