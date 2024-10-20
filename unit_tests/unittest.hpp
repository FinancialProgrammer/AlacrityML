#include <stdio.h>

#define TEST_SETUP(name) \
  fprintf(stdout,"###### Starting Test '" #name "' ######\n");

#define GOOD_CASE(arr,val) fprintf(stdout,"[\033[1;32m GOOD \033[0m] ensured '" #arr "' to be value '" #val "'\n")
#define BAD_CASE(arr,val) fprintf(stderr,"[\033[1;31m FAIL \033[0m] ensure '" #arr "' to be value '" #val "', report this on github with your machine and install specifications\n")

#define TEST_CASE(type,arr,val,len) \
  { \
    bool *ptr = (bool*)alc::malloc(sizeof(bool)); \
    bool start_value = true; \
    alc::memcpyFromHost(ptr,&start_value,sizeof(bool)); \
    alc::algo::call( \
      [len] __ALC_DEVICE_API (type *parr, type _val, bool *result) { \
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX; \
        for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) { \
          if (parr[gIdx] != _val) { *result = false; } \
        } \
      }, len, arr,val,ptr \
    ); \
    alc::sync(); \
    alc::memcpyToHost(&start_value,ptr,sizeof(bool)); \
    alc::free(ptr); \
    if (start_value) { \
      GOOD_CASE(arr,val); \
    } else { \
      BAD_CASE(arr,val); \
    } \
  }

#define TEST_CASE_ADV(type,arr,val,len) \
  { \
    bool *ptr = (bool*)alc::malloc(sizeof(bool)); \
    bool start_value = true; \
    alc::memcpyFromHost(ptr,&start_value,sizeof(bool)); \
    alc::algo::call( \
      [len] __ALC_DEVICE_API (type *parr, bool *result) { \
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX; \
        for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) { \
          if (!(parr val)) { *result = false; } \
        } \
      }, len, arr,ptr \
    ); \
    alc::sync(); \
    alc::memcpyToHost(&start_value,ptr,sizeof(bool)); \
    alc::free(ptr); \
    if (start_value) { \
      GOOD_CASE(arr,val); \
    } else { \
      BAD_CASE(arr,val); \
    } \
  }
