#include "unittest.hpp"
#include <Alacrity/__alc.hpp>

#include <stdio.h>

int main() {
  alc::__setup__();

  if (alc::__device_count > 0) {
    printf("devices found %d\n", alc::__device_count);
    for (int i = 0; i < alc::__device_count; ++i) {
      printf("\tdevice threads %zu\n",alc::__device_threads[i]);
    }
  } else {
    printf("No Seperate Devices Found\n");
  }

  alc::__clean__();
}
