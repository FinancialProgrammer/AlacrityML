#include <Alacrity/__alc.hpp>

#ifdef __ALC_HOST
  void alc::__setup__() {
    alc::__device_count = 0;
    alc::__device_p2p = nullptr;
  }
  SIZE_TYPE alc::__impl_hidx;
#endif
int alc::__device;
int alc::__device_count;
bool *alc::__device_p2p;
SIZE_TYPE *alc::__device_threads;

bool alc::__p2p_capable(int device1, int device2) {
  // Ensure __device_p2p is initialized and valid
  if (alc::__device_p2p == nullptr) {
    return false;  // P2P information is not initialized
  }

  #ifdef __ALC_BABYPROOF
    // Ensure the device indices are within the valid range
    if (device1 < 0 || device1 >= alc::__device_count || device2 < 0 || device2 >= alc::__device_count) {
      return false;  // Invalid device indices
    }
  #endif

  // Return the P2P capability between device1 and device2
  return alc::__device_p2p[device1 * alc::__device_count + device2];
}
void alc::__clean__() {
  // Clean up Global Arrays
  if (alc::__device_p2p) {
    delete[] alc::__device_p2p;
    alc::__device_p2p = nullptr;
  }
  if (alc::__device_threads) {
    delete[] alc::__device_threads;
    alc::__device_threads = nullptr;
  }  
}

SIZE_TYPE alc::block_count(SIZE_TYPE deviceId, SIZE_TYPE arrlen) {
  #ifdef __ALC_BABYPROOF
    // Ensure the device indices are within the valid range
    if (device1 < 0 || device1 >= alc::__device_count || device2 < 0 || device2 >= alc::__device_count) {
      return false;  // Invalid device indices
    }
  #endif

  return (arrlen + alc::__device_threads[deviceId] - 1) / alc::__device_threads[deviceId];
}