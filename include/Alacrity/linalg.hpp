#include <Alacrity/math.hpp>
#include <Alacrity/extalgo.hpp>

namespace alc {
  // vector variations
    template <class T>
    void addv(T *v1, T *v2, T *out, SIZE_TYPE len) {
      alc::algo::call(
        [len] __ALC_DEVICE_API (T *v1, T *v2, T *out) {
          SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
          for (; gIdx < len; __ALC_NEXT_IDX(gIdx)) {
            out[gIdx] = v1[gIdx] + v2[gIdx];
          }
        }, len, v1, v2, out
      );
    }
  // matrix-vector
    template <class T>
    void dot(T *mat, T *buffer, T *vec, T *out, SIZE_TYPE rows, SIZE_TYPE cols) {
      // vec.size == cols
      // out.size == rows
      // mat * vec
      for (SIZE_TYPE i = 0; i < rows; ++i) {
        alc::algo::call(
          [cols] __ALC_DEVICE_API (T *mat, T *vec, T *out) {
            SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
            for (; gIdx < cols; __ALC_NEXT_IDX(gIdx)) {
              out[gIdx] = mat[gIdx] * vec[gIdx];
            }
          }, cols, &mat[i * cols], vec, &buffer[i * cols]
        );
      }
      alc::sync();
      // sum
      for (SIZE_TYPE i = 0; i < rows; ++i) {
        alc::ealgo::sum(
          [] __ALC_DEVICE_API (float x, float y) { return x + y; },
          cols, &out[i], &buffer[i * cols]
        );
      }
    }
  // END
};
