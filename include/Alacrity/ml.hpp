#pragma once
#include <Alacrity/math.hpp>
#include <Alacrity/stdalgo.hpp>
#include <Alacrity/linalg.hpp>

#include <stdlib.h> // malloc free

namespace alc {
// helpers
  template <class T>
  struct nn {
    T ** hostBuffer;
    SIZE_TYPE hostSize;
    T * arenaBuffer;
    SIZE_TYPE arenaSize;
    /* bias : weights : output : matBuffer */
    
    T ** biases; // arr of vec
    T ** weights; // arr of mat
    T ** outputs; // arr of vec ( not one big buffer for backward functions )
    T *matBuffer; // mat of largest weight

    SIZE_TYPE paramSize;

    SIZE_TYPE * layerSizes;
    SIZE_TYPE layers;

    nn() = default;
    nn(SIZE_TYPE layers, SIZE_TYPE const* layerSizes) { this->init(layers,layerSizes); }
    ~nn() { this->del(); }

    bool operator!() { 
      return this->arenaBuffer == NULL || this->hostBuffer == NULL;
    }

    static SIZE_TYPE itemsize(SIZE_TYPE layers_len, SIZE_TYPE const* layers) {
      SIZE_TYPE size = layers[0];
      for (SIZE_TYPE i = 1; i < layers_len; ++i)
        size += layers[i] * 2 + layers[i] * layers[i-1]; // bias and output + weight
      return size;
    }
    static SIZE_TYPE largestparam(SIZE_TYPE layers_len, SIZE_TYPE const* layers) {
      SIZE_TYPE size = 0;
      for (SIZE_TYPE i = 1; i < layers_len; ++i) {
        auto wsize = layers[i] * layers[i-1];
        if (size < wsize) size = wsize;
      }
      return size;
    }

    void del() { 
      if (this->arenaBuffer != NULL) {
        alc::free(this->arenaBuffer);
        this->arenaBuffer = NULL;
      }
      if (this->hostBuffer != NULL) {
        ::free(this->hostBuffer);
        this->hostBuffer = NULL;
      }
    }
    bool init(SIZE_TYPE players, SIZE_TYPE const* playerSizes) {
      // set class variables
      this->layers = players;
      this->layerSizes = new SIZE_TYPE[players];
      for (SIZE_TYPE i = 0; i < players; ++i) { this->layerSizes[i] = playerSizes[i]; }
      
      // allocate arenas
      SIZE_TYPE LargestParameter = this->largestparam(players,playerSizes);
      this->paramSize = LargestParameter;
      for (SIZE_TYPE i = 0; i < players; ++i) {
        this->paramSize += playerSizes[i];
      }
      this->paramSize *= sizeof(T);

      this->arenaSize = sizeof(T) *
        (this->itemsize(players,playerSizes) + LargestParameter)
      ;
      this->arenaBuffer = (T*)alc::malloc( this->arenaSize );
      if (this->arenaBuffer == NULL) {
        return false;
      }

      this->hostSize = sizeof(T*) * (
        (players-1) * 2 // biases and weights
        + players // layers and outputs
        )
      ;
      this->hostBuffer = (T**)::malloc(hostSize);
      if (this->hostBuffer == NULL) {
        return false;
      }

      // Fill Host Arena
      this->outputs = reinterpret_cast<T**>(this->hostBuffer);
      this->biases = reinterpret_cast<T**>(&this->hostBuffer[players*sizeof(T*)]);
      this->weights = reinterpret_cast<T**>(&this->hostBuffer[(players + players-1)*sizeof(T*)]);

      // Fill Device Arena
      SIZE_TYPE offset = 0;

      // fill biases
      for (SIZE_TYPE i = 0; i < players-1; ++i) {
        this->biases[i] = reinterpret_cast<T*>(&this->arenaBuffer[offset]);
        offset += playerSizes[i+1]*sizeof(T);
      }

      // fill weights
      for (SIZE_TYPE i = 0; i < players-1; ++i) {
        this->weights[i] = reinterpret_cast<T*>(&this->arenaBuffer[offset]);
        offset += playerSizes[i+1]*playerSizes[i]*sizeof(T);
      }

      // fill outputs
      for (SIZE_TYPE i = 0; i < players; ++i) {
        this->outputs[i] = reinterpret_cast<T*>(&this->arenaBuffer[offset]);
        offset += playerSizes[i]*sizeof(T);
      }

      // fill buffer
      this->matBuffer = reinterpret_cast<T*>(&this->arenaBuffer[offset]);

      printf("%zu  %zu   ", 
        ((players-1) * 2 + players*2) * sizeof(T*),
        offset + this->largestparam(players,playerSizes)
      );


      return true;
    }
  };
// Stochastic Objective Prediction
  template <class T>
  void dense(T *input, T *bias, T *weight, T *output, SIZE_TYPE rows, SIZE_TYPE cols) {
    alc::dot(weight,input,output,rows,cols); // sum auto syncs
    alc::addv(output,bias,output,rows);
  }
// Stochastic Objective Optimization
/*
  template <class T, class F, class G>
  void gdense(T *input, T *gbias, T *gweight, T *model_out, F derivative_activation, G activation, SIZE_TYPE rows, SIZE_TYPE cols) {
    // bias_gradient = input * derivative_activation
    alc::algo::call(
      [cols,derivative_activation] __ALC_DEVICE_API (T *g, T *da, T *a) {
        SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
        for (; gIdx < cols; __ALC_NEXT_IDX(gIdx)) {
          g[gIdx] += da[gIdx] * derivative_activation(a[gIdx]);
        }
      }, cols, gbias, input, model_out
    );
    // weight_gradient = input * derivative_activation * previous_model_output
    for (SIZE_TYPE i = 0; i < rows; ++i) {
      alc::algo::call(
        [rows,cols,derivative_activation] __ALC_DEVICE_API (T *g, T *da, T *a, T *pa) {
          SIZE_TYPE gIdx = __ALC_GLOBAL_IDX;
          for (; gIdx < rows*cols; __ALC_NEXT_IDX(gIdx)) {
            SIZE_TYPE row_index = gIdx / cols;
            SIZE_TYPE col_index = gIdx % cols;
            g[gIdx] += da[row_index] * derivative_activation(a[row_index]) * activation(pa[col_index]);
          }
        }, rows*cols, gweight, input, model_out, model_pout
      );
    }
    // next_input = input * derivative_activation * weight
    alc::ealgo::sum(
      [rows,cols,derivative_activation] __ALC_DEVICE_API (SIZE_TYPE idx, T *da, T *a, T *w) {
        return da[gIdx]*derivative_activation(a[gIdx])*w[gIdx];
      },
    );
  }
*/
};

