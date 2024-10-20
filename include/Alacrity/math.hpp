#pragma once
#include <Alacrity/stdalgo.hpp>

#include <cmath>

/*
* device functions for scalars
*
*/

namespace alc {
  // Activation Functions (f(x)) (ref - https://en.wikipedia.org/wiki/Activation_function)
    template <class T> __ALC_DEVICE_API
    T binary_step(T x) { return x >= 0 ? 1 : 0; }
    template <class T> __ALC_DEVICE_API 
    T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }
    template <class T> __ALC_DEVICE_API 
    T tanh(T x) { return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x)); }
    template <class T> __ALC_DEVICE_API 
    T smht(T x, T c0, T c1, T c2, T c3) { return (std::exp(c0*x) - std::exp(-(c1*x))) / (std::exp(c2*x) + std::exp(-(c3*x))); }
    template <class T> __ALC_DEVICE_API 
    T relu(T x) { return x > 0 ? x : 0; }
    template <class T> __ALC_DEVICE_API 
    T gelu(T x) { return x * 0.5 * (1.0 + std::erf(x * (1.0 / std::sqrt(2.0)))); }
    template <class T> __ALC_DEVICE_API 
    T softplus(T x) { return std::log(1 + exp(x)); }
    template <class T> __ALC_DEVICE_API
    T elu(T x, T alpha = 1.0) { return x >= 0 ? x : alpha * (std::exp(x) - 1); }
    template <class T> __ALC_DEVICE_API
    T selu(T x) { return x >= 0 ? 1.0507 * x : 1.0507 * 1.67326 * (std::exp(x) - 1); }
    template <class T> __ALC_DEVICE_API
    T leaky_relu(T x, T alpha = 0.01) { return x > 0 ? x : alpha * x; }
    template <class T> __ALC_DEVICE_API
    T prelu(T x, T p) { return x >= 0 ? x : p * x; }
    template <class T> __ALC_DEVICE_API
    T rpsu(T x, T p) { return x >= 0 ? x : -std::pow(std::log(1 + std::exp(-x)), p); }
    template <class T> __ALC_DEVICE_API
    T silu(T x) { return x / (1 + std::exp(-x)); }
    template <class T> __ALC_DEVICE_API
    T elish(T x) { return x >= 0 ? x / (1 + std::exp(-x)) : (exp(x)-1) / (1 + exp(-x)); }
  // Derivatives (f'(x))
    template <class T> __ALC_DEVICE_API
    T dbinary_step(T x) { return 0; }  // Binary step's derivative is 0 everywhere except undefined at x = 0.
    template <class T> __ALC_DEVICE_API 
    T dsigmoid(T x) {
      T s = 1 / (1 + std::exp(-x));  // sigmoid(x)
      return s * (1 - s);  // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    }
    template <class T> __ALC_DEVICE_API 
    T dtanh(T x) {
      T t = alc::tanh(x);  // tanh(x)
      return 1 - t * t;    // tanh'(x) = 1 - tanh(x)^2
    }
    // the wikipedia page is blank for the derivative
    // template <class T> __ALC_DEVICE_API 
    // T dsmht(T x, T config[4]) {}
    template <class T> __ALC_DEVICE_API 
    T drelu(T x) { return x > 0 ? 1 : 0; }  // Derivative of ReLU is 1 for positive x, 0 for negative x.
    template <class T> __ALC_DEVICE_API 
    T dgelu(T x) {
      const T sqrt_2_pi = 0.3989422804014327;  // 1/sqrt(2*pi)
      T erf_term = std::erf(x / std::sqrt(2.0));
      T gelu_prime = 0.5 * (1 + erf_term) + (x * sqrt_2_pi * std::exp(-x * x / 2));
      return gelu_prime;
    }
    template <class T> __ALC_DEVICE_API 
    T dsoftplus(T x) {
      return 1 / (1 + std::exp(-x));  // Derivative of softplus is the sigmoid function
    }
    template <class T> __ALC_DEVICE_API
    T delu(T x, T alpha = 1.0) {
      return x >= 0 ? 1 : alpha * std::exp(x);  // Derivative of ELU: 1 for x >= 0, alpha * exp(x) for x < 0
    }
    template <class T> __ALC_DEVICE_API
    T dselu(T x) {
      const T lambda = 1.0507;
      const T alpha = 1.67326;
      return x >= 0 ? lambda : lambda * alpha * std::exp(x);  // Derivative of SELU
    }
    template <class T> __ALC_DEVICE_API
    T dleaky_relu(T x, T alpha = 0.01) {
      return x > 0 ? 1 : alpha;  // Derivative of Leaky ReLU: 1 for x >= 0, alpha for x < 0
    }
    template <class T> __ALC_DEVICE_API
    T dprelu(T x, T p) {
      return x >= 0 ? 1 : p;  // Derivative of PReLU depends on parameter p for x < 0
    }
    template <class T> __ALC_DEVICE_API
    T drpsu(T x, T p) {
      T exp_neg_x = std::exp(-x);
      T log_term = std::log(1 + exp_neg_x);
      T pow_term = std::pow(log_term, p - 1);
      return x >= 0 ? 1 : -p * pow_term * (1 - exp_neg_x / (1 + exp_neg_x));  // RPSU derivative
    }
    template <class T> __ALC_DEVICE_API
    T dsilu(T x) {
      T sigmoid_x = 1 / (1 + std::exp(-x));  // Sigmoid function
      return sigmoid_x * (1 + x * (1 - sigmoid_x));  // SiLU derivative
    }
    template <class T> __ALC_DEVICE_API
    T delish(T x) {
      T sigmoid_x = 1 / (1 + std::exp(-x));  // Sigmoid function
      return x >= 0 ? sigmoid_x * (1 + x * (1 - sigmoid_x))  // For x >= 0, similar to SiLU
                    : sigmoid_x * (1 + x * (1 - sigmoid_x)) + std::exp(x) / (1 + std::exp(-x));  // For x < 0
    }

  // Backward Derivatives (f'(f(x)))
  // END
};
