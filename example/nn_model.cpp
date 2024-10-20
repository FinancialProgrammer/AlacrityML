template <class T>
class Model {
  T ** biases;  // ptrs to arr's
  T ** weights; // ptrs to mat's

  T ** gbiases;
  T ** gweights;

  T * output_buffer;

  SIZE_TYPE layers = 0;
  SIZE_TYPE **layer_sizes;

  Model() {}
  void sforward(T *input, T ** buffers) {
    for (SIZE_TYPE i = 0; i < layers; ++i) {
      alc::dense(buffers[i],biases[i],weights[i],buffers[++i]); alc::sync();
      alc::sigmoidV(buffers[++i],buffers[++i]); alc::sync();
    }
  }
  void bforward(T *input, T * output_buffer) {
    for (SIZE_TYPE i = 0; i < layers; ++i) {
      alc::dense(input,biases[i],weights[i],output_buffer); alc::sync();
      alc::sigmoidV(output_buffer,output_buffer); alc::sync();
      input = output_buffer;
    }
  }
  void dforward(T *input, SIZE_TYPE size) {
    T *output;
    for (SIZE_TYPE i = 0; i < layers; ++i) {
      output = alc::malloc(layer_sizes[++i]*sizeof(T));
      alc::dense(input,biases[i],weights[i],output); alc::sync();
      alc::sigmoidV(outupt,output); alc::sync();
      
      alc::free(input);
      input = output;
    }
    alc::free(output);
  }
  // f'(x) = gradient of stochastic object
  void grad(T *input) { // calculates the derivative at each step
  }
  // gf( loss(f(x),v_expected) ) = gradient of stochastic object output
  void dbackward(T *loss) { // backpropagate a loss with respect to parameters
    for (SSIZE_TYPE i = layers-1; i >= 0; --i) {
      alc::gdense(loss,);
    }
  }
};



int main() {


}