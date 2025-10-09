#include "nnet.h"
#include <math.h>

float rand_float() {
  return (float)((float)rand() / (float)RAND_MAX);
};

float **NetMakeDataArray(int rows, int cols) {
  float **array = malloc(rows * sizeof(float*));    
  if (array == NULL) {
    perror("error memory allocation rows!");
    exit(EXIT_FAILURE);
  }
  
  for (int i = 0; i < rows; i++) {
    array[i] = malloc(cols * sizeof(float));
    if (array[i] == NULL) {
      perror("error memory allocation cols!");
      exit(EXIT_FAILURE);
    }
  }
  return array;
};

void NetFreeDataArray(float **data, int rows) {
  for(int i = 0; i < rows; i++){
    free(data[i]);
  }
  free(data);
};

float sigmoid(float value) {
  return 1.00f / (1.00f + exp(-value));
};

float sigmoid_derivate(float value) {
  return sigmoid(value) * (1 - sigmoid(value));
};

float relu(float value) {
  return (value > 0.00f) ? value : 0.00f;
  //return (value > 0.f) ? value : (0.01 * value);
};

float relu_derivate(float value) {
  return (relu(value) < 0) ? 0.00f : 1.00f;
  //return relu(value) < 0 ? 0.01f : 1.00f;
};
