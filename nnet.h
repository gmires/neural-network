#ifndef NNET_H
#define NNET_H 

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1 
#define OUTPUT_LAYER 2

typedef struct NAFunction {
  float (*activation)(float);
  float (*derivate)(float);
} NAFunction;

typedef struct NNeuron {
  float *w;               // -- weight
  float b;                // -- bias
  float z;                // -- sum(a-1 * w)    
  float a;                // -- act(z)
  float dz;               // -- derive_prime(z)
  float *dw;              // -- derive * a-1
  float db;               // -- derive bias
} NNeuron;

typedef struct NLayer {
  u_int8_t type;
  NNeuron *neurons;
  NAFunction *funct;
} NLayer;

typedef struct NNet {
  size_t size;
  size_t *network; 
  NLayer *layers;
} NNet;

// ---------------------------------------------------------
extern float rand_min;
extern float rand_max;

float rand_float();
float **NetMakeDataArray(int rows, int cols);
void NetFreeDataArray(float **data, int rows);
float sigmoid(float value);
float sigmoid_derivate(float value);
float relu(float value);
float relu_derivate(float value);
float tahn(float value);
float tahn_derivate(float value);
// ---------------------------------------------------------

NNet NetInit(size_t *netsize, size_t size, float(*randfloat)());
void NetPrint(NNet *nn);
NNet* NetEvaluate(NNet *nn, float *input);
NNet *NetBack(NNet *nn, float *output);
NNet *NetUpdate(NNet *nn, int rows, float lr);
float NetCost(NNet *nn, float **data, int rows);
NNet* NetTrain(NNet *nn, float **data, int rows, int epocs, float lr);

// ---------------------------------------------------------
extern NAFunction SIGMOID;
extern NAFunction RELU;
extern NAFunction TANH;

#endif