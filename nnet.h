#ifndef NNET_H
#define NNET_H 

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1 
#define OUTPUT_LAYER 2

#define OPTIMIZER_SGD 0
#define OPTIMIZER_ADAM 1

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f  

typedef struct NAFunction {
  float (*activation)(float);
  float (*derivate)(float);
} NAFunction;

typedef struct NNeuron {
  float *w;               // -- weight
  float *dw;              // -- derive * a-1
  float *m_w;             // -- ADAM - moment for weight 
  float *v_w;             // -- ADAM - moment for weight 
  float b;                // -- bias
  float db;               // -- derive bias
  float m_b;              // -- ADAM - moment for bias
  float v_b;              // -- ADAM - moment for bias
  float z;                // -- sum(a-1 * w)    
  float a;                // -- act(z)
  float dz;               // -- derive_prime(z)
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
  int t;                 // -- ADAM - time step
  int optimizer;       // -- OPTIMIZER_SGD | OPTIMIZER_ADAM
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
float l_relu(float value);
float l_relu_derivate(float value);
float _tanh(float value);
float _tanh_derivate(float value);
// ---------------------------------------------------------

NNet NetInit(size_t *netsize, size_t size, float(*randfloat)());
void NetPrint(NNet *nn);
void NetFree(NNet *nn);
NNet* NetEvaluate(NNet *nn, float *input);
NNet *NetBack(NNet *nn, float *output);
NNet *NetUpdate(NNet *nn, int rows, float lr);
NNet *NetUpdateAdam(NNet *nn, int rows, float lr, float beta1, float beta2, float eps);
float NetCost(NNet *nn, float **data, int rows);
NNet* NetTrain(NNet *nn, float **data, int rows, int epocs, float lr);

// ---------------------------------------------------------
extern NAFunction SIGMOID;
extern NAFunction RELU;
extern NAFunction TANH;
extern NAFunction LRELU;

extern float Adam_Beta1;
extern float Adam_Beta2;
extern float Adam_Eps;

#endif