#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1 
#define OUTPUT_LAYER 2

#define size_of_array(a) (sizeof(a) / sizeof(*a))

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

float rand_float(){
  return (float)((float)rand() / (float)RAND_MAX);
}

// -----------------------------------------------------------

float sigmoid(float value){
  return 1.00f / (1.00f + exp(-value));
}

float sigmoid_derivate(float value){
  return sigmoid(value) * (1 - sigmoid(value));
}

float relu(float value){
  return (value > 0.00f) ? value : 0.00f;
  //return (value > 0.f) ? value : (0.01 * value);
}

float relu_derivate(float value){
  return (relu(value) < 0) ? 0.00f : 1.00f;
  //return relu(value) < 0 ? 0.01f : 1.00f;
}

NAFunction SIGMOID = {
  .activation = &sigmoid,
  .derivate = &sigmoid_derivate
};

NAFunction RELU = {
  .activation = &relu,
  .derivate = &relu_derivate
};
// ---------------------------------------------------------

NNet NetInit(size_t *netsize, size_t size) {
  NNet n = {0};
  n.size = size;
  n.network = malloc(sizeof(*n.network)*n.size);
  for(int i = 0; i < (int)n.size; i++){
    n.network[i] = netsize[i];
  }
  n.layers = malloc(sizeof(*n.layers)*n.size);
  for(size_t i = 0; i < n.size;i++){
    // --- layer type
    if (i == 0){
      n.layers[i].type = INPUT_LAYER;
    } else if (i < n.size-1){
      n.layers[i].type = HIDDEN_LAYER;
    } else {
      n.layers[i].type = OUTPUT_LAYER;
    }
    // --- activation function
    if (n.layers[i].type == OUTPUT_LAYER) {
      n.layers[i].funct = &SIGMOID;
    } else if (i > 0) {
      n.layers[i].funct = &RELU;
    }
    n.layers[i].neurons = malloc(sizeof(*n.layers[i].neurons)*n.network[i]);
    for(size_t j = 0; j < n.network[i]; j++){
      if (i > 0){
        n.layers[i].neurons[j].w = malloc(sizeof(*n.layers[i].neurons[j].w)*n.network[i-1]);
        n.layers[i].neurons[j].dw = malloc(sizeof(*n.layers[i].neurons[j].dw)*n.network[i-1]);
        for(size_t y = 0; y < n.network[i-1]; y++){
          n.layers[i].neurons[j].w[y] = rand_float();
          n.layers[i].neurons[j].dw[y] = 0;
        }
      }
      n.layers[i].neurons[j].z = 0;
      n.layers[i].neurons[j].dz = 0;
      n.layers[i].neurons[j].a = 0;
      n.layers[i].neurons[j].b = n.layers[i].type == INPUT_LAYER ? 0 : rand_float();
      n.layers[i].neurons[j].db = 0;
    }
  }

  return n;
}

void NetPrint(NNet *nn){
  printf("\n");
  printf("-----------------------------\n");
  printf("Net Layes = %d, ", (int)nn->size);
  printf(" size : ");
  for (int i = 0; i < (int)nn->size; i++) {
    printf("%d ", (int)nn->network[i]);  
  }
  printf("\n");
 
  for(int i = 0; i < (int)nn->size; i++){
    printf("-----------------------------\n");
    printf("\n  Layer %d, Neouron = %d\n", i+1, (int)nn->network[i]);
    if (nn->layers[i].type == INPUT_LAYER){
      printf("  Type = INPUT\n");
    } else if(nn->layers[i].type == OUTPUT_LAYER ){
      printf("  Type = OUTPUT\n");
    } else {
      printf("  Type = HIDDEN\n");
    }
    for(int j = 0; j < (int)nn->network[i]; j++) {
      printf("    n = %d\n", j+1);
      printf("    bias = %f\n", nn->layers[i].neurons[j].b);
      printf("    activation = %f\n", nn->layers[i].neurons[j].a);
      if (i > 0){
        for(int x = 0; x < (int)nn->network[i-1]; x++){
          printf("    weight[%d] %f\n", x, nn->layers[i].neurons[j].w[x]);
        }
      }
      printf("    ------------------------\n");      
    }      
    printf("\n");
  }
}

NNet* NetEvaluate(NNet *nn, float *input){
  // -- set input layer
  for(int i = 0; i < (int)nn->network[0]; i++){
    nn->layers[0].neurons[i].z = input[i];
  }
  // -- Evaluatue in Network
  for (int i = 1; i < (int)nn->size; i++) {
    for(int j = 0; j < (int)nn->network[i]; j++){
      nn->layers[i].neurons[j].z = nn->layers[i].neurons[j].b; 
      for(int x = 0; x < (int)nn->network[i-1]; x++){
        nn->layers[i].neurons[j].z += nn->layers[i-1].neurons[x].z * nn->layers[i].neurons[j].w[x];
      }
      nn->layers[i].neurons[j].a = nn->layers[i].funct->activation(nn->layers[i].neurons[j].z);
    }  
  }
  return nn;
}

NNet *NetBack(NNet *nn, float *output){
  for(int i = 0; i < (int)nn->network[nn->size-1]; i++){
    nn->layers[nn->size-1].neurons[i].dz = (nn->layers[nn->size-1].neurons[i].a - output[i]) * nn->layers[nn->size-1].funct->derivate(nn->layers[nn->size-1].neurons[i].z); 
  }
  for(int i = nn->size-2; i > 0; i--){
    for(int n = 0; n < (int)nn->network[i+1]; n++){
      nn->layers[i+1].neurons[n].db += nn->layers[i+1].neurons[n].dz;
    }
    for(int n = 0; n < (int)nn->network[i]; n++){
      float sum = 0;
      for(int j = 0; j < (int)nn->network[i+1]; j++){
        sum += nn->layers[i+1].neurons[j].w[n] * nn->layers[i+1].neurons[j].dz;
        nn->layers[i+1].neurons[j].dw[n] += nn->layers[i].neurons[n].a * nn->layers[i+1].neurons[j].dz;
      }
      nn->layers[i].neurons[n].dz = sum * nn->layers[i].funct->derivate(nn->layers[i].neurons[n].z);
    }
  }
  return nn;
}

NNet *NetUpdate(NNet *nn, int rows, float lr){
  for(int i = 1; i < (int)nn->size-1; i++){
    for(int n = 0; n < (int)nn->network[i]; n++){
      nn->layers[i].neurons[n].b -= (nn->layers[i].neurons[n].db / rows  * lr);
      nn->layers[i].neurons[n].db = 0;
      for(int l = 0; l < (int)nn->network[i-1]; l++){
        nn->layers[i].neurons[n].w[l] -= (nn->layers[i].neurons[n].dw[l] / rows * lr);
        nn->layers[i].neurons[n].dw[l] = 0;
      }
    }
  }
  return nn;
}

float NetCost(NNet *nn, float **data, int rows, int cols){
  float cost = 0;
  int s_in = (int)nn->network[0];
  int s_ou = (int)nn->network[nn->size-1];
  float input[s_in];
  float output[s_ou];

  for(int r = 0; r < rows; r++){
    for(int i = 0; i < s_in; i++) input[i] = data[r][i];
    for(int i = 0; i < s_ou; i++) output[i] = data[r][s_in + i];
    NetEvaluate(nn, input);
    for(int i = 0; i < s_ou; i++){
      float y_obtained = nn->layers[nn->size-1].neurons[i].a;
      float y_expected = output[i];

      float d = y_obtained - y_expected;
      cost += d * d / s_ou;
    }
  }
  return (cost / rows);
}

NNet* NetTrain(NNet *nn, float **data, int rows, int cols, int epocs, float lr){
  int s_in = (int)nn->network[0];
  int s_ou = (int)nn->network[nn->size-1];
  int x = epocs / 10;
  
  float input[s_in];
  float output[s_ou];

  for(int e = 1; e <= epocs; e++){
    for(int r = 0; r < rows; r++){
      for(int i = 0; i < s_in; i++) input[i] = data[r][i];
      for(int i = 0; i < s_ou; i++) output[i] = data[r][s_in + i];      

      NetEvaluate(nn, input);
      NetBack(nn, output);
    }
    NetUpdate(nn, rows, lr);
    float cost = NetCost(nn, data, rows, cols);
    
    if (e == 1 || e % x == 0)
      printf("Epoch %4d | Loss = %.6f\n", e, cost);

  }
  return nn;
}

float **NetMakeDataArray(int rows, int cols){
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
}

void NetFreeDataArray(float **data, int rows){
  for(int i = 0; i < rows; i++){
    free(data[i]);
  }
  free(data);
}

////---------------------------------------------------------------------------------
#define EPOCS 10000000
#define LRATE 0.01

float TRAINING_DATA[][3] = {
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};
#define TRAINING_COUNT (sizeof(TRAINING_DATA) / sizeof(TRAINING_DATA[0]))

int main(void)
{ 
  srand(1234);

  size_t init[] = {2, 3, 2, 1};
  NNet network = NetInit(init, size_of_array(init));
  printf("TRAIN------------------------------------------------\n");

  float **data = NetMakeDataArray(TRAINING_COUNT, 3);
  for (int r = 0; r < (int)TRAINING_COUNT; r++){
    for (int c = 0; c < 3; c++) {
      data[r][c] = TRAINING_DATA[r][c];
    }
  }

  NetTrain(&network, data, TRAINING_COUNT, 3, EPOCS, LRATE);

  printf("EVALUATE AFTER TRAIN----------------------------------\n");
  float input[2] = {};
  for(int i = 0; i < (int)TRAINING_COUNT; i++){
    input[0] = TRAINING_DATA[i][0];
    input[1] = TRAINING_DATA[i][1];
    float y_expected = TRAINING_DATA[i][2];
    float y_obtained = NetEvaluate(&network, input)->layers[network.size-1].neurons[0].a;

    printf("%f ^ %f = %f | expected = %f\n", input[0], input[1], y_obtained, y_expected);
  }
  printf("-----------------------------------------------------\n");

//  NetPrint(&network);
  NetFreeDataArray(data, TRAINING_COUNT);

  return 0;
}

/// --------------------------------------------------------------------------------
