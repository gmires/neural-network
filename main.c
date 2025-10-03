#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <sys/types.h>

#define size_of_array(a) (sizeof(a) / sizeof(*a))

typedef struct NAFunction {
  float (*activation)(float);
  float (*derivate)(float);
} NAFunction;

typedef struct NNeuron {
  float *weight;
  float b;
  float activation;
} NNeuron;

typedef struct NLayer {
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
  return value * (1 - value);
}

float relu(float value){
  return (value > 0.00f) ? value : 0.00f;
}

float relu_derivate(float value){
  return (value < 0) ? 0.00f : 1.00f;
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
    if (i == n.size-1) {
      n.layers[i].funct = &SIGMOID;
    } else if (i > 0) {
      n.layers[i].funct = &RELU;
    }
    n.layers[i].neurons = malloc(sizeof(*n.layers[i].neurons)*n.network[i]);
    for(size_t j = 0; j < n.network[i]; j++){
      if (i > 0){
        n.layers[i].neurons[j].weight = malloc(sizeof(*n.layers[i].neurons[j].weight)*n.network[i-1]);
        for(size_t y = 0; y < n.network[i-1]; y++){
          n.layers[i].neurons[j].weight[y] = rand_float();
        }
      }
      n.layers[i].neurons[j].activation = 0;
      n.layers[i].neurons[j].b = rand_float();
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
    if (i == 0){
      printf("  Type = INPUT\n");
    } else if(i == (int)nn->size-1){
      printf("  Type = OUTPUT\n");
    } else {
      printf("  Type = HIDDEN\n");
    }
    for(int j = 0; j < (int)nn->network[i]; j++) {
      printf("    n = %d\n", j+1);
      printf("    bias = %f\n", nn->layers[i].neurons[j].b);
      printf("    activation = %f\n", nn->layers[i].neurons[j].activation);
      if (i > 0){
        for(int x = 0; x < (int)nn->network[i-1]; x++){
          printf("    weight[%d] %f\n", x, nn->layers[i].neurons[j].weight[x]);
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
    nn->layers[0].neurons[i].activation = input[i];
  }
  // -- Evaluatue in Network
  for (int i = 1; i < (int)nn->size; i++) {
    for(int j = 0; j < (int)nn->network[i]; j++){
      nn->layers[i].neurons[j].activation = nn->layers[i].neurons[j].b; 
      for(int x = 0; x < (int)nn->network[i-1]; x++){
        nn->layers[i].neurons[j].activation += nn->layers[i-1].neurons[x].activation * nn->layers[i].neurons[j].weight[x];
      }
      nn->layers[i].neurons[j].activation = nn->layers[i].funct->activation(nn->layers[i].neurons[j].activation);
    }  
  }
  return nn;
}

/*
float NetCost(NNet *nn, float *traindata){

  return 0;
}*/

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

  printf("-----------------------------------------------------\n");
  float cost = 0;
  float input[2] = {};
  for(int i = 0; i < (int)TRAINING_COUNT; i++){
    input[0] = TRAINING_DATA[i][0];
    input[1] = TRAINING_DATA[i][1];
    float y_expected = TRAINING_DATA[i][2];
    float y_obtained = NetEvaluate(&network, input)->layers[network.size-1].neurons[0].activation;

    float d = y_obtained - y_expected;

    printf("%f ^ %f = %f | expected = %f\n", input[0], input[1], y_obtained, y_expected);

    cost += d*d;
  }
  cost /= (float)(TRAINING_COUNT);
  printf("-----------------------------------------------------\n");
  printf("Model cost = %f\n", cost);
  printf("-----------------------------------------------------\n");

  return 0;
}
