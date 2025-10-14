#include "nnet.h"

NAFunction SIGMOID = { 
  .activation = &sigmoid, 
  .derivate = &sigmoid_derivate 
};

NAFunction RELU = { 
  .activation = &relu, 
  .derivate = &relu_derivate 
};

NAFunction TANH = { 
  .activation = &tahn, 
  .derivate = &tahn_derivate 
};

NNet NetInit(size_t *netsize, size_t size, float(*randfloat)()) {
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
          n.layers[i].neurons[j].w[y] = randfloat();
          n.layers[i].neurons[j].dw[y] = 0;
        }
      }
      n.layers[i].neurons[j].z = 0;
      n.layers[i].neurons[j].dz = 0;
      n.layers[i].neurons[j].a = 0;
      n.layers[i].neurons[j].b = n.layers[i].type == INPUT_LAYER ? 0 : randfloat();
      n.layers[i].neurons[j].db = 0;
    }
  };

  return n;
};

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
    printf("\n  Layer %d, Neuron = %d\n", i+1, (int)nn->network[i]);
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
};

NNet* NetEvaluate(NNet *nn, float *input){
  // -- set input layer
  for(int i = 0; i < (int)nn->network[0]; i++){
    nn->layers[0].neurons[i].z = input[i];
    nn->layers[0].neurons[i].a = input[i];
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
};

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
};

NNet *NetUpdate(NNet *nn, int rows, float lr){
  for(int i = 1; i < (int)nn->size; i++){
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
};

float NetCost(NNet *nn, float **data, int rows){
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
};

NNet* NetTrain(NNet *nn, float **data, int rows, int epocs, float lr){
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
    float cost = NetCost(nn, data, rows);
    
    if (e == 1 || e % x == 0)
      printf("Epoch %4d | Loss = %.6f\n", e, cost);

  }
  return nn;
};
