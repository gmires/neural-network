#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>
#include "nnet.h"

#define size_of_array(a) (sizeof(a) / sizeof(*a))

#define EPOCS 10000000
#define LRATE 0.2

float TRAINING_DATA[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};
#define TRAINING_COUNT (sizeof(TRAINING_DATA) / sizeof(TRAINING_DATA[0]))

#define INPUTS 2
#define HIDDEN 2
#define OUTPUT 1

float xavier_rand() {
  return rand_float() * sqrtf(2.0f / (INPUTS + HIDDEN));
}

int main(void)
{ 
  srand(time(NULL));

  rand_min = -1.0f;
  rand_max = 1.0f;

  printf("\n1 - XOR TABLE\n\n");
  size_t init[] = {INPUTS, HIDDEN, OUTPUT};
  NNet network = NetInit(init, size_of_array(init), &xavier_rand);
  for(int i = 1; i < (int)network.size; i++) {
    network.layers[i].funct = &TANH;
  }
  printf("TRAIN------------------------------------------------\n");

  float **data = NetMakeDataArray(TRAINING_COUNT, 3);
  for (int r = 0; r < (int)TRAINING_COUNT; r++){
    for (int c = 0; c < 3; c++) {
      data[r][c] = TRAINING_DATA[r][c];
    }
  }

  NetTrain(&network, data, TRAINING_COUNT, EPOCS, LRATE);

  printf("EVALUATE AFTER TRAIN----------------------------------\n");
  float input[2] = {};
  for(int i = 0; i < (int)TRAINING_COUNT; i++){
    input[0] = TRAINING_DATA[i][0];
    input[1] = TRAINING_DATA[i][1];
    float y_expected = TRAINING_DATA[i][2];
    float y_obtained = NetEvaluate(&network, input)->layers[network.size-1].neurons[0].a;

    printf("%f ^ %f = %f | expected = %f\n", input[0], input[1], y_obtained, y_expected);
  }
  printf("PRINT NET--------------------------------------------\n");
  NetPrint(&network);
  printf("-----------------------------------------------------\n");

  NetFreeDataArray(data, TRAINING_COUNT);  

  return 0;
}

/// --------------------------------------------------------------------------------
