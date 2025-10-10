#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include "nnet.h"

#define size_of_array(a) (sizeof(a) / sizeof(*a))

void ExecuteTrain(float train_data[][3], int rows, int epocs, float lrate){
  
  size_t init[] = {2, 3, 2, 1};
  NNet network = NetInit(init, size_of_array(init));
  printf("TRAIN------------------------------------------------\n");

  float **data = NetMakeDataArray(rows, 3);
  for (int r = 0; r < rows; r++){
    for (int c = 0; c < 3; c++) {
      data[r][c] = train_data[r][c];
    }
  }

  NetTrain(&network, data, rows, epocs, lrate);

  printf("EVALUATE AFTER TRAIN----------------------------------\n");
  float input[2] = {};
  for(int i = 0; i < rows; i++){
    input[0] = train_data[i][0];
    input[1] = train_data[i][1];
    float y_expected = train_data[i][2];
    float y_obtained = NetEvaluate(&network, input)->layers[network.size-1].neurons[0].a;

    printf("%f ^ %f = %f | expected = %f\n", input[0], input[1], y_obtained, y_expected);
  }
  printf("-----------------------------------------------------\n");

  NetFreeDataArray(data, rows);  
};

#define EPOCS 1000000
#define LRATE 0.01

float TRAINING_AND[][3] = {
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};
float TRAINING_OR[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1}
};
float TRAINING_XOR[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};
#define TRAINING_COUNT (sizeof(TRAINING_AND) / sizeof(TRAINING_AND[0]))

int main(void)
{ 
  srand(1234);

  printf("\n1 - AND TABLE\n\n");
  ExecuteTrain(TRAINING_AND, TRAINING_COUNT, EPOCS, LRATE);
  printf("\n2 - OR TABLE\n\n");
  ExecuteTrain(TRAINING_OR,  TRAINING_COUNT, EPOCS, LRATE);
  printf("\n3 - XOR TABLE\n\n");
  ExecuteTrain(TRAINING_XOR, TRAINING_COUNT, EPOCS, 0.1);

  return 0;
}

/// --------------------------------------------------------------------------------
