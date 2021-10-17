/*
 ============================================================================
 Name        : simple_or_gate.c
 Author      : Thomas Maynadie
 Version     :
 Copyright   : all right reserved
 Description : Neural net in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "NNetDef.h"
#include "tools.h"

int main(void)
{
	srand (time(0));

	const double input [4] [2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	const double output [4] = {0, 1, 1, 1};

	NNet_NeuralNet* _nnet = NNInitNeuralNet(2, 1, 2, 1);

	NNWriteWeights(_nnet);


	printf("Output for [%1.f, %1.f] is %1.f.\n", input[0][0], input[0][1], *NNRunNNet(_nnet, input[0]));
	printf("Output for [%1.f, %1.f] is %1.f.\n", input[1][0], input[1][1], *NNRunNNet(_nnet, input[1]));
	printf("Output for [%1.f, %1.f] is %1.f.\n", input[2][0], input[2][1], *NNRunNNet(_nnet, input[2]));
	printf("Output for [%1.f, %1.f] is %1.f.\n", input[3][0], input[3][1], *NNRunNNet(_nnet, input[3]));
	printf ("\n\n");

	int i;
	for (i = 0; i < 200; i++)
	{
		NNTrain(_nnet, input [0], output + 0, 3);
		NNTrain(_nnet, input [1], output + 1, 3);
		NNTrain(_nnet, input [2], output + 2, 3);
		NNTrain(_nnet, input [3], output + 3, 3);
	}

	printf ("\n\n");

    printf("\nOutput for [%1.f, %1.f] is %1.f at %f.\n\n", input[0][0], input[0][1], *NNRunNNet(_nnet, input[0]),*NNRunNNet(_nnet, input[0]));
    printf("\nOutput for [%1.f, %1.f] is %1.f at %f.\n\n", input[1][0], input[1][1], *NNRunNNet(_nnet, input[1]),*NNRunNNet(_nnet, input[1]));
    printf("\nOutput for [%1.f, %1.f] is %1.f at %f.\n\n", input[2][0], input[2][1], *NNRunNNet(_nnet, input[2]),*NNRunNNet(_nnet, input[1]));
    printf("\nOutput for [%1.f, %1.f] is %1.f at %f.\n\n", input[3][0], input[3][1], *NNRunNNet(_nnet, input[3]),*NNRunNNet(_nnet, input[1]));

	printf ("\n\n");

	system("Pause");
}
