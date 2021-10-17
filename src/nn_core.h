/*
 * nn_core.h
 *
 *  Created on: 23 May 2019
 *      Author: Thomas Maynadié
 */

#ifndef NNCORE_H_
#define NNCORE_H_

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __GNUC__
	#define likely(x)       	__builtin_expect(!!(x), 1)
	#define unlikely(x)     	__builtin_expect(!!(x), 0)
	#define unused          	__attribute__((unused))
#else
	#define likely(x)      		x
	#define unlikely(x)    		x
	#define unused
	#pragma warning(disable : 4996) /* For fscanf */
#endif

#ifndef RAND
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define RAND() (((double)rand()) / RAND_MAX)
#endif

typedef enum
{
	actSigmoid
} ACTIVATION;

struct NNet_NeuralNetStruct;

typedef double (*NN_ActivationFunction) (const struct NNet_NeuralNetStruct *_nnet, double _x, ACTIVATION _act);

typedef struct NNet_NeuralNetStruct
{
	int inputs, outputs, hiddenLayers, hidden;

	NN_ActivationFunction actHidden;
	NN_ActivationFunction actOutput;

	int totalWeight;
	int totalNeurons;

	double *weight;
	double *output;
	double *delta;
} NNet_NeuralNet;


void NNWriteWeights (NNet_NeuralNet *_nnet);

NNet_NeuralNet *NNInitNeuralNet (int, int, int, int);
NNet_NeuralNet* CreateNNetFromFile (void);

double const *NNRunNNet (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs);

void NNTrainHidden (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate);
void NNTrainOutputs (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate);
void NNTrain (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate);

void NNGetDelta (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs);

#ifdef __cplusplus
}
#endif

#endif /* NNCORE_H_ */
