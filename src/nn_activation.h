/*
 * nn_activation.h
 *
 *  Created on: 24 May 2019
 *      Author: Thomas Maynadié
 */

#ifndef NNACTIVATION_H_
#define NNACTIVATION_H_

#include "nn_core.h"

#ifdef _cplusplus
extern "C"
{
#endif

#define NNGraphStepNumber 4096

#ifndef NNActivation
#define NNHiddenActivation NNHiddenActivationIndirect
#define NNOutputActivation NNOutputActivationIndirect
#else
#define NNHiddenActivation NNActivation
#define NNOutputActivation NNActivation
#endif

double NNActivationFnc (const struct NNet_NeuralNetStruct* _nnet unused, double _x, ACTIVATION _act);

// default activation functions
#define NNActivationHidden actSigmoid
#define NNActivationOutput actSigmoid

#define NNWeightFile "NNetDefWeight.cfg"
#define NNSigmoidFile "NNSigmoidFile.cfg"

double NNActivationThreshold (const struct NNet_NeuralNetStruct* _nnet, double _a, ACTIVATION _act);
double NNActivationLinear (const struct NNet_NeuralNetStruct* _nnet unused, double _a, ACTIVATION _act);

double NNHiddenActivationIndirect (const struct NNet_NeuralNetStruct* _nnet unused, double _a, ACTIVATION _act);
double NNOutputActivationIndirect (const struct NNet_NeuralNetStruct* _nnet unused, double _a, ACTIVATION _act);

#ifdef __cplusplus
}
#endif

#endif /* NNACTIVATION_H_ */
