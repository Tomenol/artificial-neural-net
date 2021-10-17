/*
 * nn_activation.c
 *
 *  Created on: 24 May 2019
 *      Author: Thomas Maynadié
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nn_activation.h"
#include "nn_fnc.h"
#include "nn_tools.h"

double NNHiddenActivationIndirect (const struct NNet_NeuralNetStruct* _nnet, double _a, ACTIVATION _act)
{
	return _nnet->actHidden(_nnet, _a, _act);
}

double NNOutputActivationIndirect (const struct NNet_NeuralNetStruct* _nnet, double _a, ACTIVATION _act)
{
	return _nnet->actOutput(_nnet, _a, _act);
}

double NNActivationFnc (const struct NNet_NeuralNetStruct* _nnet unused, double _x, ACTIVATION _act)
{
	if (isnan(_x))
	{
		debug ("Error -> fnc NNActivationFnc : please use a number");
		return -1;
	}
	else
	{
		switch (_act)
		{
		case actSigmoid :  return NNSigmoidFnc (_x); break;
		}
	}
	return 0;
}

double inline NNActivationLinear (const struct NNet_NeuralNetStruct* _nnet, double _a, ACTIVATION _act)
{
    return _a;
}

double inline NNActivationThreshold (const struct NNet_NeuralNetStruct* _nnet, double _a, ACTIVATION _act)
{
    return _a > 0;
}



