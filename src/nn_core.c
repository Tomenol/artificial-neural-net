/*
 * nn_core.c
 *
 *  Created on: 23 May 2019
 *      Author: Thomas Maynadié
 */

#include "nn_activation.h"
#include "nn_core.h"
#include "nn_fnc.h"
#include "nn_tools.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NNet_NeuralNet *NNInitNeuralNet (int _inputs, int _hiddenLayers, int _hidden ,int _outputs)
{
	if (_hiddenLayers < 0) return 0;
	if (_inputs < 1) return 0;
	if (_outputs < 1) return 0;
	if (_hiddenLayers > 0 && _hidden < 1) return 0;

	const int _outputWeight = (_hiddenLayers ? (_hidden + 1) : (_inputs + 1)) * _outputs;
	const int _hiddenWeight = _hiddenLayers ? (_inputs + 1) * (_hidden) + (_hidden + 1) * _hidden * (_hiddenLayers - 1) : 0; // add the bias at each weight
	const int _neuronNumber = _inputs + _outputs + _hiddenLayers*_hidden;
	const int _totalWeight = _outputWeight + _hiddenWeight; // =

	debug ("Initialising Neural net.");
	debug ("output Weight : %d", _outputWeight);
	debug ("Hidden Weight : %d", _hiddenWeight);
	debug ("Total Weight : %d\n", _totalWeight);

	const int _size = sizeof (NNet_NeuralNet) + sizeof(double) * (_totalWeight + _neuronNumber + (_neuronNumber - _inputs));

	NNet_NeuralNet *_nnet = (NNet_NeuralNet*) malloc ( _size);

	_nnet->hidden = _hidden;
	_nnet->inputs = _inputs;
	_nnet->outputs = _outputs;
	_nnet->hiddenLayers = _hiddenLayers;

	_nnet->totalNeurons = _neuronNumber;
	_nnet->totalWeight = _totalWeight;

	_nnet->weight = (double*)((char*)_nnet + sizeof (NNet_NeuralNet));
	_nnet->output = _nnet->weight + _nnet->totalWeight;
	_nnet->delta = _nnet->output + _nnet->totalNeurons;

	debug ("neurons address : %p, size : %p", _nnet->weight, _nnet->totalWeight);
	debug ("output address : %p, size : %p", _nnet->output, _nnet->totalNeurons);
	debug ("delta address : %p, size %p", _nnet->delta, _hidden*_hiddenLayers + _outputs);

	int i;
	for (i = 0; i < _nnet->totalWeight; i++)
	{
		double _randWeight = RAND();
		_nnet->weight[i] = _randWeight - 0.5;
		debug ("Setting weight n%d, address %p, weight %f", i, _nnet->weight + i, _nnet->weight[i]);
	}

	_nnet->actHidden = NNActivationFnc;
	_nnet->actOutput = NNActivationFnc;

	return _nnet;
}

NNet_NeuralNet* CreateNNetFromFile (void)
{
	int _hidden, _inputs, _outputs, _hiddenLayers;

	FILE *_file = fopen(NNWeightFile, "r");

	if (_file == NULL)
	{
		debug ("can't open file. no such file in directory.");
		return NULL;
	}

	printf ("\n");
	debug ("creating Neural Net From File : %s", NNWeightFile);

	fscanf (_file, "%d %d %d %d\r", &_inputs, &_hiddenLayers, &_hidden, &_outputs);

	NNet_NeuralNet *_nnet = NNInitNeuralNet(_inputs, _outputs, _hiddenLayers, _hidden);

	int i;
	for (i = 0; i < _nnet->totalWeight; i++)
	{
		double _weight;
		fscanf (_file, "%le\r", &_weight);
		_nnet->weight[i] = _weight;

		debug ("setting weight %p : %le", _nnet->weight + i, _nnet->weight[i]);
	}

	fclose(_file);

	return _nnet;
}

void NNDeleteNNet (NNet_NeuralNet *_nnet)
{
	free (_nnet);
}

void NNWriteWeights (NNet_NeuralNet *_nnet)
{
	FILE *_file;

	_file = fopen(NNWeightFile, "w");

	if (_file == NULL)
	{
		debug("can't open %s, no such file in directory", NNWeightFile);
		exit(-1);
	}

	fprintf (_file, "%d %d %d %d\r", _nnet->inputs, _nnet->hiddenLayers, _nnet->hidden, _nnet->outputs);

	printf ("\n");

	int i;
	for (i = 0; i < _nnet->totalWeight; i++)
	{
		debug("Writing weight l.%d : %le", i + 1, _nnet->weight[i]);
	}

	fclose (_file);
}

double const *NNRunNNet (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs)
{
	double const *_weight = _nnet->weight;
	double const *_input = _nnet->output;
	double *_output = _nnet->output + _nnet->inputs; // number of outputs = address of start + number of inputs

	memcpy (_nnet->output, _inputs, _nnet->inputs*sizeof (double));

	if (!_nnet->hiddenLayers) // if no hidden layers
	{
		debug ("no hidden layers");
		double *_return = _output; // set the start address for the output array

		int i, j;

		for (i = 0; i < _nnet->outputs; i++) // for each neuron of the output layer
		{
			double _sum = *_weight++ * (-1); // set the bias

			for (j = 0; j < _nnet->inputs; j++) // get all their inputs
			{
				_sum += *_weight++ * _input [j]; // sum them

			}
			*_output++ = NNHiddenActivation (_nnet, _sum, NNActivationHidden); // get the activation value in the new array
		}
		return _return;
	}

	// get input layer
	//NNgetLayerOutput(_nnet, 0, _nnet->hidden, _weight, _nnet->inputs, _input, _output, NNActivationOutput);
	int i, j, k;

	debug ("getting input layer");
	for (i = 0; i < _nnet->hidden; ++i) // for each first hidden layer neurons
	{
		double _sum = *_weight++ * (-1); // set bias value

		for (j = 0; j < _nnet->inputs; ++j) // for each
		{
			_sum += *_weight++ * _input[j]; //
		}
		*_output++ = NNHiddenActivation (_nnet, _sum, NNActivationHidden);
	}

	_input += _nnet->inputs;// set output address as first hidden layer

	// get hidden layer
	//NNgetLayerOutput(_nnet, _nnet->hiddenLayers, _nnet->hidden, _weight, _nnet->hidden, _input, _output, NNActivationHidden);
	for (i = 1; i < _nnet->hiddenLayers; ++i)
	{
		for (j = 0; j < _nnet->hidden; ++j)
		{
			double _sum = *_weight++ * (-1);

			for (k = 0; k < _nnet->hidden; ++k)
			{
				_sum += *_weight++ * _input [k];
			}
			*_output++ = NNHiddenActivation(_nnet, _sum, NNActivationHidden);
		}

		_input += _nnet->hidden;
	}

	double const *_return = _output; // return the new array

	// get output layers
	//NNgetLayerOutput(_nnet, 0, _nnet->outputs, _weight, _nnet->hidden, _input, _output, NNActivationOutput);
	for (i = 0; i < _nnet->outputs; ++i)
	{
		double _sum = *_weight++ * (-1);

		for (j = 0; j < _nnet->hidden; ++j)
		{
			_sum += *_weight++ * _input [j];
		}
		*_output++ = NNOutputActivation (_nnet, _sum, NNActivationOutput);
	}

	// is all data used
	debug ("%d, %d", _weight - _nnet->weight, _nnet->totalWeight);
	debug ("%d, %d",_output - _nnet->output, _nnet->totalNeurons);

	assert(_weight - _nnet->weight == _nnet->totalWeight);
	assert(_output - _nnet->output == _nnet->totalNeurons);

	return _return;
}

void NNgetLayerOutput (const struct NNet_NeuralNetStruct *_nnet, const int _LayerNumber, const int _outputSize, const double *_weight, const int _inputLayerSize, const double *_inputLayer, double *_output, NN_ActivationFunction _actFnc, ACTIVATION _act)
{
	int i, j, k;

	for (k = 0; k < _LayerNumber; k++)
	{
		for (i = 0; i < _outputSize; i++)
		{
			double _sum = *_weight++ * (-1);

			for (j = 0; j < _inputLayerSize; j++)
			{
				_sum += *_weight++ * _inputLayer [j];
			}
			*_output++ = _actFnc (_nnet, _sum, _act);
		}
	}
}

void NNTrain (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate)
{
	NNRunNNet(_nnet, _inputs);

	NNGetDelta (_nnet, _inputs, _desiredOutputs);

	NNTrainOutputs (_nnet, _inputs, _desiredOutputs, _rate);
	NNTrainHidden (_nnet, _inputs, _desiredOutputs, _rate);
}

void NNTrainOutputs (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate)
{
	double const  *_delta = _nnet->delta + (_nnet->hiddenLayers * _nnet->hidden);
	double *_weight = _nnet->weight + (_nnet->hiddenLayers
			? _nnet->hidden * (_nnet->inputs + 1) + (_nnet->hiddenLayers - 1) * _nnet->hidden * (_nnet->hidden + 1)
			: 0);
	double const *_prevOutput = _nnet->output + (_nnet->hiddenLayers
			? _nnet->inputs + _nnet->hidden * (_nnet->hiddenLayers - 1)
			: 0);

	int i, j;
	for (i = 0; i < _nnet->outputs; i++)
	{
		*_weight++ += *_delta * _rate * -1;
		for (j = 1; j < (_nnet->hiddenLayers ? _nnet->hidden : _nnet->inputs) + 1; j++)
		{
			*_weight++ += *_delta * _prevOutput [j - 1] * _rate;
		}
		_delta++;
	}

	assert(_weight - _nnet->weight == _nnet->totalWeight);
}

void NNTrainHidden (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs, double _rate)
{
	int i, j, k;

	for (i = _nnet->hiddenLayers - 1; i >= 0; i--)
	{
		double const *_delta = _nnet->delta + i * _nnet->hidden;
		double *_weight = _nnet->weight + (i
				? _nnet->hidden * (_nnet->inputs + 1) + (i - 1) * _nnet->hidden * (_nnet->hidden + 1)
				: 0);
		double const *_prevOutput = _nnet->output + (i
				? _nnet->inputs + _nnet->hidden * (i - 1)
				: 0);

		for (j = 0; j < _nnet->hidden; j++)
		{
			*_weight++ += *_delta * _rate * -1;
			for (k = 1; k < (i == 0 ? _nnet->inputs : _nnet->hidden) + 1; k++)
			{
				*_weight++ += *_delta * _prevOutput [k - 1] * _rate;
			}
			_delta++;
		}

	}
}

void NNGetDelta (const struct NNet_NeuralNetStruct *_nnet, double const *_inputs, double const *_desiredOutputs)
{
	{
		double *_delta = _nnet->delta + _nnet->hiddenLayers * _nnet->hidden;
		double const *_output = _nnet->output + _nnet->inputs + _nnet->hiddenLayers * _nnet->hidden;
		double const *_target = _desiredOutputs;

		int i;
		for (i = 0; i < _nnet->outputs; i++)
		{
			*_delta++ = NNSigmoidDerivFnc(*_output) * (*_target - *_output);
			debug ("gradient value for output %d : %.3f or %.3f", i, *(_delta - 1), *_output * (1 - *_output) * (*_target - *_output));
			_output++;
			_target++;
		}
	}

	int i;
	// get all deltas for hidden layers
	for (i = _nnet->hiddenLayers - 1; i >= 0; i--)
	{
		const double *_output = _nnet->output + _nnet->inputs + _nnet->hidden * i;
		double *_delta = _nnet->delta + _nnet->hidden * i;

		double *_nextWeight = _nnet->weight + (_nnet->inputs + 1) * _nnet->hidden + (i) * (_nnet->hidden + 1) * _nnet->hidden;
		double *_nextDelta = _nnet->delta + _nnet->hidden * (i + 1);

		int j;
		for (j = 0; j < _nnet->hidden; j++)
		{
			double _deltaSum = 0;

			int k;
			for (k = 0; k < (i == _nnet->hiddenLayers - 1 ? _nnet->outputs : _nnet->hidden ); k++)
			{
				int wNum = (j + 1) + k * (_nnet->hidden + 1);
				_deltaSum += _nextWeight [wNum] * _nextDelta [k];
			}

			*_delta++ = NNSigmoidDerivFnc(*_output) * _deltaSum;
			_output++;
		}
	}
}

