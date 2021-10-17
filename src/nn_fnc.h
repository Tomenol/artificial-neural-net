/*
 * nn_fnc.h
 *
 *  Created on: 24 May 2019
 *      Author: Thomas Maynadié
 */

#ifndef NNFNC_H_
#define NNFNC_H_

#include <math.h>

#define NNSigmoidFnc(x)				1/(1 + exp(-x))
#define NNSigmoidDerivFnc(u)		u*(1 - u)

#endif /* NNFNC_H_ */
