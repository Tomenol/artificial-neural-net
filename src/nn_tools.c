/*
 * nn_tools.c
 *
 *  Created on: 23 May 2019
 *      Author: Thomas Maynadié
 */

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include "nn_tools.h"

void debug (const char* _txt, ...)
{
	if (NN_debugMode)
	{
		printf ("[DEBUG MODE] : ");

		va_list _vl;
		va_start (_vl, _txt);

		vprintf(_txt, _vl);

		va_end(_vl);

		printf ("\n");
	}
}
