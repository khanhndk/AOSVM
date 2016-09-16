// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for (int t = times; t>0; t /= 2)
	{
		if (t % 2 == 1) ret *= tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define mydouble float

#define SHOWERROR(name) {std::cout << name; std::getchar();}

#define SHOWVAR(name) {std::cout << #name << ": " << name << std::endl;}

#define SAVEVAR(f, name) f << #name << ": " << name << "\t";
#define SAVEVAR(f, name, expr) f << #name << ": " << expr << "\t";

#include "targetver.h"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <map>
#include <vector>

#include <time.h>
#include <random>

#include "svm_node.h"
#include "svm_parameter.h"
#include "svm_problem.h"

#include "kernel.h"

#include "alg_type.h"
#include "kernel_type.h"

#include "svm_model.h"



#include "report_predict.h"
#include "report.h"


#include "learner.h"
#include "aosvm.h"



// TODO: reference additional headers your program requires here
