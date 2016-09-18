#include "stdafx.h"

report* svm_train(svm_problem * prob, const svm_parameter * param)
{
	switch (param->svm_type)
	{
	case AOSVM:
		return solve_aosvm(prob, param);
		break;
	default:
		break;
	}
	SHOWERROR("unknown svm_type")
	return NULL;
}

void svm_predict(report * report, const svm_problem * test_prob, mydouble*& predict)
{
	switch (report->model->param.svm_type)
	{
	case AOSVM:
		aosvm_predict(report, test_prob, predict);
		break;
	default:
		SHOWERROR("undefine svm_type prediction")
		break;
	}
	
}
