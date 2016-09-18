#pragma once

struct aosvm_model : svm_model
{
	int switch_label;
	mydouble* beta;
	int* beta_index;
	int beta_l;
};

struct aosvm_report : report
{

};

report* solve_aosvm(svm_problem *prob, const svm_parameter *param);

void aosvm_predict(report * report, const svm_problem * test_prob, mydouble *& predict);