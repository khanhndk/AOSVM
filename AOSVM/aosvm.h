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

report* solve_ofoc(svm_problem *prob, const svm_parameter *param);