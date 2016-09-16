#include "stdafx.h"
#include <chrono>

report * solve_ofoc(svm_problem * train_prob, const svm_parameter * param)
{
	aosvm_report* result = new aosvm_report();
	aosvm_model* model = new aosvm_model();
	result->model = model;

	svm_problem* prob = prob_formalise_unbal_bin(*train_prob, model->index_label, model->label_index, model->switch_label);
	model->prob = prob;

	int N = prob->l;

	//just in case dataset is not random
	std::vector<int> n_index;
	for (int n = 0; n < N; n++)
		n_index.push_back(n);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(n_index.begin(), n_index.end(), std::default_random_engine(seed));

	mydouble lbd = param->lbd;
	int M = (int)param->M;

	std::vector<std::vector<mydouble>>* KR = new std::vector<std::vector<mydouble>>(); 
	int R = 1;
	model->beta_index = new int[N]; //exactly is R, but for just in case we don't prune
	model->beta_index[0] = n_index[0];
	model->beta = new mydouble[N]; //exactly is R
	model->beta[0] = 0;

	

	model->beta_l = R;

	return result;
}
