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
	mydouble C = param->C;
	
	
	int R = 1;
	int* beta_index = new int[N]; //exactly is R, but for just in case we don't prune
	beta_index[0] = n_index[0];
	mydouble* beta = new mydouble[N]; //exactly is R
	beta[0] = 0;
	
	std::vector<std::vector<mydouble>*> KR;
	std::vector<mydouble>* KR_tmp = new std::vector<mydouble>();
	KR_tmp->push_back(Kernel::k_function(prob->x[n_index[0]], prob->x[n_index[0]], *param));
	KR.push_back(KR_tmp);
	
	std::vector<std::vector<mydouble>*> P;
	std::vector<mydouble>* P_tmp = new std::vector<mydouble>();
	P_tmp->push_back(1.0 / KR[0]->at(0));
	P.push_back(P_tmp);

	std::vector<mydouble> q;
	q.push_back(0);


	for (int n = 0; n < N; n++)
	{
		int nt = n_index[n];
		svm_node* xnt = prob->x[nt];

		mydouble* kn = new mydouble[R];
		for (int r = 0; r < R; r++)
			kn[r] = Kernel::k_function(prob->x[beta_index[r]], xnt, *param);

		mydouble fn = -1;
		for (int r = 0; r < R; r++)
			fn += kn[r] * beta[r];

		mydouble en = -fn;
		mydouble an = 0;
		if (en >= C / M)
			an = C / en;
		else if (en >= 0)
			an = M;
		
		for (int r = 0; r < R; r++)
			q[r] = lbd * q[r] + 2 * kn[r] * an;


		delete[] kn;
	}


	model->beta = beta;
	model->beta_index = beta_index;
	model->beta_l = R;

	return result;
}
