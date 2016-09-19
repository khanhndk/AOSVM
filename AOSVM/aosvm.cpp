#include "stdafx.h"
#include <chrono>

report * solve_aosvm(svm_problem * train_prob, const svm_parameter * param)
{
	aosvm_report* result = new aosvm_report();
	aosvm_model* model = new aosvm_model();
	result->model = model;

	svm_problem* full_prob = prob_formalise_unbal_bin(*train_prob, model->index_label, model->label_index, model->switch_label);
	svm_problem* prob = get_sub_problem(*full_prob, 1);
	model->prob = prob;

	int N = prob->l;

	//just in case dataset is not random
	std::vector<int> n_index;
	for (int n = 0; n < N; n++)
		n_index.push_back(n);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//shuffle(n_index.begin(), n_index.end(), std::default_random_engine(seed));

	mydouble lbd = param->lbd;
	mydouble neglbd_plusone = 1.0 - lbd;
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

	std::vector<mydouble> qn;
	qn.push_back(0);

	std::vector<mydouble> an;
	an.push_back(1);

	int n_miss = 0;
	for (int n = 1; n < N; n++)
	{
		int nt = n_index[n];
		svm_node* xnt = prob->x[nt];

		mydouble* kn = new mydouble[R];
		for (int r = 0; r < R; r++)
			kn[r] = Kernel::k_function(prob->x[beta_index[r]], xnt, *param);

		mydouble fn = -1;
		for (int r = 0; r < R; r++)
			fn += kn[r] * beta[r];

		if (prob->y[nt] * (fn + 1) < 0)
			n_miss++;
		printf("%f\n", 1.0 * n_miss / n);

		mydouble en = -fn;
		mydouble ann = 0;
		if (en >= C / M)
			ann = C / en;
		else if (en >= 0)
			ann = M;
		an.push_back(ann);
		
		for (int r = 0; r < R; r++)
			qn[r] = lbd * qn[r] + 2 * kn[r] * ann;

		mydouble* cov = new mydouble[R*R];
		for (int r1 = 0; r1 < R; r1++)
		{
			int rstart = r1 * R;
			KR_tmp = KR[r1];
			for (int r2 = 0; r2 < R; r2++)
				cov[r2 + rstart] = lbd * (*KR_tmp)[r2];
		}

		mydouble* mu = new mydouble[R];
		for (int r = 0; r < R; r++)
			mu[r] = 0;

		int seed = rand();
		mydouble* gn = multinormal_sample(R, 1, cov, mu, &seed);
		mydouble* knprime = new mydouble[R];
		for (int r = 0; r < R; r++)
			knprime[r] = kn[r];// +gn[r];

		#pragma region Calc P
		if (ann != 0)
		{
			//calc kT_P
			mydouble* kT_P = new mydouble[R];
			for (int r = 0; r < R; r++)
				kT_P[r] = 0;
			for (int r1 = 0; r1 < R; r1++)
			{
				mydouble knr1 = knprime[r1];
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					kT_P[r2] += knr1 * (*P_tmp)[r2];
			}
			mydouble P_scale = 0;
			for (int r = 0; r < R; r++)
				P_scale += kT_P[r] * knprime[r];
			P_scale = 1.0 / ann + P_scale / lbd; //CARE divide zero

			mydouble* P_k = new mydouble[R];
			for (int r1 = 0; r1 < R; r1++)
			{
				mydouble tmp = 0;
				P_tmp = P[r1];

				for (int r2 = 0; r2 < R; r2++)
					tmp += (*P_tmp)[r2] * knprime[r2];
				P_k[r1] = tmp;
			}

			for (int r1 = 0; r1 < R; r1++)
			{
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					(*P_tmp)[r2] = (*P_tmp)[r2] / lbd - P_k[r1] * kT_P[r2] / (P_scale * lbd*lbd);
			}

			delete[] P_k;
			delete[] kT_P;

		}
		else
		{
			for (int r1 = 0; r1 < R; r1++)
			{
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					(*P_tmp)[r2] = (*P_tmp)[r2] / lbd;
			}
		}
		#pragma endregion

		for (int r1 = 0; r1 < R; r1++)
		{
			mydouble tmp = 0;
			P_tmp = P[r1];
			for (int r2 = 0; r2 < R; r2++)
				tmp += (*P_tmp)[r2] * qn[r2];
			beta[r1] = tmp;
		}


		#pragma region Add SV
		if (en > 0)
		{
			//Update beta
			beta[R] = 0;
			beta_index[R] = nt;
			
			//Update KR
			for (int r = 0; r < R; r++)
				KR[r]->push_back(Kernel::k_function(prob->x[beta_index[r]], xnt, *param));
			KR_tmp = new std::vector<mydouble>();
			for (int r = 0; r <= R; r++)
				KR_tmp->push_back(Kernel::k_function(xnt, prob->x[beta_index[r]], *param));
			KR.push_back(KR_tmp);
						
			//Update P
			mydouble* c_tmp = new mydouble[R];
			for (int ci = 0; ci < R; ci++)
			{
				mydouble tmp = 0;
				KR_tmp = KR[ci];
				//for (int bi = 0; bi < R; bi++)
				//	tmp += kn[bi] * an[bi] * (*KR_tmp)[bi];
				for (int ni = 0; ni <= n; ni++)
					tmp += Kernel::k_function(xnt, prob->x[n_index[ni]], *param) * an[ni] * 
								Kernel::k_function(prob->x[beta_index[ci]], prob->x[n_index[ni]], *param);
				//tmp += 1.0 * ann * 1.0; //just for RBF
				c_tmp[ci] = tmp;
			}
			mydouble d_tmp = 0;
			for (int bi = 0; bi < R; bi++)
				d_tmp += kn[bi] * an[bi] * kn[bi];
			d_tmp += 1.0 * ann * 1.0; //just for RBF
			mydouble* cT_P = new mydouble[R];
			for (int r = 0; r < R; r++)
				cT_P[r] = 0;
			for (int r1 = 0; r1 < R; r1++)
			{
				mydouble c_tmpr1 = c_tmp[r1];
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					cT_P[r2] += c_tmpr1 * (*P_tmp)[r2];
			}
			mydouble g_tmp = 0;
			for (int r = 0; r < R; r++)
				g_tmp += cT_P[r] * c_tmp[r];
			g_tmp = 1.0 / (d_tmp - g_tmp);
			mydouble* P_c_tmp = new mydouble[R];
			for (int r1 = 0; r1 < R; r1++)
			{
				mydouble tmp = 0;
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					tmp += (*P_tmp)[r2] * c_tmp[r2];
				P_c_tmp[r1] = tmp;
			}
			for (int r1 = 0; r1 < R; r1++)
			{
				P_tmp = P[r1];
				for (int r2 = 0; r2 < R; r2++)
					(*P_tmp)[r2] = (*P_tmp)[r2] + g_tmp * P_c_tmp[r1] * cT_P[r2];
				P_tmp->push_back(-g_tmp * P_c_tmp[r1]);
			}
			P_tmp = new std::vector<mydouble>();
			for (int r = 0; r < R; r++)
				P_tmp->push_back(-g_tmp * P_c_tmp[r]);
			P_tmp->push_back(g_tmp);
			P.push_back(P_tmp);

			delete[] cT_P;
			delete[] c_tmp;


			//Update an
			//Have update above
			//Update q
			qn.push_back(0); //not sure, maybe k_a*a_n
			R++;
		}

		delete[] knprime;
		delete[] gn;
		delete[] kn;
		#pragma endregion

	}

	SHOWVAR(R);
	model->beta = beta;
	model->beta_index = beta_index;
	model->beta_l = R;
	model->param = *param;
	return result;
}

void aosvm_predict(report * report, const svm_problem * test_prob, mydouble *& predict)
{
	aosvm_model* model = (aosvm_model*)report->model;
	report->reset();
	report->start();

	mydouble* beta = model->beta;
	int* beta_index = model->beta_index;
	int beta_l = model->beta_l;

	predict = new mydouble[test_prob->l];

	for (int n = 0; n < test_prob->l; n++)
	{
		svm_node* x = test_prob->x[n];

		mydouble obj = -1;
		for (int bi = 0; bi < beta_l; bi++)
			obj += beta[bi] * Kernel::k_function(model->prob->x[beta_index[bi]], x, model->param); //check bi wrong

		predict[n] = (obj > 0) ? 1 : -1;
		if (model->switch_label)
			predict[n] = -predict[n];
	}

	report->predict_time = report->stop();
}