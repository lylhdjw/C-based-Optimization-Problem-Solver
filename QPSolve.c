#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "QPMatrix.h"
#include "QPSolve.h"
#define FREE()\
do{\
	matrix_free(&KKT);\
	matrix_free(&KKT_I); \
	matrix_free(&rhs);\
	matrix_free(&dxdlds);\
	matrix_free(&temp1);\
	matrix_free(&temp2);\
	matrix_free(&temp3); \
	matrix_free(&dx);\
	matrix_free(&dlambda);\
	matrix_free(&ds);\
	matrix_free(&AT);\
} while (0);



/* 初始化函数 */
QPSolveError qp_solve_init(QPProblem* qpp, IPMSolution* sol, int VariableNum, int ConstraintNum) 
{
	// 初始化问题维度
	qpp->n = VariableNum;
	qpp->m = ConstraintNum;

	// 分配矩阵内存并初始化为0
	Matrix *Q = (Matrix*)malloc(sizeof(Matrix));
	Matrix *c = (Matrix*)malloc(sizeof(Matrix));
	Matrix *A = (Matrix*)malloc(sizeof(Matrix));
	Matrix *b = (Matrix*)malloc(sizeof(Matrix));

	matrix_create(Q, VariableNum, VariableNum);
	matrix_create(c, VariableNum, 1);
	matrix_create(A, ConstraintNum, VariableNum);
	matrix_create(b, ConstraintNum, 1);
	matrix_set_zero(Q);
	matrix_set_zero(c);
	matrix_set_zero(A);
	matrix_set_zero(b);

	qpp->Q = Q;
	qpp->c = c;
	qpp->A = A;
	qpp->b = b;

	// 初始化解决方案
	sol->x = (Matrix*)malloc(sizeof(Matrix));
	sol->lambda = (Matrix*)malloc(sizeof(Matrix));
	sol->s = (Matrix*)malloc(sizeof(Matrix));

	matrix_create(sol->x, VariableNum, 1);
	matrix_create(sol->lambda, ConstraintNum, 1);
	matrix_create(sol->s, ConstraintNum, 1);
	matrix_set_zero(sol->x);
	matrix_set_zero(sol->lambda);
	matrix_set_zero(sol->s);
	// 初始化参数
	sol->mu = 1.0;
	sol->duality_gap = 0.0;

	// 设置初始值为全1
	for (int i = 0; i < VariableNum; ++i)
	{
		matrix_set(sol->x, i, 0, 1.0);
	}
	for (int i = 0; i < ConstraintNum; ++i) 
	{
		matrix_set(sol->lambda, i, 0, 10.0);
		matrix_set(sol->s, i, 0, 1.0+ 4.231e-1);
	}
	return QPSOLVE_SUCCESS;
}

/* 对偶间隙计算 */
static double compute_duality_gap(const Matrix* lambda, const Matrix* s) 
{
	double gap = 0.0;
	for (int i = 0; i < lambda->rows; ++i) {
		gap += matrix_get(lambda, i, 0) * matrix_get(s, i, 0);
	}
	return gap / lambda->rows;
}

/* 最大步长计算 */
static double compute_max_step(const Matrix* v, const Matrix* dv, double tau) 
{
	double max_step = BIGNUM;
	double step;
	for (int i = 0; i < v->rows; ++i) 
	{
		if (myabs(dv->data[i]) < 1e-12) continue;
		if (dv->data[i] < 0) 
		{
			step = -v->data[i] / (dv->data[i] + 1e-5);
			if (step < max_step) max_step = step;
		}
	}
	return tau * (myabs(max_step - BIGNUM)<1e-8 ? 1.0 : max_step);
}

/* 主求解函数 */
QPSolveError qp_solve_problem_main(const QPProblem* qpp, IPMSolution* sol,
	int IterNumMax, double TolMax,
	double mu, double tau)
{
	int n = qpp->n;
	int m = qpp->m;
	Matrix *Q = qpp->Q, *A = qpp->A, *c = qpp->c, *b = qpp->b;
	Matrix *x = sol->x, *lambda = sol->lambda, *s = sol->s;

	double current_mu = sol->mu;
	int iter = 0;
	MatrixError err;
	double alpha, beta, step;

	Matrix KKT, rhs, dxdlds, temp1, temp2, temp3, dx, dlambda, ds, AT;
	Matrix rhs_x, x_sol;
	Matrix KKT_I;

	double dx_norm = 0.0;
	double gap;

	// 处理无约束情况（m == 0）
	if (m == 0) {
		matrix_create(&rhs_x, n, 1);
		matrix_create(&x_sol, n, 1);
		err = matrix_scalar_multiply(&rhs_x, c, -1.0); // rhs_x = -c
		if (err != MATRIX_SUCCESS) {
			matrix_free(&rhs_x);
			matrix_free(&x_sol);
			printf("ERROR:QPSOLVE_ERROR_SINGULAR \n");
			return QPSOLVE_ERROR_SINGULAR;
		}

		if (err = matrix_solve_lu(Q, &rhs_x, &x_sol) != MATRIX_SUCCESS)
		{
			matrix_free(&rhs_x);
			matrix_free(&x_sol);
			printf("ERROR:QPSOLVE_ERROR_SINGULAR \n");
			return QPSOLVE_ERROR_SINGULAR;
		}
		matrix_copy(sol->x, &x_sol);
		sol->duality_gap = 0.0;
		sol->mu = 0.0;
		printf("QPSOLVE_SUCCESS \n");
		matrix_free(&rhs_x);
		matrix_free(&x_sol);
		return QPSOLVE_SUCCESS;
	}
	// 以下处理有约束情况（m > 0）
	// 创建临时矩阵
	
	matrix_create(&KKT, n + 2 * m, n + 2 * m);
	matrix_create(&KKT_I, n + 2 * m, n + 2 * m);
	matrix_create(&rhs, n + 2 * m, 1);
	matrix_create(&dxdlds, n + 2 * m, 1);
	matrix_create(&temp1, n, 1);
	matrix_create(&temp2, n, 1);
	matrix_create(&temp3, m, 1);
	matrix_create(&dx, n, 1);
	matrix_create(&dlambda, m, 1);
	matrix_create(&ds, m, 1);
	matrix_create(&AT, n, m);
	matrix_set_zero(&KKT);
	matrix_set_zero(&rhs);
	matrix_identity(&KKT_I);
	matrix_scalar_multiply(&KKT_I, &KKT_I, 1e-8);

	/*初始化s*/
	if (err = matrix_multiply(&temp3, A, x) != MATRIX_SUCCESS)
	{
		printf("ERROR: MATRIX_ERROR_DIM\n");
		FREE();
		return QPSOLVE_ERROR_DIM;
	}
	for (int i = 0; i < m; ++i) {
		double s_i = matrix_get(&temp3, i, 0) - matrix_get(b, i, 0);
		matrix_set(s, i, 0, s_i);
	}

	/*基于原始可行点预估，初始化lambda*/
	for (int kk = 0; kk < m; kk++)
	{
		matrix_set(lambda, kk, 0, mu/(s->data[kk] + 1.5e-5));
	}

	while (iter++ < IterNumMax) {
		// 构建KKT矩阵和右端向量
		// 构建KKT矩阵 [Q  A^T  0 ][dx]	=	-[Qx + c + A^T*l]			Q[n*n]	AT[n*m]	0[n*m]
		//			   [A   0  -I ][dl]	=	-[Ax + s - b]				A[m*n]	0[m*m]	-I[m*m]
		//             [0   S   L ][ds]	=	-[S*L*e - mu*e]				0[m*n]	S[m*m]	L[m*m]
		printf("The iter number is %d\n", iter);
		
		// Block Q
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				matrix_set(&KKT, i, j, matrix_get(Q, i, j));
			}
		}
		// Block A^T
		matrix_transpose(&AT, A);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				matrix_set(&KKT, i, n + j, matrix_get(&AT, i, j));
			}
		}
		// Block A
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				matrix_set(&KKT, n + i, j, matrix_get(A, i, j));
			}
		}
		// Block -I
		for (int i = 0; i < m; ++i) {
			matrix_set(&KKT, n + i, n + m + i, -1.0);
		}
		// 对角线块 S, L
		for (int i = 0; i < m; ++i) {
			matrix_set(&KKT, n + m + i, n + i, matrix_get(s, i, 0));
			matrix_set(&KKT, n + m + i, n + m + i, matrix_get(lambda, i, 0));
		}
		matrix_add(&KKT, &KKT, &KKT_I);
		
		// 构建右端向量
		
		if (err = matrix_multiply(&temp1, Q, x) != MATRIX_SUCCESS){
			printf("ERROR: MATRIX_ERROR_DIM\n");
			FREE();
			return QPSOLVE_ERROR_DIM;
		}
		if (err = matrix_multiply(&temp2, &AT, lambda) != MATRIX_SUCCESS){
			printf("ERROR: MATRIX_ERROR_DIM\n");
			FREE();
			return QPSOLVE_ERROR_DIM;
		}
		//matrix_print("temp2", &temp2);
		// 构建KKT矩阵 [Q  A^T  0 ][dx]	=	-[Qx + c + A^T*l]
		//			   [A   0  -I ][dl]	=	-[Ax + s - b]
		//             [0   S   L ][ds]	=	-[S*L*e - mu*e]
		for (int i = 0; i < n; ++i) {
			rhs.data[i] = -matrix_get(&temp1, i, 0) - matrix_get(&temp2, i, 0) - matrix_get(c, i, 0);
			//printf("rhs.data[%d] is %f\n", i, rhs.data[i]);
		}

		if (err = matrix_multiply(&temp3, A, x) != MATRIX_SUCCESS)
		{
			printf("ERROR: MATRIX_ERROR_DIM\n");
			FREE();
			return QPSOLVE_ERROR_DIM;
		}
		for (int i = 0; i < m; ++i) {
			rhs.data[n + i] = -matrix_get(&temp3, i, 0) + matrix_get(s, i, 0) + matrix_get(b, i, 0);
			//printf("rhs.data[%d] is %f\n", n+i, rhs.data[n+i]);
		}
		for (int i = 0; i < m; ++i) {
			rhs.data[n + m + i] = -matrix_get(s, i, 0)*matrix_get(lambda, i, 0) + current_mu;
			//printf("rhs.data[%d] is %f\n", n+m+i, rhs.data[n+m+i]);
		}
		/*
		matrix_print("KKT", &KKT);
		printf("********************\n");
		matrix_print("rhs", &rhs);
		printf("********************\n");
		*/

		// 解KKT系统
		if ((err = matrix_solve_lu(&KKT, &rhs, &dxdlds)) != MATRIX_SUCCESS) {
			FREE();
			printf("ERROR:QPSOLVE_ERROR_SINGULAR\n");
			return QPSOLVE_ERROR_SINGULAR;
		}
		/*
		matrix_print("dxdlds", &dxdlds);
		printf("********************\n");
		*/
		// 提取各分量
		for (int i = 0; i < n; ++i) dx.data[i] = dxdlds.data[i];
		for (int i = 0; i < m; ++i) dlambda.data[i] = dxdlds.data[n + i];
		for (int i = 0; i < m; ++i) ds.data[i] = dxdlds.data[n + m + i];

		// 计算步长
		alpha = compute_max_step(lambda, &dlambda, tau);
		beta = compute_max_step(s, &ds, tau);
		step = MIN(alpha, beta);

		// 更新变量
		for (int i = 0; i < n; ++i) {
			x->data[i] += step * dx.data[i];
		}
		for (int i = 0; i < m; ++i) {
			lambda->data[i] += step * dlambda.data[i];
			s->data[i] += step * ds.data[i];
			if (lambda->data[i] < 0 || s->data[i] < 0)
			{
				FREE();
				printf("ERROR: lambda or s < 0!\n");
				return QPSOLVE_ERROR_NUMERICAL;
			}
		}
		printf("Answer is :\n");
		matrix_print("x", x);
		printf("\n");
		/*
		printf("****************************");
		printf("\n");
		matrix_print("lambda", lambda);
		printf("\n");
		printf("****************************");
		printf("\n");
		matrix_print("s", s);
		printf("\n");
		printf("****************************");
		printf("\n");
		*/
		// 更新障碍参数
		current_mu = (compute_duality_gap(lambda, s) / m) * mu;
		sol->mu = current_mu;
		gap = compute_duality_gap(lambda, s);
		printf("gap is %f\n", gap);
		if (myabs(gap)< TolMax) break;
		if (myabs(gap)>1e10)
		{
			FREE();
			printf("ERROR: Numerical diverge!\n");
			return QPSOLVE_ERROR_NUMERICAL;
		}
	}
	// 释放资源
	FREE();
	printf("QPSolve Success!\n");
	return QPSOLVE_SUCCESS;
}


/* 缓存释放函数 */
void QPSolution_free(IPMSolution* sol) {
	matrix_free(sol->x); free(sol->x);
	matrix_free(sol->lambda); free(sol->lambda);
	matrix_free(sol->s); free(sol->s);
}

void QPProblem_free(QPProblem* qpp) {
	matrix_free(qpp->Q); free(qpp->Q);
	matrix_free(qpp->c); free(qpp->c);
	matrix_free(qpp->A); free(qpp->A);
	matrix_free(qpp->b); free(qpp->b);
}
