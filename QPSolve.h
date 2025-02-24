#ifndef __QPSOLVE__
#define __QPSOLVE__

#include "QPMatrix.h"

#define MAX_ITER 20      // ����������
#define TOL 1e-6          // �������̶�
#define MU 10.0           // �ϰ�����˥��ϵ��
#define TAU 0.90          // ����˥��ϵ��
#define BIGNUM 1.0e20	  // ����һ�����������

#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef enum {
	QPSOLVE_SUCCESS				= 0x0000,
	QPSOLVE_ERROR_DIM			= 0x00f1,
	QPSOLVE_ERROR_SINGULAR		= 0x00f2,
	QPSOLVE_ERROR_NOT_SQUARE	= 0x00f3,
	QPSOLVE_ERROR_ALLOC			= 0x00f4,
	QPSOLVE_ERROR_MEMORY		= 0x00f5,
	QPSOLVE_ERROR_NULL_PTR		= 0x00f6,
	QPSOLVE_ERROR_UNSUPPORTED	= 0x00f7,
	QPSOLVE_ERROR_NUMERICAL		= 0x00f8
}QPSolveError;

typedef struct {
	int n;                // ����ά��
	int m;                // Լ��������
	Matrix *Q;            // ���������(n x n)
	Matrix *c;            // һ����(n x 1)
	Matrix *A;            // ����ʽԼ������(m x n)
	Matrix *b;            // ����ʽԼ���Ҷ���(m x 1)
} QPProblem;

typedef struct {
	Matrix *x;            // ԭʼ����
	Matrix *lambda;       // ��ż����
	Matrix *s;            // �ɳڱ���
	double mu;            // �ϰ�����
	double duality_gap;   // ��ż��϶
} IPMSolution;

/*����ͷ*/
/*��������ʼ������*/
QPSolveError qp_solve_init(QPProblem* qpp,//��ʼ������ṹ��ָ��
							IPMSolution* sol,//��ʼ������ṹ��ָ��
							int VariableNum,//��ʼ��Ŀ�����ά��
							int ConstraintNum);//��ʼ��Լ��ά��
/*�������������*/
QPSolveError qp_solve_problem_main(const QPProblem* qpp,//���������ṹ��
									IPMSolution* sol,//������ṹ��
									int IterNumMax,  // ���ѭ������
									double TolMax,   // ����������
									double mu,       // �ϰ�����˥��ϵ��
									double tau);	 //����˥��ϵ��

static double compute_max_step(const Matrix* v, const Matrix* dv, double tau);/* ��󲽳����� */
static double compute_duality_gap(const Matrix* lambda, const Matrix* s);/* ��ż��϶���� */

/*�����ͷź���*/
void QPSolution_free(IPMSolution* sol);
void QPProblem_free(QPProblem* qpp);


#endif