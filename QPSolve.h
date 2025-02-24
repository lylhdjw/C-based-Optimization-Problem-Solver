#ifndef __QPSOLVE__
#define __QPSOLVE__

#include "QPMatrix.h"

#define MAX_ITER 20      // 最大迭代次数
#define TOL 1e-6          // 收敛容忍度
#define MU 10.0           // 障碍参数衰减系数
#define TAU 0.90          // 步长衰减系数
#define BIGNUM 1.0e20	  // 定义一个超大的数字

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
	int n;                // 变量维度
	int m;                // 约束条件数
	Matrix *Q;            // 二次项矩阵(n x n)
	Matrix *c;            // 一次项(n x 1)
	Matrix *A;            // 不等式约束矩阵(m x n)
	Matrix *b;            // 不等式约束右端项(m x 1)
} QPProblem;

typedef struct {
	Matrix *x;            // 原始变量
	Matrix *lambda;       // 对偶变量
	Matrix *s;            // 松弛变量
	double mu;            // 障碍参数
	double duality_gap;   // 对偶间隙
} IPMSolution;

/*函数头*/
/*求解问题初始化函数*/
QPSolveError qp_solve_init(QPProblem* qpp,//初始化问题结构体指针
							IPMSolution* sol,//初始化结果结构体指针
							int VariableNum,//初始化目标变量维数
							int ConstraintNum);//初始化约束维数
/*求解问题主函数*/
QPSolveError qp_solve_problem_main(const QPProblem* qpp,//待求解问题结构体
									IPMSolution* sol,//求解结果结构体
									int IterNumMax,  // 最大循环次数
									double TolMax,   // 最大误差容限
									double mu,       // 障碍参数衰减系数
									double tau);	 //步长衰减系数

static double compute_max_step(const Matrix* v, const Matrix* dv, double tau);/* 最大步长计算 */
static double compute_duality_gap(const Matrix* lambda, const Matrix* s);/* 对偶间隙计算 */

/*缓存释放函数*/
void QPSolution_free(IPMSolution* sol);
void QPProblem_free(QPProblem* qpp);


#endif