#ifndef __QPMATRIX__
#define __QPMATRIX__

#include <complex.h>

typedef int bool;
#ifndef true
#define true 1
#endif

#ifndef false
#define false 0
#endif



// 错误码定义
typedef enum {
	MATRIX_SUCCESS,
	MATRIX_ERROR_DIM,
	MATRIX_ERROR_SINGULAR,
	MATRIX_ERROR_NOT_SQUARE,
	MATRIX_ERROR_ALLOC,
	MATRIX_ERROR_MEMORY,
	MATRIX_ERROR_NULL_PTR,
	MATRIX_ERROR_UNSUPPORTED,
	MATRIX_ERROR_NUMERICAL
} MatrixError;

// 矩阵结构体
typedef struct {
	int rows;
	int cols;
	double* data;
} Matrix;

double myabs(x);//自定义绝对值函数
int compare_double(const void* a, const void* b);/*辅助函数-用于实现ab大小比较，a大于b返回1，否则返回-1*/
bool matrix_is_symmetric(const Matrix* A);/*判定矩阵A是否是对称阵的函数，是返回true，否则返回False*/
MatrixError matrix_create(Matrix* mat, int rows, int cols);/*矩阵初始化函数*/
void matrix_free(Matrix* mat);/*矩阵释放函数*/
MatrixError matrix_copy(Matrix* dest, const Matrix* src);/*矩阵赋值函数，dest为目标，src为源*/
double matrix_get(const Matrix* mat, int row, int col);/*获取row行col列这个元素的值*/
void matrix_set(Matrix* mat, int row, int col, double val);/*对矩阵row行col列元素赋val值*/
MatrixError matrix_add(Matrix* C, const Matrix* A, const Matrix* B);/* 矩阵相加算法，返回为C */
MatrixError matrix_multiply(Matrix* C, const Matrix* A, const Matrix* B);/* 矩阵相乘算法，返回为C = A*B */
MatrixError matrix_lu_decompose(const Matrix* A, Matrix* LU, int *pivot_sign, int* pivots);/* 矩阵A的LU分解算法，pivot_sign和pivot为过程中进行过的行变换的符号记录和矩阵记录 */
MatrixError matrix_identity(Matrix* mat);/*创建单位矩阵*/
MatrixError matrix_row_swap(Matrix* mat, int row1, int row2);/*矩阵行互换*/
MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar);/*矩阵C = scalar*A，数乘运算*/
MatrixError matrix_determinant(const Matrix* A, double* det);/* 基于LU分解的矩阵行列式计算,返回det */
MatrixError matrix_solve_lu(const Matrix* A, const Matrix* B, Matrix* X);/*基于LU分解的线性方程组求解算法，方程组形式为AX=B*/
MatrixError matrix_inverse(Matrix* inv, const Matrix* A);/* 基于线性方程组求解的矩阵求逆函数，结果储存于inv中 */
MatrixError matrix_transpose(Matrix* dest, const Matrix* src);/*矩阵转置函数*/ 
MatrixError matrix_augment(Matrix* dest, const Matrix* A, const Matrix* B, char direction);/* 矩阵增广函数 */
MatrixError matrix_qr_decompose(const Matrix* A, Matrix* Q, Matrix* R);/* QR分解算法 */
MatrixError matrix_symmetric_diagonalize(const Matrix* A, Matrix* Q, Matrix* D, int max_iterations, double tolerance);
/*基于QR分解的对称矩阵相似对角化函数*/
MatrixError matrix_set_zero(Matrix* mat);/*矩阵全置0函数*/
MatrixError matrix_frobenius_norm(const Matrix* mat, double* norm);/*矩阵的F-范数计算方法*/
MatrixError matrix_condition_number(const Matrix* A, double* cond);/*基于F-范数，计算矩阵条件数*/
MatrixError matrix_print_only(Matrix* A);/*矩阵打印到屏幕*/
void matrix_print(const char* name, const Matrix* mat);/*矩阵打印到屏幕1.0*/

#endif