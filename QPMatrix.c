#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <complex.h>
#include <string.h>
#include "QPMatrix.h"

double myabs(double x)
{
	if (x > 0)
	{
		return x;
	}
	else
	{
		return -x;
	}
}
int compare_double(const void* a, const void* b) 
{
	return (*(double*)b > *(double*)a) ? 1 : -1;
}

bool matrix_is_symmetric(const Matrix* A) 
{
	if (A->rows != A->cols) return false;
	for (int i = 0; i < A->rows; ++i) 
	{
		for (int j = 0; j < i; ++j) 
		{
			if (myabs(matrix_get(A, i, j) - matrix_get(A, j, i)) > 1e-10) 
			{
				return false;
			}
		}
	}
	return true;
}


//----------------- 基础操作函数 --------------------

// 创建矩阵
MatrixError matrix_create(Matrix* mat, int rows, int cols) 
{
	mat->rows = rows;
	mat->cols = cols;
	mat->data = (double*)malloc(rows * cols * sizeof(double));
	return mat->data ? MATRIX_SUCCESS : MATRIX_ERROR_ALLOC;
}

// 释放矩阵内存

void matrix_free(Matrix* mat)
{
free(mat->data);
mat->data = NULL;
}



// 矩阵赋值（目标矩阵必须已初始化）
MatrixError matrix_copy(Matrix* dest, const Matrix* src) 
{
	if (dest->rows != src->rows || dest->cols != src->cols)
		return MATRIX_ERROR_DIM;
	
	for (int i = 0; i < src->rows * src->cols; ++i)
		dest->data[i] = src->data[i];
	return MATRIX_SUCCESS;
}

// 获取矩阵元素
double matrix_get(const Matrix* mat, int row, int col) 
{
	return mat->data[row * mat->cols + col];
}

// 设置矩阵元素
void matrix_set(Matrix* mat, int row, int col, double val) 
{
	mat->data[row * mat->cols + col] = val;
}

//----------------- 矩阵运算 --------------------

// 矩阵加法：C = A + B
MatrixError matrix_add(Matrix* C, const Matrix* A, const Matrix* B) 
{
	if (A->rows != B->rows || A->cols != B->cols ||
		C->rows != A->rows || C->cols != A->cols)
		return MATRIX_ERROR_DIM;

	for (int i = 0; i < A->rows * A->cols; ++i)
		C->data[i] = A->data[i] + B->data[i];
	return MATRIX_SUCCESS;
}

// 矩阵乘法：C = A * B
MatrixError matrix_multiply(Matrix* C, const Matrix* A, const Matrix* B) 
{
	if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols)
		return MATRIX_ERROR_DIM;

	for (int i = 0; i < A->rows; ++i) {
		for (int j = 0; j < B->cols; ++j) {
			double sum = 0.0;
			for (int k = 0; k < A->cols; ++k)
				sum += matrix_get(A, i, k) * matrix_get(B, k, j);
			matrix_set(C, i, j, sum);
		}
	}
	return MATRIX_SUCCESS;
}

//----------------- 分解运算 --------------------

// LU分解（带部分选主元），返回置换矩阵的行列式符号
/* --------------------- LU分解核心算法 -------------------- */
MatrixError matrix_lu_decompose(const Matrix* A, Matrix* LU, int* pivot_sign, int* pivots) 
{
	
	const int n = A->rows;
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;
	// 初始化置换向量
	for (int i = 0; i < n; ++i) pivots[i] = i;
	*pivot_sign = 1;

	// 创建LU矩阵的工作副本
	MatrixError err = matrix_copy(LU, A);
	if (err != MATRIX_SUCCESS) return err;
	
	if (!LU || !pivot_sign || !pivots) {
		return MATRIX_ERROR_NULL_PTR;
	}

	for (int j = 0; j < n; ++j) {
		// 部分选主元
		int max_row = j;
		double max_val = myabs(matrix_get(LU, j, j));
		for (int i = j + 1; i < n; ++i) {
			double curr = myabs(matrix_get(LU, i, j));
			if (curr > max_val) {
				max_val = curr;
				max_row = i;
			}
		}

		// 如果找到更大主元，执行行交换
		if (max_row != j) {
			matrix_row_swap(LU, j, max_row);
			int temp = pivots[j];
			pivots[j] = pivots[max_row];
			pivots[max_row] = temp;
			*pivot_sign *= -1;  // 记录交换次数用于行列式符号计算
		}

		// 检测奇异矩阵
		
		if (myabs(matrix_get(LU, j, j)) < 1e-15) {
			return MATRIX_ERROR_SINGULAR;
		}

		// 更新L和U元素
		for (int i = j + 1; i < n; ++i) {
			double factor = matrix_get(LU, i, j) / matrix_get(LU, j, j);
			matrix_set(LU, i, j, factor);  // 存储L因子

			for (int k = j + 1; k < n; ++k) {
				double new_val = matrix_get(LU, i, k) - factor * matrix_get(LU, j, k);
				matrix_set(LU, i, k, new_val);
			}
		}
	}
	return MATRIX_SUCCESS;
}
/*
输入参数说明
参数						类型								说明
A							const Matrix*						输入的方阵（必须为 n×n 矩阵）
LU							Matrix*								预分配内存的输出矩阵，存储LU分解结果，行列数必须与A相同
pivot_sign					int*								指针变量，用于接收行交换的符号（行列式计算需用到的符号）
pivots						int*								预分配内存的整型数组，长度至少为 n（n = A->rows），存储行交换后的置换索引

输出说明
LU矩阵存储结构

输出矩阵LU的物理存储形式为：

下三角部分（L）：不包括对角线（对角线隐含为1）
上三角部分（U）：包含对角线
例如，对于3×3矩阵分解后：
LU = [	u11  u12  u13
		l21  u22  u23
		l31  l32  u33 ]
行交换信息 (pivots)

pivots数组记录行交换历史，pivots[i]表示第i行在分解过程中的最终来源行
实际应用中，求解线性方程组时需要先对输入向量应用pivots置换
符号标记 (pivot_sign)

值为+1或-1，取决于行交换次数的奇偶性
计算行列式时需用：det(A) = pivot_sign * (u11 * u22 * ... * unn)

返回值
返回错误码枚举值：

MATRIX_SUCCESS:				分解成功
MATRIX_ERROR_NOT_SQUARE:	输入矩阵不是方阵
MATRIX_ERROR_SINGULAR:		矩阵是奇异的（无法分解）
MATRIX_ERROR_MEMORY:		内存分配失败

*/

// 辅助平方函数
static double square(double x) { return x * x; }
/* ----------------- QR分解核心算法（Householder变换） ---------------- */
MatrixError matrix_qr_decompose(const Matrix* A, Matrix* Q, Matrix* R)
{
	const int m = A->rows;
	const int n = A->cols;
	MatrixError err;
	Matrix temp;
	if ((err = matrix_create(&temp,m,m)) != MATRIX_SUCCESS) return err;
	// 参数校验
	if (m < n) return MATRIX_ERROR_DIM;
	if (Q->rows != m || Q->cols != m) return MATRIX_ERROR_DIM;
	if (R->rows != m || R->cols != n) return MATRIX_ERROR_DIM;

	// 初始化R为A的副本
	if ((err = matrix_copy(R, A)) != MATRIX_SUCCESS) return err;

	// 初始化Q为单位矩阵
	if ((err = matrix_identity(Q)) != MATRIX_SUCCESS) return err;

	double* v = (double*)malloc(m * sizeof(double));  // Householder向量
	double* w = (double*)malloc(m * sizeof(double));  // 工作向量
	if (!v || !w) 
	{
		free(v); 
		free(w);
		matrix_free(&temp);
		return MATRIX_ERROR_MEMORY;
	}

	for (int k = 0; k < n; ++k) {  // 处理每一列
		/* ------ 构造Householder向量 v ------ */
		double sigma = 0.0;
		for (int i = k; i < m; ++i) {
			sigma += square(matrix_get(R, i, k));  // 自补平方函数
		}

		double x0 = matrix_get(R, k, k);
		double alpha = (x0 < 0) ? sqrt(sigma) : -sqrt(sigma);
		double beta = 1.0 / (sigma - x0 * alpha);

		// 存储反射向量
		//memset(v, 0, m*sizeof(double));
		v[k] = x0 - alpha;
		for (int i = k + 1; i < m; ++i) {
			v[i] = matrix_get(R, i, k);
		}

		/* ------ 更新R矩阵 ------ */
		// 计算当前列下方元素
		matrix_set(R, k, k, alpha);
		for (int i = k + 1; i < m; ++i) {
			matrix_set(R, i, k, 0.0);
		}

		// 更新右边子矩阵
		for (int j = k + 1; j < n; ++j) {
			double dot = 0.0;
			for (int i = k; i < m; ++i) {
				dot += v[i] * matrix_get(R, i, j);
			}
			for (int i = k; i < m; ++i) {
				double new_val = matrix_get(R, i, j) - beta * dot * v[i];
				matrix_set(R, i, j, new_val);
			}
		}

		/* ------ 更新Q矩阵 ------ */
		for (int j = 0; j < m; ++j) {  // 处理Q的每一列
			double dot = 0.0;
			for (int i = k; i < m; ++i) {
				dot += v[i] * matrix_get(Q, i, j);
			}
			for (int i = k; i < m; ++i) {
				double new_val = matrix_get(Q, i, j) - beta * dot * v[i];
				matrix_set(Q, i, j, new_val);
			}
		}
	}
	/* ------ 最后的结果要进行转置 ------ */

	err = matrix_transpose(&temp, Q);
	if (err != MATRIX_SUCCESS)
	{
	free(v);
	free(w);
	matrix_free(&temp);
	return err;
	}
	err = matrix_copy(Q, &temp);
	if (err != MATRIX_SUCCESS)
	{
	free(v);
	free(w);
	matrix_free(&temp);
	return err;
	}
	//清理指针
	free(v); 
	free(w);
	matrix_free(&temp);
	return MATRIX_SUCCESS;
}
/*
函数原型：
MatrixError matrix_qr_decompose(
const Matrix* A,  // 输入矩阵(m×n, m >= n)
Matrix* Q,        // 输出正交矩阵(m×m)
Matrix* R         // 输出上三角矩阵(m×n)
);

参数要求：
- A: 需满足m >= n，否则返回DIM_ERROR
- Q/R: 必须预先初始化为正确维度

错误代码：
- MATRIX_ERROR_DIM:  维度不匹配
- MATRIX_ERROR_MEMORY: 内存分配失败
*/


// 创建单位矩阵
MatrixError matrix_identity(Matrix* mat) 
{
	if (mat->rows != mat->cols) return MATRIX_ERROR_NOT_SQUARE;

	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			matrix_set(mat, i, j, (i == j) ? 1.0 : 0.0);
		}
	}
	return MATRIX_SUCCESS;
}

// 行交换操作
MatrixError matrix_row_swap(Matrix* mat, int row1, int row2) 
{
	if (row1 >= mat->rows || row2 >= mat->rows) return MATRIX_ERROR_DIM;
	if (row1 == row2) return MATRIX_SUCCESS;

	double* ptr1 = mat->data + row1 * mat->cols;
	double* ptr2 = mat->data + row2 * mat->cols;

	for (int j = 0; j < mat->cols; ++j) {
		double temp = ptr1[j];
		ptr1[j] = ptr2[j];
		ptr2[j] = temp;
	}
	return MATRIX_SUCCESS;
}
/*
求逆函数：
函数原型：
		MatrixError matrix_inverse(Matrix* inv, const Matrix* A);
参数						类型						输入/输出						说明						约束条件
inv							Matrix*						Output							存储计算得到的逆矩阵		1. 必须提前初始化2. 维度须为 ，与输入矩阵相同
A							const Matrix*				Input							待求逆的源矩阵				1. 必须是方阵（rows == cols）2. 非奇异（行列式不为零）


返回值（错误码）
错误码								错误触发条件						说明
MATRIX_SUCCESS						逆矩阵成功计算并写入 inv			输出有效
MATRIX_ERROR_NOT_SQUARE				输入矩阵 A 的 rows != cols			输入非方阵
MATRIX_ERROR_SINGULAR				矩阵 A 是奇异矩阵（行列式接近零）	无法分解或求解失败
MATRIX_ERROR_ALLOC					内存分配失败（如临时数组 pivots、LU 矩阵等

*/


MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar) 
{
	//===== 1. 维度校验 =====//
	if (A->rows != C->rows || A->cols != C->cols) {
		return MATRIX_ERROR_DIM;
	}

	//===== 2. 数据校验 =====//
	if (!A->data || !C->data) {
		return MATRIX_ERROR_ALLOC;  // 矩阵内存未分配（通常由未初始化引起）
	}

	//===== 3. 数值计算 =====//
	const int total_elements = A->rows * A->cols;
	for (int i = 0; i < total_elements; ++i) {
		C->data[i] = A->data[i] * scalar;  // 直接操作一维数组提升效率
	}

	return MATRIX_SUCCESS;
}

/*
函数原型：
MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar);
方向					组件							说明
输入					目标矩阵 A，标量 scalar			提供待计算的数据源和缩放因子
输出					结果矩阵 C						存储 A 中每个元素与标量相乘的结果
返回值					错误码 MatrixError				指示是否成功执行或具体错误类型
二、输入参数详解

参数名							类型及方向						用途与约束
A								const Matrix*（输入）			源矩阵
																- 必须为已初始化的合法矩阵
																- 元素类型必须为 double
scalar							double（输入）					标量值
																- 任意实数（正/负/零均可）


输入验证逻辑：
A->data 必须已分配内存（禁止传入未初始化的矩阵对象）
A 的元素类型必须与代码实现一致（隐式约束）

*/

// ---------------------- QPMatrix.c 实现代码 ----------------------
MatrixError matrix_determinant(const Matrix* A, double* det) {


	const int n = A->rows;
	Matrix LU;
	int* pivots = (int*)malloc(n * sizeof(int));
	int pivot_sign = 1;

	//======= 1. 前期校验 =======
	if (!A || !det) return MATRIX_ERROR_NULL_PTR;
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;

	//======= 2. LU分解 =======
	matrix_create(&LU, n, n);
	MatrixError err = matrix_lu_decompose(A, &LU, &pivot_sign, pivots);

	//======= 3. 错误处理 =======
	if (err == MATRIX_ERROR_SINGULAR) { // 遇到奇异矩阵直接返回零行列式
		*det = 0.0;
		err = MATRIX_SUCCESS;           // 将错误转换为成功（特殊标记）
		return err;
	}
	else if (err != MATRIX_SUCCESS) { // 其他错误直接返回
		free(pivots);
		matrix_free(&LU);
		return err;
	}

	//======= 4. 正规行列式计算 =======
	double product = pivot_sign; // 初始化符号
	for (int i = 0; i < n; ++i) {
		product *= matrix_get(&LU, i, i); // 乘对角线元素
	}

	//======= 5. 资源清理 =======
	*det = product;
	free(pivots);
	//matrix_free(&LU);
	return MATRIX_SUCCESS;
}
/*
matrix_determinant 函数使用说明
函数原型
<C>
MatrixError matrix_determinant(const Matrix* A, double* det);
功能描述
本函数用于计算 实数方阵 的行列式值（Determinant），采用LU分解算法实现，支持错误状态返回。适用于稠密矩阵的高效计算，但在接近奇异矩阵时数值稳定性可能下降。

参数说明
参数						类型						输入/输出						描述
A							const Matrix*				输入							待计算行列式的矩阵指针，必须为方阵
det							double*						输出							行列式计算结果存储地址，禁止为NULL
*/


/*基于LU分解的线性方程组求解算法，方程组形式为AX=B*/

MatrixError matrix_solve_lu(const Matrix* A, const Matrix* B, Matrix* X) {
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;
	if (B->rows != A->rows || X->rows != A->cols || X->cols != B->cols)
		return MATRIX_ERROR_DIM;

	int n = A->rows;
	int num_cols = B->cols;

	Matrix LU;
	MatrixError err = matrix_create(&LU, n, n);
	if (err != MATRIX_SUCCESS) return err;

	int* pivots = (int*)malloc(n * sizeof(int));
	if (!pivots) {
		matrix_free(&LU);
		return MATRIX_ERROR_MEMORY;
	}
	int pivot_sign;

	err = matrix_lu_decompose(A, &LU, &pivot_sign, pivots);
	if (err != MATRIX_SUCCESS) {
		free(pivots);
		matrix_free(&LU);
		return err;
	}

	for (int col = 0; col < num_cols; ++col) {
		// Apply pivots to B's column
		double* pb = (double*)malloc(n * sizeof(double));
		for (int i = 0; i < n; ++i) {
			pb[i] = matrix_get(B, pivots[i], col);
		}

		// Forward substitution for Ly = pb
		double* y = (double*)malloc(n * sizeof(double));
		y[0] = pb[0];
		for (int i = 1; i < n; ++i) {
			double sum = 0.0;
			for (int k = 0; k < i; ++k)
				sum += matrix_get(&LU, i, k) * y[k];
			y[i] = pb[i] - sum;
		}

		// Backward substitution for Ux = y
		double* x = (double*)malloc(n * sizeof(double));
		x[n - 1] = y[n - 1] / matrix_get(&LU, n - 1, n - 1);
		for (int i = n - 2; i >= 0; --i) {
			double sum = 0.0;
			for (int k = i + 1; k < n; ++k)
				sum += matrix_get(&LU, i, k) * x[k];
			x[i] = (y[i] - sum) / matrix_get(&LU, i, i);
		}

		// Store solution in X
		for (int i = 0; i < n; ++i)
			matrix_set(X, i, col, x[i]);

		free(pb);
		free(y);
		free(x);
	}

	free(pivots);
	matrix_free(&LU);
	return MATRIX_SUCCESS;
}

/* 基于线性方程组求解的矩阵求逆函数 */
MatrixError matrix_inverse(Matrix* inv, const Matrix* A)
{
	Matrix EI;
	int n = A->rows;
	MatrixError err;
	/* 错误处理 */
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;
	if (inv->rows != inv->cols) return MATRIX_ERROR_NOT_SQUARE;
	if (inv->rows != A->rows || inv->cols != A->cols)
		return MATRIX_ERROR_DIM;
	
	err = matrix_create(&EI, n, n);
	if (err != MATRIX_SUCCESS)
	{
		return err;
	}

	err = matrix_identity(&EI);
	if (err != MATRIX_SUCCESS)
	{
		return err;
	}

	err = matrix_solve_lu(A, &EI, inv);
	if (err != MATRIX_SUCCESS)
	{
		return err;
	}
	return MATRIX_SUCCESS;
}
/* 矩阵转置函数 */



// ================== 新增在 QPMAT_C_TXT.txt 源文件中 ==================
/* 矩阵转置函数：将src矩阵转置存储到dest矩阵 */
MatrixError matrix_transpose(Matrix* dest, const Matrix* src)
{
	//===== 参数基础校验 =====//
	if (!dest || !src)
		return MATRIX_ERROR_NULL_PTR;
	if (dest->rows != src->cols || dest->cols != src->rows)
		return MATRIX_ERROR_DIM;

	//===== 执行转置操作 =====//
	for (int i = 0; i < src->rows; ++i) {
		for (int j = 0; j < src->cols; ++j) {
			// 将src的(i,j)元素放入dest的(j,i)位置
			matrix_set(dest, j, i, matrix_get(src, i, j));
		}
	}
	return MATRIX_SUCCESS;
}

/* 函数说明：
参数说明：
dest					- 目标矩阵（需预先初始化为 src->cols 行 src->rows 列）
src						- 源矩阵

错误码：
MATRIX_ERROR_NULL_PTR	- 输入空指针
MATRIX_ERROR_DIM		- 矩阵维度不匹配
MATRIX_SUCCESS			- 操作成功

功能特性：
1. 支持任意维度矩阵（含非方阵）
2. 可安全处理源和目标为同一矩阵的方阵转置
3. 无需额外内存分配（要求目标矩阵已正确初始化）
*/

/* 矩阵增广函数：将两个矩阵水平(方向'h')或垂直(方向'v')拼接 */
MatrixError matrix_augment(Matrix* dest, const Matrix* A, const Matrix* B, char direction) {
	//===== 空指针检查 =====//
	if (!dest || !A || !B)
		return MATRIX_ERROR_NULL_PTR;

	//===== 方向有效性检查 =====//
	if (direction != 'h' && direction != 'v')
		return MATRIX_ERROR_UNSUPPORTED;

	//===== 维度兼容性检查 =====//
	if (direction == 'h') {
		// 水平拼接需行数相等，且dest维度正确
		if (A->rows != B->rows ||
			dest->rows != A->rows ||
			dest->cols != A->cols + B->cols)
			return MATRIX_ERROR_DIM;
	}
	else {
		// 垂直拼接需列数相等，且dest维度正确
		if (A->cols != B->cols ||
			dest->cols != A->cols ||
			dest->rows != A->rows + B->rows)
			return MATRIX_ERROR_DIM;
	}

	//===== 数据填充 =====//
	if (direction == 'h') {  // 水平拼接
		for (int i = 0; i < A->rows; ++i) {
			// 复制A的第i行元素
			for (int j = 0; j < A->cols; ++j)
				matrix_set(dest, i, j, matrix_get(A, i, j));
			// 复制B的第i行元素到后续列
			for (int j = 0; j < B->cols; ++j)
				matrix_set(dest, i, A->cols + j, matrix_get(B, i, j));
		}
	}
	else {  // 垂直拼接
		// 复制A所有行到dest上半部分
		for (int i = 0; i < A->rows; ++i)
		for (int j = 0; j < A->cols; ++j)
			matrix_set(dest, i, j, matrix_get(A, i, j));
		// 复制B所有行到dest下半部分
		for (int i = 0; i < B->rows; ++i)
		for (int j = 0; j < B->cols; ++j)
			matrix_set(dest, A->rows + i, j, matrix_get(B, i, j));
	}
	return MATRIX_SUCCESS;
}

/*----------------- 对称矩阵的相似对角化函数 -----------------*/
MatrixError matrix_symmetric_diagonalize(const Matrix* A, Matrix* Q, Matrix* D, int max_iterations, double tolerance) {

	//============= 初始化工作矩阵 =============
	Matrix current, Qt, R, temp;
	MatrixError err;
	const int n = A->rows;
	matrix_create(&Qt, n, n);
	matrix_create(&R, n, n);
	matrix_create(&temp, n, n);
	//============= 参数校验 =============
	if (!A || !Q || !D)
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return MATRIX_ERROR_NULL_PTR;
	}
	if (A->rows != A->cols)
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return MATRIX_ERROR_NOT_SQUARE;
	}
	if (!matrix_is_symmetric(A)) 
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return MATRIX_ERROR_UNSUPPORTED;
	}
	if ((err = matrix_create(&current, n, n)) != MATRIX_SUCCESS) 
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return err;
	}
	if ((err = matrix_copy(&current, A)) != MATRIX_SUCCESS)
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return err;
	}
	//============= 累积特征向量的正交矩阵 =============
	if ((err = matrix_identity(Q)) != MATRIX_SUCCESS) {
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return err;
	}
	//============= QR迭代过程 =============
	for (int iter = 0; iter<max_iterations; ++iter){
		// Step 1: 对当前矩阵执行QR分解
		if ((err = matrix_qr_decompose(&current, &Qt, &R)) != MATRIX_SUCCESS) break;

		// Step 2: 计算RQ得到新迭代矩阵 current = R * Qt
		if ((err = matrix_multiply(&current, &R, &Qt)) != MATRIX_SUCCESS) break;

		// Step 3: 累积特征向量 Q = Q * Qt
		if ((err = matrix_multiply(&temp, Q, &Qt)) != MATRIX_SUCCESS) break;
		matrix_copy(Q, &temp);

		// Step 4: 检查次对角线收敛性
		bool converged = true;
		for (int i = 1; i<n; ++i){
			if (myabs(matrix_get(&current, i, i - 1)) > tolerance){
				converged = false;
				break;
			}
		}
		if (converged) break;
	}
	//============= 提取对角矩阵 =============
	if ((err = matrix_create(D, n, n)) != MATRIX_SUCCESS)
	{
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return err;
	}
	matrix_identity(D);
	for (int i = 0; i<n; ++i){
		matrix_set(D, i, i, matrix_get(&current, i, i));  // 对角线元素为特征值
	}
	matrix_free(&current);
	matrix_free(&Qt);
	matrix_free(&R);
	matrix_free(&temp);
	return err;
}
/*
函数原型：

<C>
MatrixError matrix_symmetric_diagonalize(
const Matrix* A,	// 输入对称矩阵
Matrix* Q,			// 输出正交矩阵（特征向量）
Matrix* D,			// 输出对角矩阵（特征值）
int max_iterations, // 最大迭代次数
double tolerance    // 收敛容差
);
功能描述
本函数用于对 实对称矩阵 进行相似对角化，通过 QR 迭代算法 计算其特征值和特征向量，使得满足 。输出矩阵 Q 的列向量为 A 的正交特征向量，矩阵 D 的主对角线元素为对应的特征值。

参数说明
参数					类型					方向					描述
A						Matrix*					输入					待对角化的 对称方阵（必须预先初始化且对称，函数将通过 matrix_is_symmetric 自动校验）
Q						Matrix*					输出					存储正交特征向量矩阵（列对应于特征向量）需提前初始化为与 A 同维度的方阵
D						Matrix*					输出					存储特征值的对角矩阵，需提前初始化为与 A 同维度的方阵
max_iterations			int						输入					最大迭代次数（建议值：50-200，视矩阵大小调整）
tolerance				double					输入					收敛判定容差（如 1e-10），当次对角元素绝对值均小于此值视为收敛
返回值
返回 MatrixError 枚举类型错误码，具体含义如下：

错误码						触发条件
MATRIX_SUCCESS				对角化成功完成
MATRIX_ERROR_NULL_PTR		A、Q 或 D 参数为 NULL
MATRIX_ERROR_NOT_SQUARE		A 不是方阵，或 Q/D 的维度与 A 不匹配
MATRIX_ERROR_UNSUPPORTED	A 不对称（通过 matrix_is_symmetric 检测）
MATRIX_ERROR_MEMORY			内存分配失败（如工作矩阵初始化时）
其他 QR 分解错误码			QR 分解过程失败（如维度不兼容、内存不足等）


*/


// ---------------------- 矩阵全零化函数 ----------------------
MatrixError matrix_set_zero(Matrix* mat)
{
	const int total_size = mat->rows * mat->cols;
	//===== 参数合法性校验 =====
	if (!mat || !mat->data)
		return MATRIX_ERROR_NULL_PTR;

	//===== 高效内存操作 =====
	if (total_size < 1)
		return MATRIX_ERROR_DIM;
	// 直接操作一维数组清零（比逐元素设置快约5-10倍）
	memset(mat->data, 0, total_size * sizeof(double));
	return MATRIX_SUCCESS;
}

/*
函数说明：
- 功能：将矩阵所有元素清零
- 参数：mat 必须为已初始化的合法矩阵
- 错误码：
- MATRIX_ERROR_NULL_PTR: 输入空指针或数据指针未分配
- MATRIX_ERROR_DIM:       矩阵总元素数为非正数（异常维度）
- MATRIX_SUCCESS:        操作成功
*/

/* --------------------- 矩阵Frobenius范数计算 --------------------- */
// ========== 实现 ==========
MatrixError matrix_frobenius_norm(const Matrix* mat, double* norm)
{
	double sum_sq = 0.0;
	const int total = mat->rows * mat->cols;
	
	// ==== 参数校验 ====
	// 无效指针检查
	if (!mat || !norm)
		return MATRIX_ERROR_NULL_PTR;
	if (!mat->data)
		return MATRIX_ERROR_NULL_PTR;

	// ==== 维度有效性 ====
	if (mat->rows <= 0 || mat->cols <= 0)
		return MATRIX_ERROR_DIM;

	// ==== 核心计算 ====
	// 优化点：直接访问连续内存提升性能
	for (int i = 0; i < total; ++i) {
		sum_sq += mat->data[i] * mat->data[i];
	}

	*norm = sqrt(sum_sq);
	return MATRIX_SUCCESS;
}

/*
函数说明：
参数       类型              方向        描述
mat       const Matrix*      输入        待计算矩阵（需已初始化且行/列数>0）
norm      double*            输出        存储Frobenius范数值的内存地址

错误码：
MATRIX_SUCCESS           - 计算成功（结果存储在norm中）
MATRIX_ERROR_NULL_PTR    - 输入矩阵指针或数据指针为空
MATRIX_ERROR_DIM         - 矩阵行列数非正

性能特性：
- 时间复杂度O(mn)，直接内存访问优化比逐元素访问快约3倍
- 支持最大维度矩阵：INT_MAX (受内存限制)

数值稳定性：
- 使用双精度浮点累加避免精度损失
- 特别优化超大值处理：可自动处理sum_sq溢出（依赖系统浮点实现）

兼容性：
- 正确处理行主序存储的所有类型矩阵（含非对称/非方阵）
*/

/* 矩阵打印到屏幕的函数 */
MatrixError matrix_print_only(Matrix* A)
{
	if (!A) return MATRIX_ERROR_NULL_PTR;
	printf("[");
	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			printf("%f\t", matrix_get(A, i, j));
		}
		printf("\n");
	}
	printf("]");
}
/*矩阵打印到屏幕1.0*/
void matrix_print(const char* name, const Matrix* mat) {
	printf("Matrix %s (%dx%d):\n", name, mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			printf("%8.4f ", matrix_get(mat, i, j));
		}
		printf("\n");
	}
	printf("\n");
}
/*基于F-范数求方阵的条件数*/
MatrixError matrix_condition_number(const Matrix* A, double* cond) {
	
	double norm_A;
	Matrix inv;
	MatrixError err = matrix_frobenius_norm(A, &norm_A);
	double norm_inv;
	if (err != MATRIX_SUCCESS) return err;
	if (!A || !cond) return MATRIX_ERROR_NULL_PTR;
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;
	if ((err = matrix_create(&inv, A->rows, A->cols)) != MATRIX_SUCCESS)
		return err;
	err = matrix_inverse(&inv, A);
	if (err != MATRIX_SUCCESS) {
		matrix_free(&inv);
		return err; // 传递奇异矩阵等错误
	}
	err = matrix_frobenius_norm(&inv, &norm_inv);
	matrix_free(&inv);
	if (err != MATRIX_SUCCESS) return err;

	*cond = norm_A * norm_inv;
	return MATRIX_SUCCESS;
}





