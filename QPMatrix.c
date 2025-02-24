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


//----------------- ������������ --------------------

// ��������
MatrixError matrix_create(Matrix* mat, int rows, int cols) 
{
	mat->rows = rows;
	mat->cols = cols;
	mat->data = (double*)malloc(rows * cols * sizeof(double));
	return mat->data ? MATRIX_SUCCESS : MATRIX_ERROR_ALLOC;
}

// �ͷž����ڴ�

void matrix_free(Matrix* mat)
{
free(mat->data);
mat->data = NULL;
}



// ����ֵ��Ŀ���������ѳ�ʼ����
MatrixError matrix_copy(Matrix* dest, const Matrix* src) 
{
	if (dest->rows != src->rows || dest->cols != src->cols)
		return MATRIX_ERROR_DIM;
	
	for (int i = 0; i < src->rows * src->cols; ++i)
		dest->data[i] = src->data[i];
	return MATRIX_SUCCESS;
}

// ��ȡ����Ԫ��
double matrix_get(const Matrix* mat, int row, int col) 
{
	return mat->data[row * mat->cols + col];
}

// ���þ���Ԫ��
void matrix_set(Matrix* mat, int row, int col, double val) 
{
	mat->data[row * mat->cols + col] = val;
}

//----------------- �������� --------------------

// ����ӷ���C = A + B
MatrixError matrix_add(Matrix* C, const Matrix* A, const Matrix* B) 
{
	if (A->rows != B->rows || A->cols != B->cols ||
		C->rows != A->rows || C->cols != A->cols)
		return MATRIX_ERROR_DIM;

	for (int i = 0; i < A->rows * A->cols; ++i)
		C->data[i] = A->data[i] + B->data[i];
	return MATRIX_SUCCESS;
}

// ����˷���C = A * B
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

//----------------- �ֽ����� --------------------

// LU�ֽ⣨������ѡ��Ԫ���������û����������ʽ����
/* --------------------- LU�ֽ�����㷨 -------------------- */
MatrixError matrix_lu_decompose(const Matrix* A, Matrix* LU, int* pivot_sign, int* pivots) 
{
	
	const int n = A->rows;
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;
	// ��ʼ���û�����
	for (int i = 0; i < n; ++i) pivots[i] = i;
	*pivot_sign = 1;

	// ����LU����Ĺ�������
	MatrixError err = matrix_copy(LU, A);
	if (err != MATRIX_SUCCESS) return err;
	
	if (!LU || !pivot_sign || !pivots) {
		return MATRIX_ERROR_NULL_PTR;
	}

	for (int j = 0; j < n; ++j) {
		// ����ѡ��Ԫ
		int max_row = j;
		double max_val = myabs(matrix_get(LU, j, j));
		for (int i = j + 1; i < n; ++i) {
			double curr = myabs(matrix_get(LU, i, j));
			if (curr > max_val) {
				max_val = curr;
				max_row = i;
			}
		}

		// ����ҵ�������Ԫ��ִ���н���
		if (max_row != j) {
			matrix_row_swap(LU, j, max_row);
			int temp = pivots[j];
			pivots[j] = pivots[max_row];
			pivots[max_row] = temp;
			*pivot_sign *= -1;  // ��¼����������������ʽ���ż���
		}

		// ����������
		
		if (myabs(matrix_get(LU, j, j)) < 1e-15) {
			return MATRIX_ERROR_SINGULAR;
		}

		// ����L��UԪ��
		for (int i = j + 1; i < n; ++i) {
			double factor = matrix_get(LU, i, j) / matrix_get(LU, j, j);
			matrix_set(LU, i, j, factor);  // �洢L����

			for (int k = j + 1; k < n; ++k) {
				double new_val = matrix_get(LU, i, k) - factor * matrix_get(LU, j, k);
				matrix_set(LU, i, k, new_val);
			}
		}
	}
	return MATRIX_SUCCESS;
}
/*
�������˵��
����						����								˵��
A							const Matrix*						����ķ��󣨱���Ϊ n��n ����
LU							Matrix*								Ԥ�����ڴ��������󣬴洢LU�ֽ�����������������A��ͬ
pivot_sign					int*								ָ����������ڽ����н����ķ��ţ�����ʽ�������õ��ķ��ţ�
pivots						int*								Ԥ�����ڴ���������飬��������Ϊ n��n = A->rows�����洢�н�������û�����

���˵��
LU����洢�ṹ

�������LU������洢��ʽΪ��

�����ǲ��֣�L�����������Խ��ߣ��Խ�������Ϊ1��
�����ǲ��֣�U���������Խ���
���磬����3��3����ֽ��
LU = [	u11  u12  u13
		l21  u22  u23
		l31  l32  u33 ]
�н�����Ϣ (pivots)

pivots�����¼�н�����ʷ��pivots[i]��ʾ��i���ڷֽ�����е�������Դ��
ʵ��Ӧ���У�������Է�����ʱ��Ҫ�ȶ���������Ӧ��pivots�û�
���ű�� (pivot_sign)

ֵΪ+1��-1��ȡ�����н�����������ż��
��������ʽʱ���ã�det(A) = pivot_sign * (u11 * u22 * ... * unn)

����ֵ
���ش�����ö��ֵ��

MATRIX_SUCCESS:				�ֽ�ɹ�
MATRIX_ERROR_NOT_SQUARE:	��������Ƿ���
MATRIX_ERROR_SINGULAR:		����������ģ��޷��ֽ⣩
MATRIX_ERROR_MEMORY:		�ڴ����ʧ��

*/

// ����ƽ������
static double square(double x) { return x * x; }
/* ----------------- QR�ֽ�����㷨��Householder�任�� ---------------- */
MatrixError matrix_qr_decompose(const Matrix* A, Matrix* Q, Matrix* R)
{
	const int m = A->rows;
	const int n = A->cols;
	MatrixError err;
	Matrix temp;
	if ((err = matrix_create(&temp,m,m)) != MATRIX_SUCCESS) return err;
	// ����У��
	if (m < n) return MATRIX_ERROR_DIM;
	if (Q->rows != m || Q->cols != m) return MATRIX_ERROR_DIM;
	if (R->rows != m || R->cols != n) return MATRIX_ERROR_DIM;

	// ��ʼ��RΪA�ĸ���
	if ((err = matrix_copy(R, A)) != MATRIX_SUCCESS) return err;

	// ��ʼ��QΪ��λ����
	if ((err = matrix_identity(Q)) != MATRIX_SUCCESS) return err;

	double* v = (double*)malloc(m * sizeof(double));  // Householder����
	double* w = (double*)malloc(m * sizeof(double));  // ��������
	if (!v || !w) 
	{
		free(v); 
		free(w);
		matrix_free(&temp);
		return MATRIX_ERROR_MEMORY;
	}

	for (int k = 0; k < n; ++k) {  // ����ÿһ��
		/* ------ ����Householder���� v ------ */
		double sigma = 0.0;
		for (int i = k; i < m; ++i) {
			sigma += square(matrix_get(R, i, k));  // �Բ�ƽ������
		}

		double x0 = matrix_get(R, k, k);
		double alpha = (x0 < 0) ? sqrt(sigma) : -sqrt(sigma);
		double beta = 1.0 / (sigma - x0 * alpha);

		// �洢��������
		//memset(v, 0, m*sizeof(double));
		v[k] = x0 - alpha;
		for (int i = k + 1; i < m; ++i) {
			v[i] = matrix_get(R, i, k);
		}

		/* ------ ����R���� ------ */
		// ���㵱ǰ���·�Ԫ��
		matrix_set(R, k, k, alpha);
		for (int i = k + 1; i < m; ++i) {
			matrix_set(R, i, k, 0.0);
		}

		// �����ұ��Ӿ���
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

		/* ------ ����Q���� ------ */
		for (int j = 0; j < m; ++j) {  // ����Q��ÿһ��
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
	/* ------ ���Ľ��Ҫ����ת�� ------ */

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
	//����ָ��
	free(v); 
	free(w);
	matrix_free(&temp);
	return MATRIX_SUCCESS;
}
/*
����ԭ�ͣ�
MatrixError matrix_qr_decompose(
const Matrix* A,  // �������(m��n, m >= n)
Matrix* Q,        // �����������(m��m)
Matrix* R         // ��������Ǿ���(m��n)
);

����Ҫ��
- A: ������m >= n�����򷵻�DIM_ERROR
- Q/R: ����Ԥ�ȳ�ʼ��Ϊ��ȷά��

������룺
- MATRIX_ERROR_DIM:  ά�Ȳ�ƥ��
- MATRIX_ERROR_MEMORY: �ڴ����ʧ��
*/


// ������λ����
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

// �н�������
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
���溯����
����ԭ�ͣ�
		MatrixError matrix_inverse(Matrix* inv, const Matrix* A);
����						����						����/���						˵��						Լ������
inv							Matrix*						Output							�洢����õ��������		1. ������ǰ��ʼ��2. ά����Ϊ �������������ͬ
A							const Matrix*				Input							�������Դ����				1. �����Ƿ���rows == cols��2. �����죨����ʽ��Ϊ�㣩


����ֵ�������룩
������								���󴥷�����						˵��
MATRIX_SUCCESS						�����ɹ����㲢д�� inv			�����Ч
MATRIX_ERROR_NOT_SQUARE				������� A �� rows != cols			����Ƿ���
MATRIX_ERROR_SINGULAR				���� A �������������ʽ�ӽ��㣩	�޷��ֽ�����ʧ��
MATRIX_ERROR_ALLOC					�ڴ����ʧ�ܣ�����ʱ���� pivots��LU �����

*/


MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar) 
{
	//===== 1. ά��У�� =====//
	if (A->rows != C->rows || A->cols != C->cols) {
		return MATRIX_ERROR_DIM;
	}

	//===== 2. ����У�� =====//
	if (!A->data || !C->data) {
		return MATRIX_ERROR_ALLOC;  // �����ڴ�δ���䣨ͨ����δ��ʼ������
	}

	//===== 3. ��ֵ���� =====//
	const int total_elements = A->rows * A->cols;
	for (int i = 0; i < total_elements; ++i) {
		C->data[i] = A->data[i] * scalar;  // ֱ�Ӳ���һά��������Ч��
	}

	return MATRIX_SUCCESS;
}

/*
����ԭ�ͣ�
MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar);
����					���							˵��
����					Ŀ����� A������ scalar			�ṩ�����������Դ����������
���					������� C						�洢 A ��ÿ��Ԫ���������˵Ľ��
����ֵ					������ MatrixError				ָʾ�Ƿ�ɹ�ִ�л�����������
��������������

������							���ͼ�����						��;��Լ��
A								const Matrix*�����룩			Դ����
																- ����Ϊ�ѳ�ʼ���ĺϷ�����
																- Ԫ�����ͱ���Ϊ double
scalar							double�����룩					����ֵ
																- ����ʵ������/��/����ɣ�


������֤�߼���
A->data �����ѷ����ڴ棨��ֹ����δ��ʼ���ľ������
A ��Ԫ�����ͱ��������ʵ��һ�£���ʽԼ����

*/

// ---------------------- QPMatrix.c ʵ�ִ��� ----------------------
MatrixError matrix_determinant(const Matrix* A, double* det) {


	const int n = A->rows;
	Matrix LU;
	int* pivots = (int*)malloc(n * sizeof(int));
	int pivot_sign = 1;

	//======= 1. ǰ��У�� =======
	if (!A || !det) return MATRIX_ERROR_NULL_PTR;
	if (A->rows != A->cols) return MATRIX_ERROR_NOT_SQUARE;

	//======= 2. LU�ֽ� =======
	matrix_create(&LU, n, n);
	MatrixError err = matrix_lu_decompose(A, &LU, &pivot_sign, pivots);

	//======= 3. ������ =======
	if (err == MATRIX_ERROR_SINGULAR) { // �����������ֱ�ӷ���������ʽ
		*det = 0.0;
		err = MATRIX_SUCCESS;           // ������ת��Ϊ�ɹ��������ǣ�
		return err;
	}
	else if (err != MATRIX_SUCCESS) { // ��������ֱ�ӷ���
		free(pivots);
		matrix_free(&LU);
		return err;
	}

	//======= 4. ��������ʽ���� =======
	double product = pivot_sign; // ��ʼ������
	for (int i = 0; i < n; ++i) {
		product *= matrix_get(&LU, i, i); // �˶Խ���Ԫ��
	}

	//======= 5. ��Դ���� =======
	*det = product;
	free(pivots);
	//matrix_free(&LU);
	return MATRIX_SUCCESS;
}
/*
matrix_determinant ����ʹ��˵��
����ԭ��
<C>
MatrixError matrix_determinant(const Matrix* A, double* det);
��������
���������ڼ��� ʵ������ ������ʽֵ��Determinant��������LU�ֽ��㷨ʵ�֣�֧�ִ���״̬���ء������ڳ��ܾ���ĸ�Ч���㣬���ڽӽ��������ʱ��ֵ�ȶ��Կ����½���

����˵��
����						����						����/���						����
A							const Matrix*				����							����������ʽ�ľ���ָ�룬����Ϊ����
det							double*						���							����ʽ�������洢��ַ����ֹΪNULL
*/


/*����LU�ֽ�����Է���������㷨����������ʽΪAX=B*/

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

/* �������Է��������ľ������溯�� */
MatrixError matrix_inverse(Matrix* inv, const Matrix* A)
{
	Matrix EI;
	int n = A->rows;
	MatrixError err;
	/* ������ */
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
/* ����ת�ú��� */



// ================== ������ QPMAT_C_TXT.txt Դ�ļ��� ==================
/* ����ת�ú�������src����ת�ô洢��dest���� */
MatrixError matrix_transpose(Matrix* dest, const Matrix* src)
{
	//===== ��������У�� =====//
	if (!dest || !src)
		return MATRIX_ERROR_NULL_PTR;
	if (dest->rows != src->cols || dest->cols != src->rows)
		return MATRIX_ERROR_DIM;

	//===== ִ��ת�ò��� =====//
	for (int i = 0; i < src->rows; ++i) {
		for (int j = 0; j < src->cols; ++j) {
			// ��src��(i,j)Ԫ�ط���dest��(j,i)λ��
			matrix_set(dest, j, i, matrix_get(src, i, j));
		}
	}
	return MATRIX_SUCCESS;
}

/* ����˵����
����˵����
dest					- Ŀ�������Ԥ�ȳ�ʼ��Ϊ src->cols �� src->rows �У�
src						- Դ����

�����룺
MATRIX_ERROR_NULL_PTR	- �����ָ��
MATRIX_ERROR_DIM		- ����ά�Ȳ�ƥ��
MATRIX_SUCCESS			- �����ɹ�

�������ԣ�
1. ֧������ά�Ⱦ��󣨺��Ƿ���
2. �ɰ�ȫ����Դ��Ŀ��Ϊͬһ����ķ���ת��
3. ��������ڴ���䣨Ҫ��Ŀ���������ȷ��ʼ����
*/

/* �������㺯��������������ˮƽ(����'h')��ֱ(����'v')ƴ�� */
MatrixError matrix_augment(Matrix* dest, const Matrix* A, const Matrix* B, char direction) {
	//===== ��ָ���� =====//
	if (!dest || !A || !B)
		return MATRIX_ERROR_NULL_PTR;

	//===== ������Ч�Լ�� =====//
	if (direction != 'h' && direction != 'v')
		return MATRIX_ERROR_UNSUPPORTED;

	//===== ά�ȼ����Լ�� =====//
	if (direction == 'h') {
		// ˮƽƴ����������ȣ���destά����ȷ
		if (A->rows != B->rows ||
			dest->rows != A->rows ||
			dest->cols != A->cols + B->cols)
			return MATRIX_ERROR_DIM;
	}
	else {
		// ��ֱƴ����������ȣ���destά����ȷ
		if (A->cols != B->cols ||
			dest->cols != A->cols ||
			dest->rows != A->rows + B->rows)
			return MATRIX_ERROR_DIM;
	}

	//===== ������� =====//
	if (direction == 'h') {  // ˮƽƴ��
		for (int i = 0; i < A->rows; ++i) {
			// ����A�ĵ�i��Ԫ��
			for (int j = 0; j < A->cols; ++j)
				matrix_set(dest, i, j, matrix_get(A, i, j));
			// ����B�ĵ�i��Ԫ�ص�������
			for (int j = 0; j < B->cols; ++j)
				matrix_set(dest, i, A->cols + j, matrix_get(B, i, j));
		}
	}
	else {  // ��ֱƴ��
		// ����A�����е�dest�ϰ벿��
		for (int i = 0; i < A->rows; ++i)
		for (int j = 0; j < A->cols; ++j)
			matrix_set(dest, i, j, matrix_get(A, i, j));
		// ����B�����е�dest�°벿��
		for (int i = 0; i < B->rows; ++i)
		for (int j = 0; j < B->cols; ++j)
			matrix_set(dest, A->rows + i, j, matrix_get(B, i, j));
	}
	return MATRIX_SUCCESS;
}

/*----------------- �Գƾ�������ƶԽǻ����� -----------------*/
MatrixError matrix_symmetric_diagonalize(const Matrix* A, Matrix* Q, Matrix* D, int max_iterations, double tolerance) {

	//============= ��ʼ���������� =============
	Matrix current, Qt, R, temp;
	MatrixError err;
	const int n = A->rows;
	matrix_create(&Qt, n, n);
	matrix_create(&R, n, n);
	matrix_create(&temp, n, n);
	//============= ����У�� =============
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
	//============= �ۻ������������������� =============
	if ((err = matrix_identity(Q)) != MATRIX_SUCCESS) {
		matrix_free(&current);
		matrix_free(&Qt);
		matrix_free(&R);
		matrix_free(&temp);
		return err;
	}
	//============= QR�������� =============
	for (int iter = 0; iter<max_iterations; ++iter){
		// Step 1: �Ե�ǰ����ִ��QR�ֽ�
		if ((err = matrix_qr_decompose(&current, &Qt, &R)) != MATRIX_SUCCESS) break;

		// Step 2: ����RQ�õ��µ������� current = R * Qt
		if ((err = matrix_multiply(&current, &R, &Qt)) != MATRIX_SUCCESS) break;

		// Step 3: �ۻ��������� Q = Q * Qt
		if ((err = matrix_multiply(&temp, Q, &Qt)) != MATRIX_SUCCESS) break;
		matrix_copy(Q, &temp);

		// Step 4: ���ζԽ���������
		bool converged = true;
		for (int i = 1; i<n; ++i){
			if (myabs(matrix_get(&current, i, i - 1)) > tolerance){
				converged = false;
				break;
			}
		}
		if (converged) break;
	}
	//============= ��ȡ�ԽǾ��� =============
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
		matrix_set(D, i, i, matrix_get(&current, i, i));  // �Խ���Ԫ��Ϊ����ֵ
	}
	matrix_free(&current);
	matrix_free(&Qt);
	matrix_free(&R);
	matrix_free(&temp);
	return err;
}
/*
����ԭ�ͣ�

<C>
MatrixError matrix_symmetric_diagonalize(
const Matrix* A,	// ����Գƾ���
Matrix* Q,			// ���������������������
Matrix* D,			// ����ԽǾ�������ֵ��
int max_iterations, // ����������
double tolerance    // �����ݲ�
);
��������
���������ڶ� ʵ�Գƾ��� �������ƶԽǻ���ͨ�� QR �����㷨 ����������ֵ������������ʹ������ ��������� Q ��������Ϊ A �������������������� D �����Խ���Ԫ��Ϊ��Ӧ������ֵ��

����˵��
����					����					����					����
A						Matrix*					����					���Խǻ��� �ԳƷ��󣨱���Ԥ�ȳ�ʼ���ҶԳƣ�������ͨ�� matrix_is_symmetric �Զ�У�飩
Q						Matrix*					���					�洢�����������������ж�Ӧ����������������ǰ��ʼ��Ϊ�� A ͬά�ȵķ���
D						Matrix*					���					�洢����ֵ�ĶԽǾ�������ǰ��ʼ��Ϊ�� A ͬά�ȵķ���
max_iterations			int						����					����������������ֵ��50-200���Ӿ����С������
tolerance				double					����					�����ж��ݲ�� 1e-10�������ζԽ�Ԫ�ؾ���ֵ��С�ڴ�ֵ��Ϊ����
����ֵ
���� MatrixError ö�����ʹ����룬���庬�����£�

������						��������
MATRIX_SUCCESS				�Խǻ��ɹ����
MATRIX_ERROR_NULL_PTR		A��Q �� D ����Ϊ NULL
MATRIX_ERROR_NOT_SQUARE		A ���Ƿ��󣬻� Q/D ��ά���� A ��ƥ��
MATRIX_ERROR_UNSUPPORTED	A ���Գƣ�ͨ�� matrix_is_symmetric ��⣩
MATRIX_ERROR_MEMORY			�ڴ����ʧ�ܣ��繤�������ʼ��ʱ��
���� QR �ֽ������			QR �ֽ����ʧ�ܣ���ά�Ȳ����ݡ��ڴ治��ȣ�


*/


// ---------------------- ����ȫ�㻯���� ----------------------
MatrixError matrix_set_zero(Matrix* mat)
{
	const int total_size = mat->rows * mat->cols;
	//===== �����Ϸ���У�� =====
	if (!mat || !mat->data)
		return MATRIX_ERROR_NULL_PTR;

	//===== ��Ч�ڴ���� =====
	if (total_size < 1)
		return MATRIX_ERROR_DIM;
	// ֱ�Ӳ���һά�������㣨����Ԫ�����ÿ�Լ5-10����
	memset(mat->data, 0, total_size * sizeof(double));
	return MATRIX_SUCCESS;
}

/*
����˵����
- ���ܣ�����������Ԫ������
- ������mat ����Ϊ�ѳ�ʼ���ĺϷ�����
- �����룺
- MATRIX_ERROR_NULL_PTR: �����ָ�������ָ��δ����
- MATRIX_ERROR_DIM:       ������Ԫ����Ϊ���������쳣ά�ȣ�
- MATRIX_SUCCESS:        �����ɹ�
*/

/* --------------------- ����Frobenius�������� --------------------- */
// ========== ʵ�� ==========
MatrixError matrix_frobenius_norm(const Matrix* mat, double* norm)
{
	double sum_sq = 0.0;
	const int total = mat->rows * mat->cols;
	
	// ==== ����У�� ====
	// ��Чָ����
	if (!mat || !norm)
		return MATRIX_ERROR_NULL_PTR;
	if (!mat->data)
		return MATRIX_ERROR_NULL_PTR;

	// ==== ά����Ч�� ====
	if (mat->rows <= 0 || mat->cols <= 0)
		return MATRIX_ERROR_DIM;

	// ==== ���ļ��� ====
	// �Ż��㣺ֱ�ӷ��������ڴ���������
	for (int i = 0; i < total; ++i) {
		sum_sq += mat->data[i] * mat->data[i];
	}

	*norm = sqrt(sum_sq);
	return MATRIX_SUCCESS;
}

/*
����˵����
����       ����              ����        ����
mat       const Matrix*      ����        ������������ѳ�ʼ������/����>0��
norm      double*            ���        �洢Frobenius����ֵ���ڴ��ַ

�����룺
MATRIX_SUCCESS           - ����ɹ�������洢��norm�У�
MATRIX_ERROR_NULL_PTR    - �������ָ�������ָ��Ϊ��
MATRIX_ERROR_DIM         - ��������������

�������ԣ�
- ʱ�临�Ӷ�O(mn)��ֱ���ڴ�����Ż�����Ԫ�ط��ʿ�Լ3��
- ֧�����ά�Ⱦ���INT_MAX (���ڴ�����)

��ֵ�ȶ��ԣ�
- ʹ��˫���ȸ����ۼӱ��⾫����ʧ
- �ر��Ż�����ֵ�������Զ�����sum_sq���������ϵͳ����ʵ�֣�

�����ԣ�
- ��ȷ����������洢���������;��󣨺��ǶԳ�/�Ƿ���
*/

/* �����ӡ����Ļ�ĺ��� */
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
/*�����ӡ����Ļ1.0*/
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
/*����F-���������������*/
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
		return err; // �����������ȴ���
	}
	err = matrix_frobenius_norm(&inv, &norm_inv);
	matrix_free(&inv);
	if (err != MATRIX_SUCCESS) return err;

	*cond = norm_A * norm_inv;
	return MATRIX_SUCCESS;
}





