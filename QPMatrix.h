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



// �����붨��
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

// ����ṹ��
typedef struct {
	int rows;
	int cols;
	double* data;
} Matrix;

double myabs(x);//�Զ������ֵ����
int compare_double(const void* a, const void* b);/*��������-����ʵ��ab��С�Ƚϣ�a����b����1�����򷵻�-1*/
bool matrix_is_symmetric(const Matrix* A);/*�ж�����A�Ƿ��ǶԳ���ĺ������Ƿ���true�����򷵻�False*/
MatrixError matrix_create(Matrix* mat, int rows, int cols);/*�����ʼ������*/
void matrix_free(Matrix* mat);/*�����ͷź���*/
MatrixError matrix_copy(Matrix* dest, const Matrix* src);/*����ֵ������destΪĿ�꣬srcΪԴ*/
double matrix_get(const Matrix* mat, int row, int col);/*��ȡrow��col�����Ԫ�ص�ֵ*/
void matrix_set(Matrix* mat, int row, int col, double val);/*�Ծ���row��col��Ԫ�ظ�valֵ*/
MatrixError matrix_add(Matrix* C, const Matrix* A, const Matrix* B);/* ��������㷨������ΪC */
MatrixError matrix_multiply(Matrix* C, const Matrix* A, const Matrix* B);/* ��������㷨������ΪC = A*B */
MatrixError matrix_lu_decompose(const Matrix* A, Matrix* LU, int *pivot_sign, int* pivots);/* ����A��LU�ֽ��㷨��pivot_sign��pivotΪ�����н��й����б任�ķ��ż�¼�;����¼ */
MatrixError matrix_identity(Matrix* mat);/*������λ����*/
MatrixError matrix_row_swap(Matrix* mat, int row1, int row2);/*�����л���*/
MatrixError matrix_scalar_multiply(Matrix* C, const Matrix* A, double scalar);/*����C = scalar*A����������*/
MatrixError matrix_determinant(const Matrix* A, double* det);/* ����LU�ֽ�ľ�������ʽ����,����det */
MatrixError matrix_solve_lu(const Matrix* A, const Matrix* B, Matrix* X);/*����LU�ֽ�����Է���������㷨����������ʽΪAX=B*/
MatrixError matrix_inverse(Matrix* inv, const Matrix* A);/* �������Է��������ľ������溯�������������inv�� */
MatrixError matrix_transpose(Matrix* dest, const Matrix* src);/*����ת�ú���*/ 
MatrixError matrix_augment(Matrix* dest, const Matrix* A, const Matrix* B, char direction);/* �������㺯�� */
MatrixError matrix_qr_decompose(const Matrix* A, Matrix* Q, Matrix* R);/* QR�ֽ��㷨 */
MatrixError matrix_symmetric_diagonalize(const Matrix* A, Matrix* Q, Matrix* D, int max_iterations, double tolerance);
/*����QR�ֽ�ĶԳƾ������ƶԽǻ�����*/
MatrixError matrix_set_zero(Matrix* mat);/*����ȫ��0����*/
MatrixError matrix_frobenius_norm(const Matrix* mat, double* norm);/*�����F-�������㷽��*/
MatrixError matrix_condition_number(const Matrix* A, double* cond);/*����F-�������������������*/
MatrixError matrix_print_only(Matrix* A);/*�����ӡ����Ļ*/
void matrix_print(const char* name, const Matrix* mat);/*�����ӡ����Ļ1.0*/

#endif