#ifndef __SPARSE_MATRIX__
#define __SPARSE_MATRIX__

#include "QPMatrix.h"

// ϡ�����ڵ㶨��
typedef struct SparseNode {
	int id;             // Ψһ���
	int row;            // �к�
	int col;            // �к�
	double value;       // Ԫ��ֵ
	struct SparseNode* next; // ��һ���ڵ�ָ��
} SparseNode;

// ϡ�����ṹ��
typedef struct {
	int rows;           // ��������
	int cols;           // ��������
	int node_count;     // ����Ԫ�ؼ���
	SparseNode* head;   // ����ͷָ��
} SparseMatrix;

// ��������
SparseMatrix* sparse_create(int rows, int cols);
void sparse_free(SparseMatrix* matrix);
MatrixError sparse_insert(SparseMatrix* matrix, int row, int col, double value);
MatrixError sparse_remove(SparseMatrix* matrix, int row, int col);
MatrixError sparse_get(const SparseMatrix* matrix, int row, int col, double* value);

MatrixError sparse_to_dense(const SparseMatrix* sparse, Matrix* dense);
MatrixError dense_to_sparse(const Matrix* dense, SparseMatrix* sparse);

MatrixError sparse_add(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C);
MatrixError sparse_multiply(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C);
MatrixError sparse_scalar_multiply(SparseMatrix* matrix, double scalar);

#endif
