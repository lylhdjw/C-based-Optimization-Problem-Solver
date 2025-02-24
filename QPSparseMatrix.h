#ifndef __SPARSE_MATRIX__
#define __SPARSE_MATRIX__

#include "QPMatrix.h"

// 稀疏矩阵节点定义
typedef struct SparseNode {
	int id;             // 唯一编号
	int row;            // 行号
	int col;            // 列号
	double value;       // 元素值
	struct SparseNode* next; // 下一个节点指针
} SparseNode;

// 稀疏矩阵结构体
typedef struct {
	int rows;           // 矩阵行数
	int cols;           // 矩阵列数
	int node_count;     // 非零元素计数
	SparseNode* head;   // 链表头指针
} SparseMatrix;

// 函数声明
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
