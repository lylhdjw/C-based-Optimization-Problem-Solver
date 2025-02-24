#include <stdio.h>
#include <stdlib.h>
#include "QPSparseMatrix.h"
#include "QPMatrix.h"
/*基础链表功能*/
// 创建稀疏矩阵
SparseMatrix* sparse_create(int rows, int cols) {
	SparseMatrix* matrix = (SparseMatrix*)malloc(sizeof(SparseMatrix));
	if (!matrix) return NULL;

	matrix->rows = rows;
	matrix->cols = cols;
	matrix->node_count = 0;
	matrix->head = NULL;
	return matrix;
}

// 释放稀疏矩阵
void sparse_free(SparseMatrix* matrix) {
	SparseNode* current = matrix->head;
	while (current != NULL) {
		SparseNode* temp = current;
		current = current->next;
		free(temp);
	}
	free(matrix);
}

// 插入新节点（自动处理重复位置）
MatrixError sparse_insert(SparseMatrix* matrix, int row, int col, double value) 
{
	// 查找是否已存在该位置
	SparseNode* prev = NULL;
	SparseNode* current = matrix->head;
	if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->cols)
		return MATRIX_ERROR_DIM;
	while (current != NULL) {
		if (current->row == row && current->col == col) {
			// 更新已有节点
			current->value = value;
			return MATRIX_SUCCESS;
		}
		prev = current;
		current = current->next;
	}
	// 创建新节点
	SparseNode* new_node = (SparseNode*)malloc(sizeof(SparseNode));
	if (!new_node) return MATRIX_ERROR_MEMORY;

	new_node->id = matrix->node_count++;
	new_node->row = row;
	new_node->col = col;
	new_node->value = value;
	new_node->next = NULL;

	// 插入链表
	if (prev == NULL) {
		matrix->head = new_node;
	}
	else {
		prev->next = new_node;
	}
	return MATRIX_SUCCESS;
}

// 删除指定位置节点
MatrixError sparse_remove(SparseMatrix* matrix, int row, int col) {
	SparseNode* prev = NULL;
	SparseNode* current = matrix->head;

	while (current != NULL) {
		if (current->row == row && current->col == col) {
			if (prev == NULL) {
				matrix->head = current->next;
			}
			else {
				prev->next = current->next;
			}
			free(current);
			matrix->node_count--;
			return MATRIX_SUCCESS;
		}
		prev = current;
		current = current->next;
	}
	return MATRIX_ERROR_NUMERICAL; // 未找到元素
}
/*------------矩阵转换功能--------------*/

// 稀疏矩阵转常规矩阵
MatrixError sparse_to_dense(const SparseMatrix* sparse, Matrix* dense) {
	if (dense->rows != sparse->rows || dense->cols != sparse->cols)
		return MATRIX_ERROR_DIM;

	matrix_set_zero(dense);

	SparseNode* current = sparse->head;
	while (current != NULL) {
		matrix_set(dense, current->row, current->col, current->value);
		current = current->next;
	}
	return MATRIX_SUCCESS;
}

// 常规矩阵转稀疏矩阵
MatrixError dense_to_sparse(const Matrix* dense, SparseMatrix* sparse) {
	sparse->rows = dense->rows;
	sparse->cols = dense->cols;

	for (int i = 0; i < dense->rows; ++i) {
		for (int j = 0; j < dense->cols; ++j) {
			double val = matrix_get(dense, i, j);
			if (val != 0.0) {
				MatrixError err = sparse_insert(sparse, i, j, val);
				if (err != MATRIX_SUCCESS) return err;
			}
		}
	}
	return MATRIX_SUCCESS;
}
/*------------------矩阵运算功能实现-----------------------*/
// 稀疏矩阵加法
MatrixError sparse_add(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {
	

	// 复制A到C
	SparseNode* current;
	if (A->rows != B->rows || A->cols != B->cols ||
		C->rows != A->rows || C->cols != A->cols)
		return MATRIX_ERROR_DIM;
	current = A->head;
	while (current != NULL) {
		sparse_insert(C, current->row, current->col, current->value);
		current = current->next;
	}

	// 添加B的元素
	current = B->head;
	while (current != NULL) {
		double existing_val;
		if (sparse_get(C, current->row, current->col, &existing_val) == MATRIX_SUCCESS) {
			sparse_insert(C, current->row, current->col, existing_val + current->value);
		}
		else {
			sparse_insert(C, current->row, current->col, current->value);
		}
		current = current->next;
	}
	return MATRIX_SUCCESS;
}

// 稀疏矩阵乘法
MatrixError sparse_multiply(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {
	

	// 创建临时存储行和列的数组
	SparseNode* row_nodes = (SparseNode*)malloc(sizeof(SparseNode));
	SparseNode* col_nodes = (SparseNode*)malloc(sizeof(SparseNode));

	if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols)
		return MATRIX_ERROR_DIM;

	// 矩阵乘法核心算法
	for (int i = 0; i < A->rows; ++i) {
		for (int j = 0; j < B->cols; ++j) {
			double sum = 0.0;

			// 遍历A的第i行和B的第j列
			SparseNode* a = A->head;
			while (a != NULL) {
				if (a->row == i) {
					SparseNode* b = B->head;
					while (b != NULL) {
						if (b->col == j && a->col == b->row) {
							sum += a->value * b->value;
						}
						b = b->next;
					}
				}
				a = a->next;
			}

			if (sum != 0.0) {
				sparse_insert(C, i, j, sum);
			}
		}
	}
	return MATRIX_SUCCESS;
}

// 数乘运算
MatrixError sparse_scalar_multiply(SparseMatrix* matrix, double scalar) {
	SparseNode* current = matrix->head;
	while (current != NULL) {
		current->value *= scalar;
		if (current->value == 0.0) { // 自动移除零元素
			SparseNode* temp = current;
			current = current->next;
			sparse_remove(matrix, temp->row, temp->col);
		}
		else {
			current = current->next;
		}
	}
	return MATRIX_SUCCESS;
}

// 获取指定位置的元素值
MatrixError sparse_get(const SparseMatrix* matrix, int row, int col, double* value) {
	

	SparseNode* current;
	if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->cols)
		return MATRIX_ERROR_DIM;

	current = matrix->head;
	while (current != NULL) {
		if (current->row == row && current->col == col) {
			*value = current->value;
			return MATRIX_SUCCESS;
		}
		current = current->next;
	}
	*value = 0.0;  // 未找到时返回 0（稀疏矩阵默认值）
	return MATRIX_SUCCESS;
}
















