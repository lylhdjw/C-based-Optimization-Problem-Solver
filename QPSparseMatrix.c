#include <stdio.h>
#include <stdlib.h>
#include "QPSparseMatrix.h"
#include "QPMatrix.h"
/*����������*/
// ����ϡ�����
SparseMatrix* sparse_create(int rows, int cols) {
	SparseMatrix* matrix = (SparseMatrix*)malloc(sizeof(SparseMatrix));
	if (!matrix) return NULL;

	matrix->rows = rows;
	matrix->cols = cols;
	matrix->node_count = 0;
	matrix->head = NULL;
	return matrix;
}

// �ͷ�ϡ�����
void sparse_free(SparseMatrix* matrix) {
	SparseNode* current = matrix->head;
	while (current != NULL) {
		SparseNode* temp = current;
		current = current->next;
		free(temp);
	}
	free(matrix);
}

// �����½ڵ㣨�Զ������ظ�λ�ã�
MatrixError sparse_insert(SparseMatrix* matrix, int row, int col, double value) 
{
	// �����Ƿ��Ѵ��ڸ�λ��
	SparseNode* prev = NULL;
	SparseNode* current = matrix->head;
	if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->cols)
		return MATRIX_ERROR_DIM;
	while (current != NULL) {
		if (current->row == row && current->col == col) {
			// �������нڵ�
			current->value = value;
			return MATRIX_SUCCESS;
		}
		prev = current;
		current = current->next;
	}
	// �����½ڵ�
	SparseNode* new_node = (SparseNode*)malloc(sizeof(SparseNode));
	if (!new_node) return MATRIX_ERROR_MEMORY;

	new_node->id = matrix->node_count++;
	new_node->row = row;
	new_node->col = col;
	new_node->value = value;
	new_node->next = NULL;

	// ��������
	if (prev == NULL) {
		matrix->head = new_node;
	}
	else {
		prev->next = new_node;
	}
	return MATRIX_SUCCESS;
}

// ɾ��ָ��λ�ýڵ�
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
	return MATRIX_ERROR_NUMERICAL; // δ�ҵ�Ԫ��
}
/*------------����ת������--------------*/

// ϡ�����ת�������
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

// �������תϡ�����
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
/*------------------�������㹦��ʵ��-----------------------*/
// ϡ�����ӷ�
MatrixError sparse_add(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {
	

	// ����A��C
	SparseNode* current;
	if (A->rows != B->rows || A->cols != B->cols ||
		C->rows != A->rows || C->cols != A->cols)
		return MATRIX_ERROR_DIM;
	current = A->head;
	while (current != NULL) {
		sparse_insert(C, current->row, current->col, current->value);
		current = current->next;
	}

	// ���B��Ԫ��
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

// ϡ�����˷�
MatrixError sparse_multiply(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {
	

	// ������ʱ�洢�к��е�����
	SparseNode* row_nodes = (SparseNode*)malloc(sizeof(SparseNode));
	SparseNode* col_nodes = (SparseNode*)malloc(sizeof(SparseNode));

	if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols)
		return MATRIX_ERROR_DIM;

	// ����˷������㷨
	for (int i = 0; i < A->rows; ++i) {
		for (int j = 0; j < B->cols; ++j) {
			double sum = 0.0;

			// ����A�ĵ�i�к�B�ĵ�j��
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

// ��������
MatrixError sparse_scalar_multiply(SparseMatrix* matrix, double scalar) {
	SparseNode* current = matrix->head;
	while (current != NULL) {
		current->value *= scalar;
		if (current->value == 0.0) { // �Զ��Ƴ���Ԫ��
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

// ��ȡָ��λ�õ�Ԫ��ֵ
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
	*value = 0.0;  // δ�ҵ�ʱ���� 0��ϡ�����Ĭ��ֵ��
	return MATRIX_SUCCESS;
}
















