"""Core vector and matrix types backed by NumPy arrays.

Method names (mat_vec, mat_mat) and error messages match Aland's code.
"""

import numpy as np


class Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        if self.data.size == 0:
            raise ValueError("A vector must contain at least one number.")

    def __repr__(self):
        return f"Vector({self.data.tolist()})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        if self.data.ndim != 2 or self.data.size == 0:
            raise ValueError("A matrix must have at least one row and column.")
        if len({row.size for row in self.data}) != 1:
            raise ValueError("All matrix rows must have the same length.")

        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

    def __repr__(self):
        return f"Matrix({self.data.tolist()})"

    def __getitem__(self, key):
        # Accepts matrix[0, 0], matrix[0][0], and slices alike.
        return self.data[key]

    def mat_vec(self, vector):
        if self.cols != len(vector):
            raise ValueError("Matrix columns must match vector length.")
        return Vector(self.data @ vector.data)

    def mat_mat(self, other):
        if self.cols != other.rows:
            raise ValueError("Left matrix columns must match right matrix rows.")
        return Matrix(self.data @ other.data)


def batch_mat_vec(matrices, vectors):
    """Multiply matrices and vectors in a non-empty batch."""
    if len(matrices) == 0 or len(matrices) != len(vectors):
        raise ValueError("Use one vector per matrix in a non-empty batch.")
    matrix_data = []
    vector_data = []
    for i in range(len(matrices)):
        matrix_data.append(matrices[i].data)
        vector_data.append(vectors[i].data)
    matrix_array = np.stack(matrix_data)
    vector_array = np.stack(vector_data)
    if matrix_array.shape[2] != vector_array.shape[1]:
        raise ValueError("Matrix columns must match vector length.")

    # Each vector becomes a column, so matmul pairs the matching batch items.
    columns = vector_array.reshape(len(vectors), vector_array.shape[1], 1)
    products = np.matmul(matrix_array, columns)
    result = []
    for product in products:
        result.append(Vector(product[:, 0]))
    return result


def batch_mat_mat(left_matrices, right_matrices):
    """Multiply matrices in two non-empty batches."""
    if len(left_matrices) == 0 or len(left_matrices) != len(right_matrices):
        raise ValueError("Both non-empty matrix batches must have equal length.")

    left_data = []
    right_data = []
    for i in range(len(left_matrices)):
        left_data.append(left_matrices[i].data)
        right_data.append(right_matrices[i].data)

    left_array = np.stack(left_data)
    right_array = np.stack(right_data)
    if left_array.shape[2] != right_array.shape[1]:
        raise ValueError("Left matrix columns must match right matrix rows.")

    products = np.matmul(left_array, right_array)
    result = []
    for product in products:
        result.append(Matrix(product))
    return result
