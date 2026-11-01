import numpy as np

from .matrix import Matrix
from .vector import Vector


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
