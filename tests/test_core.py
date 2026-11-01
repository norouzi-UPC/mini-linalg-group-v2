"""Core contract tests extended for the function based products."""

import pytest

from mini_linalg import Matrix, Vector
from mini_linalg.core import matrix_mul_matrix, matrix_mul_vector


def test_vector_copies_input():
    data = [1.0, 2.0]
    vector = Vector(data)
    data[0] = 99.0
    assert vector.data[0] == 1.0


def test_matrix_copies_input_rows():
    data = [[1.0, 2.0], [3.0, 4.0]]
    matrix = Matrix(data)
    data[0][0] = 99.0
    assert matrix.data[0][0] == 1.0


def test_vector_repr():
    assert repr(Vector([1.0, 2.0])) == "Vector([1.0, 2.0])"


def test_matrix_repr():
    assert repr(Matrix([[1.0, 2.0]])) == "Matrix([[1.0, 2.0]])"

def test_matrix_vector_product():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    result = matrix.mat_vec(Vector([1, 0, -1]))
    assert list(result.data) == [-2.0, -2.0]


def test_matrix_matrix_product():
    left = Matrix([[1, 2, 3], [4, 5, 6]])
    right = Matrix([[7, 8], [9, 10], [11, 12]])
    result = left.mat_mat(right)
    assert result[0, 0] == 58.0
    assert result[0, 1] == 64.0
    assert result[1, 0] == 139.0
    assert result[1, 1] == 154.0


def test_vector_index_and_length():
    vector = Vector([1, 2, 3])
    assert len(vector) == 3
    assert vector[0] == 1
    assert vector[-1] == 3


def test_matrix_dimensions_and_index():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix.rows == 2
    assert matrix.cols == 3
    assert matrix[1, 2] == 6