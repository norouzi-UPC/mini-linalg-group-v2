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


def test_matrix_mul_vector():
    matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
    vector = Vector([5.0, 7.0])
    result = matrix_mul_vector(matrix, vector)
    assert result.data == [19.0, 43.0]


def test_matrix_mul_matrix():
    left = Matrix([[1.0, 2.0], [3.0, 4.0]])
    right = Matrix([[2.0, 0.0], [1.0, 1.0]])
    result = matrix_mul_matrix(left, right)
    assert result.data == [[4.0, 2.0], [10.0, 4.0]]


def test_matrix_mul_vector_rejects_dimension_mismatch():
    matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
    vector = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        matrix_mul_vector(matrix, vector)


def test_matrix_mul_matrix_rejects_dimension_mismatch():
    left = Matrix([[1.0, 2.0], [3.0, 4.0]])
    right = Matrix([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError):
        matrix_mul_matrix(left, right)
