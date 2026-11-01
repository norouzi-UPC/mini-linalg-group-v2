import numpy as np
import pytest

from mini_linalg import Matrix, Vector


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


def test_vectors_have_independent_data():
    first = Vector([1.0, 2.0])
    second = Vector([1.0, 2.0])
    first.data[0] = 8.0
    assert second.data[0] == 1.0


def test_matrix_rows_have_independent_data():
    row = [1.0, 2.0]
    matrix = Matrix([row, row])
    matrix.data[0][0] = 8.0
    assert matrix.data[1][0] == 1.0


def test_vector_keeps_input_order():
    vector = Vector([3.0, -1.0, 0.5])
    assert vector.data[0] == 3.0
    assert vector.data[1] == -1.0
    assert vector.data[2] == 0.5


def test_rectangular_matrix_keeps_entries():
    matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert len(matrix.data) == 2
    assert len(matrix.data[0]) == 3
    assert matrix.data[1][2] == 6.0


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


def test_empty_vector_is_rejected():
    with pytest.raises(ValueError):
        Vector([])


def test_empty_matrix_is_rejected():
    with pytest.raises(ValueError):
        Matrix([])


def test_empty_matrix_row_is_rejected():
    with pytest.raises(ValueError):
        Matrix([[]])


def test_ragged_matrix_is_rejected():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])


def test_matrix_vector_dimension_mismatch():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        matrix.mat_vec(Vector([1, 2]))


def test_matrix_matrix_dimension_mismatch():
    left = Matrix([[1, 2, 3]])
    right = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        left.mat_mat(right)


def test_identity_products():
    identity = Matrix([[1, 0], [0, 1]])
    matrix = Matrix([[2, 3], [4, 5]])
    np.testing.assert_allclose(identity.mat_mat(matrix).data, matrix.data)
    np.testing.assert_allclose(matrix.mat_mat(identity).data, matrix.data)
    np.testing.assert_allclose(identity.mat_vec(Vector([6, 7])).data, [6, 7])


def test_zero_products():
    zero = Matrix([[0, 0], [0, 0]])
    matrix = Matrix([[1, 2], [3, 4]])
    np.testing.assert_allclose(zero.mat_vec(Vector([5, 6])).data, [0, 0])
    np.testing.assert_allclose(matrix.mat_mat(zero).data, [[0, 0], [0, 0]])


def test_negative_and_fractional_entries():
    matrix = Matrix([[-1.5, 2], [0, 3.25]])
    result = matrix.mat_vec(Vector([2, -2]))
    np.testing.assert_allclose(result.data, [-7, -6.5])


def test_single_entry_products():
    matrix = Matrix([[2.5]])
    np.testing.assert_allclose(matrix.mat_vec(Vector([-4])).data, [-10])
    np.testing.assert_allclose(matrix.mat_mat(Matrix([[4]])).data, [[10]])


def test_single_column_products():
    matrix = Matrix([[1], [2], [3]])
    np.testing.assert_allclose(matrix.mat_vec(Vector([4])).data, [4, 8, 12])
    result = matrix.mat_mat(Matrix([[5, 6]]))
    np.testing.assert_allclose(result.data, [[5, 6], [10, 12], [15, 18]])


def test_vector_index_out_of_range():
    with pytest.raises(IndexError):
        Vector([1, 2])[2]


def test_matrix_index_out_of_range():
    with pytest.raises(IndexError):
        Matrix([[1, 2]])[1, 0]


def test_products_leave_inputs_unchanged():
    matrix = Matrix([[1, 2], [3, 4]])
    other = Matrix([[5, 6], [7, 8]])
    vector = Vector([9, 10])
    matrix.mat_vec(vector)
    matrix.mat_mat(other)
    np.testing.assert_allclose(matrix.data, [[1, 2], [3, 4]])
    np.testing.assert_allclose(other.data, [[5, 6], [7, 8]])
    np.testing.assert_allclose(vector.data, [9, 10])


def test_products_return_new_objects():
    matrix = Matrix([[1, 0], [0, 1]])
    vector = Vector([2, 3])
    vector_result = matrix.mat_vec(vector)
    matrix_result = matrix.mat_mat(matrix)
    assert isinstance(vector_result, Vector)
    assert isinstance(matrix_result, Matrix)
    vector_result.data[0] = 99
    matrix_result.data[0][0] = 99
    assert vector.data[0] == 2
    assert matrix.data[0][0] == 1


def test_rectangular_products_match_numpy():
    generator = np.random.default_rng(12)
    left = generator.normal(size=(3, 4))
    right = generator.normal(size=(4, 2))
    vector = generator.normal(size=4)
    matrix = Matrix(left)
    np.testing.assert_allclose(matrix.mat_vec(Vector(vector)).data, left @ vector)
    np.testing.assert_allclose(matrix.mat_mat(Matrix(right)).data, left @ right)
