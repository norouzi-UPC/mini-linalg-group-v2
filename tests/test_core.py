import numpy as np
import pytest

from mini_linalg import Matrix, Vector, batch_mat_mat, batch_mat_vec


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


def test_batch_matrix_vector_products():
    matrices = [Matrix([[1, 2], [3, 4]]), Matrix([[2, 0], [0, 3]])]
    vectors = [Vector([1, 1]), Vector([4, 5])]
    result = batch_mat_vec(matrices, vectors)
    assert len(result) == 2
    assert isinstance(result[0], Vector)
    np.testing.assert_allclose(result[0].data, [3, 7])
    np.testing.assert_allclose(result[1].data, [8, 15])


def test_batch_matrix_matrix_products():
    left = [Matrix([[1, 2], [3, 4]]), Matrix([[2, 0], [0, 3]])]
    right = [Matrix([[1, 0], [0, 1]]), Matrix([[4, 1], [2, 5]])]
    result = batch_mat_mat(left, right)
    assert len(result) == 2
    assert isinstance(result[0], Matrix)
    np.testing.assert_allclose(result[0].data, [[1, 2], [3, 4]])
    np.testing.assert_allclose(result[1].data, [[8, 2], [6, 15]])


def test_batch_rectangular_products():
    matrices = [Matrix([[1, 2, 3], [4, 5, 6]])]
    vectors = [Vector([1, 0, -1])]
    np.testing.assert_allclose(batch_mat_vec(matrices, vectors)[0].data, [-2, -2])
    right = [Matrix([[7, 8], [9, 10], [11, 12]])]
    result = batch_mat_mat(matrices, right)
    np.testing.assert_allclose(result[0].data, [[58, 64], [139, 154]])


def test_batch_empty_vector_inputs_are_rejected():
    with pytest.raises(ValueError):
        batch_mat_vec([], [])


def test_batch_empty_matrix_inputs_are_rejected():
    with pytest.raises(ValueError):
        batch_mat_mat([], [])


def test_batch_vector_count_mismatch():
    matrices = [Matrix([[1]])]
    with pytest.raises(ValueError):
        batch_mat_vec(matrices, [Vector([1]), Vector([2])])


def test_batch_matrix_count_mismatch():
    left = [Matrix([[1]])]
    with pytest.raises(ValueError):
        batch_mat_mat(left, [Matrix([[1]]), Matrix([[2]])])


def test_batch_vector_dimension_mismatch():
    with pytest.raises(ValueError):
        batch_mat_vec([Matrix([[1, 2]])], [Vector([1])])


def test_batch_matrix_dimension_mismatch():
    with pytest.raises(ValueError):
        batch_mat_mat([Matrix([[1, 2]])], [Matrix([[1, 2]])])


def test_batch_mixed_matrix_shapes_are_rejected():
    matrices = [Matrix([[1]]), Matrix([[1, 2]])]
    with pytest.raises(ValueError):
        batch_mat_vec(matrices, [Vector([1]), Vector([2])])


def test_batch_mixed_vector_shapes_are_rejected():
    matrices = [Matrix([[1, 2]]), Matrix([[3, 4]])]
    with pytest.raises(ValueError):
        batch_mat_vec(matrices, [Vector([1]), Vector([2, 3])])


def test_batch_mixed_right_matrix_shapes_are_rejected():
    left = [Matrix([[1]]), Matrix([[2]])]
    right = [Matrix([[1]]), Matrix([[1, 2]])]
    with pytest.raises(ValueError):
        batch_mat_mat(left, right)


def test_batch_leaves_inputs_unchanged():
    matrix = Matrix([[2, 3], [4, 5]])
    vector = Vector([6, 7])
    batch_mat_vec([matrix], [vector])
    batch_mat_mat([matrix], [matrix])
    np.testing.assert_allclose(matrix.data, [[2, 3], [4, 5]])
    np.testing.assert_allclose(vector.data, [6, 7])


def test_batch_results_are_independent():
    matrix = Matrix([[2]])
    vector = Vector([3])
    vectors = batch_mat_vec([matrix, matrix], [vector, vector])
    matrices = batch_mat_mat([matrix, matrix], [matrix, matrix])
    vectors[0].data[0] = 99
    matrices[0].data[0][0] = 99
    assert vectors[1].data[0] == 6
    assert matrices[1].data[0][0] == 4
    assert matrix.data[0][0] == 2
    assert vector.data[0] == 3


def test_three_item_batch_matches_numpy():
    generator = np.random.default_rng(23)
    left_data = generator.normal(size=(3, 2, 4))
    right_data = generator.normal(size=(3, 4, 3))
    vector_data = generator.normal(size=(3, 4))
    left = [Matrix(item) for item in left_data]
    right = [Matrix(item) for item in right_data]
    vectors = [Vector(item) for item in vector_data]
    vector_results = batch_mat_vec(left, vectors)
    matrix_results = batch_mat_mat(left, right)
    for i in range(3):
        np.testing.assert_allclose(vector_results[i].data,
                                   left_data[i] @ vector_data[i])
        np.testing.assert_allclose(matrix_results[i].data,
                                   left_data[i] @ right_data[i])
