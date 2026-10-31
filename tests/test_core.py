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
