import numpy as np

class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = [list(row) for row in data]

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def __getitem__(self, key):
        row, col = key
        return self.data[row, col]


def matrix_mul_vector(matrix, vector):
    """Return the matrix-vector product as a new Vector."""
    if len(matrix.data[0]) != len(vector):
        raise ValueError("Matrix columns must match vector length.")
    result = np.array(matrix.data, dtype=float) @ np.array(vector.data, dtype=float)
    return Vector(result.tolist())


def matrix_mul_matrix(matrixA, matrixB):
    """Return the matrix-matrix product as a new Matrix."""
    if len(matrixA.data[0]) != len(matrixB.data):
        raise ValueError("Left matrix columns must match right matrix rows.")
    result = np.array(matrixA.data, dtype=float) @ np.array(matrixB.data, dtype=float)
    return Matrix(result.tolist())
