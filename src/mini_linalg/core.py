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

