class Vector:
    def __init__(self, data):
        self.data = list(data)
        if len(self.data) == 0:
            raise ValueError("A vector must contain at least one number.")

    def __repr__(self):
        return f"Vector({self.data})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        if len(self.data) == 0 or len(self.data[0]) == 0:
            raise ValueError("A matrix must have at least one row and column.")

        self.rows = len(self.data)
        self.cols = len(self.data[0])
        for row in self.data:
            if len(row) != self.cols:
                raise ValueError("All matrix rows must have the same length.")

    def __repr__(self):
        return f"Matrix({self.data})"

    def __getitem__(self, key):
        row, col = key
        return self.data[row][col]

    def mat_vec(self, vector):
        if self.cols != len(vector):
            raise ValueError("Matrix columns must match vector length.")

        result = []
        for i in range(self.rows):
            total = 0.0
            for j in range(self.cols):
                total += self.data[i][j] * vector[j]
            result.append(total)
        return Vector(result)

    def mat_mat(self, other):
        if self.cols != other.rows:
            raise ValueError("Left matrix columns must match right matrix rows.")

        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                total = 0.0
                for k in range(self.cols):
                    total += self.data[i][k] * other.data[k][j]
                row.append(total)
            result.append(row)
        return Matrix(result)
