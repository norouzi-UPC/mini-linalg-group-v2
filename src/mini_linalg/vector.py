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
