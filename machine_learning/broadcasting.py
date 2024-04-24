import numpy as np


# Broadcasting in NumPy allows for arithmetic operations on arrays with different shapes.
# When operating on two arrays, NumPy automatically "broadcasts" the smaller array across
# the larger array. This is done by aligning the shapes of the arrays:
# 1. It starts by comparing their dimensions from the trailing to the leading dimensions.
# 2. If one array has fewer dimensions, it is virtually padded with dimensions of size one at the beginning.
# 3. In each dimension, if the arrays differ in size and one of them is 1, the smaller dimension is
#    stretched to match the larger one. This virtual replication happens without copying data.
# Broadcasting makes operations memory-efficient and fast as it avoids explicit replication of data.
# It allows for cleaner code by avoiding manual loops over array elements for matching shapes.



# Generate a 4D array with random values, shape (64, 3, 32, 10)
# This might represent 64 samples, each with 3 channels, each channel being a 32x10 matrix.
x = np.random.random((64, 3, 32, 10))

# Generate a 2D array with random values, shape (32, 10)
# This might be a parameter or mask that applies to all samples and channels.
y = np.random.random((32, 10))

# Compute the element-wise maximum between x and y using broadcasting.
# y (32, 10) is treated as if it were (1, 1, 32, 10) to align with the dimensions of x.
z = np.maximum(x, y)

# Print the shape of the result to confirm it matches the shape of x.
print(z.shape)  # Output will be (64, 3, 32, 10), confirming the broadcasting worked.


def naive_add_matrix_and_vector(x, y):
    # Ensure x is a 2D matrix and y is a 1D vector
    assert len(x.shape) == 2, "x must be a 2D matrix"
    assert len(y.shape) == 1, "y must be a 1D vector"
    # Ensure the number of columns in x matches the length of y
    assert x.shape[1] == y.shape[0], "x's columns must match the length of y"

    # Copy x to avoid modifying the original matrix
    x = x.copy()

    # Add vector y to each row of matrix x
    # Iterate over all rows
    for i in range(x.shape[0]):
        # Iterate over all columns
        for j in range(x.shape[1]):
            # Add element j of vector y to element (i, j) of matrix x
            x[i, j] += y[j]

    # Return the modified matrix
    print(x)
    return x

naive_add_matrix_and_vector(np.array([[1, 2, 3], [4, 5, 6]]), np.array([10, 20, 30]))

 