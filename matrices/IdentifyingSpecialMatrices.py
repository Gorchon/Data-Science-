# GRADED FUNCTION
import numpy as np


# Our function will go through the matrix replacing each row in order turning it into echelon form.
# If at any point it fails because it can't put a 1 in the leading diagonal,
# we will return the value True, otherwise, we will return False.
# There is no need to edit this function.
def isSingular(A):
    B = np.array(A, dtype=np.float_)  # Make B as a copy of A, since we're going to alter it's values.
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False


# This next line defines our error flag. For when things go wrong if the matrix is singular.
# There is no need to edit this line.
class MatrixIsSingular(Exception): pass


# For Row Zero, all we require is the first element is equal to 1.
# We'll divide the row by the value of A[0, 0].
# This will get us in trouble though if A[0, 0] equals 0, so first we'll test for that,
# and if this is true, we'll add one of the lower rows to the first one before the division.
# We'll repeat the test going down each lower row until we can do the division.
# There is no need to edit this function.
def fixRowZero(A):
    if A[0, 0] == 0:
        A[0] = A[0] + A[1]
    if A[0, 0] == 0:
        A[0] = A[0] + A[2]
    if A[0, 0] == 0:
        A[0] = A[0] + A[3]
    if A[0, 0] == 0:
        raise MatrixIsSingular()
    A[0] = A[0] / A[0, 0]
    return A


# First we'll set the sub-diagonal elements to zero, i.e. A[1,0].
# Next we want the diagonal element to be equal to one.
# We'll divide the row by the value of A[1, 1].
# Again, we need to test if this is zero.
# If so, we'll add a lower row and repeat setting the sub-diagonal elements to zero.
# There is no need to edit this function.
def fixRowOne(A):
    A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        raise MatrixIsSingular()
    A[1] = A[1] / A[1, 1]
    return A


# This is the first function that you should complete.
# Follow the instructions inside the function at each comment.
def fixRowTwo(A):
    # Eliminate the sub-diagonal elements of row two (A[2,0] and A[2,1]).
    A[2] = A[2] - A[2, 0] * A[0]
    A[2] = A[2] - A[2, 1] * A[1]

    # Check if the diagonal element is zero.
    if A[2, 2] == 0:
        # Add a lower row (A[3]) to row 2.
        A[2] = A[2] + A[3]
        # Attempt to set the sub-diagonal elements to zero again.
        A[2] = A[2] - A[2, 0] * A[0]
        A[2] = A[2] - A[2, 1] * A[1]

    if A[2, 2] == 0:
        raise MatrixIsSingular()
    # Set the diagonal element to one.
    A[2] = A[2] / A[2, 2]
    return A


# You should also complete this function
# Follow the instructions inside the function at each comment.
def fixRowThree(A):
    # Eliminate the sub-diagonal elements of row three.
    A[3] = A[3] - A[3, 0] * A[0]
    A[3] = A[3] - A[3, 1] * A[1]
    A[3] = A[3] - A[3, 2] * A[2]

    # Check if the diagonal element is zero.
    if A[3, 3] == 0:
        raise MatrixIsSingular()
    # Set the diagonal element to one.
    A[3] = A[3] / A[3, 3]
    return A


# Test cases
test_matrices = {
    'non_singular_matrix': [
        [4, 2, 2, 1],
        [3, 6, 2, 1],
        [2, 1, 3, 1],
        [1, 1, 1, 2]
    ],
    'singular_matrix': [
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    'singular_matrix_column_zeros': [
        [1, 2, 3, 0],
        [5, 6, 7, 0],
        [9, 10, 11, 0],
        [13, 14, 15, 0]
    ],
    'non_singular_matrix_needs_row_swaps': [
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ]
}

# Testing the matrices
for name, matrix in test_matrices.items():
    is_singular = isSingular(matrix)
    print(f"{name}: {is_singular}")