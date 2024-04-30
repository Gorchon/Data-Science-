# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Our first function will perform the Gram-Schmidt procedure for 4 basis vectors.
# We'll take this list of vectors as the columns of a matrix, A.
# We'll then go through the vectors one at a time and set them to be orthogonal
# to all the vectors that came before it. Before normalising.
# Follow the instructions inside the function at each comment.
# You will be told where to add code to complete the function.
def gsBasis4(A):
    B = np.array(A, dtype=np.float_)  # Copy input matrix A to B to avoid modifying A directly.
    
    # Normalize the zeroth column (first basis vector)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])  # Divide by its norm to make it a unit vector.

    # Process the first column
    # Remove projection of the first column onto the normalized zeroth column
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # Normalize the first column if it is not near zero
    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])  # Set to zero if it's dependent on the zeroth column

    # Process the second column
    # Remove projection onto the zeroth and first column
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]
    # Normalize the second column
    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])

    # Process the third column
    # Remove projection onto all previous columns
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]
    # Normalize the third column
    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    return B



# The second part of this exercise will generalise the procedure.
# Previously, we could only have four vectors, and there was a lot of repeating in the code.
# We'll use a for-loop here to iterate the process for each vector.

def gsBasis(A):
    B = np.array(A, dtype=np.float_)  # Create a mutable copy of A as B.
    for i in range(B.shape[1]):  # Loop over each column vector in matrix B.
        for j in range(i):  # Loop over all previous vectors to orthogonalize against.
            # Subtract the component of the i-th vector that is in the direction of the j-th vector.
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        # After orthogonalization, check if the vector is non-zero and normalize it.
        if la.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])  # If the vector is near zero, set it to zero vector.
    return B


# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
