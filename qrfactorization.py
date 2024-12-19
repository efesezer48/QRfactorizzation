"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)  # For cleaner output

# Generate a 5x4 random matrix
A = np.random.normal(size=(5, 4))
print("Original Matrix A:")
print(A)
print("\nShape of A:", A.shape)

# Compute QR factorization using NumPy
Q, R = np.linalg.qr(A)

print("\nQ matrix (orthogonal matrix):")
print(Q)
print("\nShape of Q:", Q.shape)

print("\nR matrix (upper triangular):")
print(R)
print("\nShape of R:", R.shape)

# Verify the factorization
print("\nVerification:")
print("1. Check if Q*R equals original A:")
print("Difference between A and Q*R:")
print(np.abs(A - Q @ R))
print("\nMaximum difference:", np.max(np.abs(A - Q @ R)))

# Verify Q is orthogonal (Q^T * Q should be identity)
print("\n2. Verify Q is orthogonal (Q^T * Q should be identity):")
print(Q.T @ Q)

# Check R is upper triangular
print("\n3. Verify R is upper triangular:")
is_upper = np.allclose(R, np.triu(R))
print("Is R upper triangular?", is_upper)

"""
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)  # For cleaner output

# Define the matrices
A = np.array([[-3, -4], [4, 6], [1, 1]])
B = np.array([[-11, -10, 16], [7.8, -11, 9]])
C = np.array([[0, -1, 6], [0.1, -4, 0.5]])  # Fixed C matrix with proper dimensions

# Check dimensions
print("Matrix dimensions:")
print("A shape:", A.shape)
print("B shape:", B.shape)
print("C shape:", C.shape)

# Function to check if a matrix is a left inverse
def is_left_inverse(M, A):
    try:
        result = M @ A
        identity = np.eye(result.shape[0])
        is_identity = np.allclose(result, identity, rtol=1e-10)
        print("\nM @ A =")
        print(result)
        print("Is identity matrix?", is_identity)
        return is_identity
    except ValueError as e:
        print("\nError computing M @ A:", e)
        return False

# Check if B is a left inverse of A
print("\nChecking if B is a left inverse of A:")
is_B_left_inverse = is_left_inverse(B, A)

# Check if C is a left inverse of A
print("\nChecking if C is a left inverse of A:")
is_C_left_inverse = is_left_inverse(C, A)

"""
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)  # For cleaner output

# Generate a random 3x3 matrix
A = np.random.rand(3, 3)

print("Original Matrix A:")
print(A)

# Calculate the inverse
A_inv = np.linalg.inv(A)

print("\nInverse Matrix A^(-1):")
print(A_inv)

# Verify the inverse by multiplying A * A^(-1)
print("\nVerification A * A^(-1) (should be identity matrix):")
print(A @ A_inv)

# Verify the inverse by multiplying A^(-1) * A
print("\nVerification A^(-1) * A (should be identity matrix):")
print(A_inv @ A)
"""
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Generate the 5x4 random matrix with normal distribution
np.random.seed(42)
A = np.random.normal(size=(5, 4))

# Compute QR factorization
Q, R = np.linalg.qr(A)

# Get the pseudoinverse
R_square = R[:4, :]
R_inv = np.linalg.inv(R_square)
Q_T = Q.T
A_inv = R_inv @ Q_T[:4, :]

print("Original matrix A shape:", A.shape)
print("\nOriginal matrix A:")
print(A)

print("\nPseudoinverse of A shape:", A_inv.shape)
print("\nPseudoinverse of A:")
print(A_inv)

# Verify the pseudoinverse without special characters
print("\nVerification (A * A_inv * A should equal A):")
verification = A @ A_inv @ A
print(np.allclose(verification, A))
print("\nDifference between A and (A * A_inv * A):")
print(np.abs(A - verification))
"""

import numpy as np
import time

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return end_time - start_time

# Set the size
n = 5000
print(f"Matrix size: {n}x{n}")

# Generate random nxn matrix and n-vector
A = np.random.rand(n, n)
b = np.random.rand(n)

# 1. Measure Ax=b (using solve)
print("\n1. Solving Ax=b")
solve_time = measure_time(np.linalg.solve, A, b)
print(f"Time taken: {solve_time:.2f} seconds")
print(f"Theoretical complexity: O(n³)")

# 2. Measure A²
print("\n2. Computing A²")
square_time = measure_time(np.matmul, A, A)
print(f"Time taken: {square_time:.2f} seconds")
print(f"Theoretical complexity: O(n³)")

# 3. Measure A inverse
print("\n3. Computing A inverse")
inverse_time = measure_time(np.linalg.inv, A)
print(f"Time taken: {inverse_time:.2f} seconds")
print(f"Theoretical complexity: O(n³)")

# 4. Measure A³
print("\n4. Computing A³")
cube_time = measure_time(lambda x: np.matmul(np.matmul(x, x), x), A)
print(f"Time taken: {cube_time:.2f} seconds")
print(f"Theoretical complexity: O(n³) but requires 2 matrix multiplications")
print("\nTheoretical Analysis:")
print("1. Solving Ax=b: O(n³) using LU decomposition")
print("2. Computing A²: O(n³) for matrix multiplication")
print("3. Computing inverse: O(n³) using LU decomposition")
print("4. Computing A³: O(n³) but with larger constant factor (2 multiplications)")
baseline = solve_time
print(f"Ax=b (baseline): 1.00x")
print(f"A²: {square_time/baseline:.2f}x")
print(f"A inverse: {inverse_time/baseline:.2f}x")
print(f"A³: {cube_time/baseline:.2f}x")
