import numpy as np
import gsvd
from tabulate import tabulate

# Create two example matrices (3x6 and 3x6) in Fortran (column-major) order.
A = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
], order='F')

B = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
], order='F')

# Compute the GSVD.
result = gsvd.gsvd(A, B)

# Extract factors
k = result["k"]
l = result["l"]
U = result["U"]
V = result["V"]
Q = result["Q"]
alpha = result["alpha"]
beta = result["beta"]

n = Q.shape[0]
m = U.shape[0]
p = V.shape[0]
r = k + l

# Extract R (upper triangular matrix)
R = result["R"]

# Construct Sigma1 and Sigma2
if m - r >= 0:
    sigma1 = np.block([
        [np.eye(k), np.zeros((k, l))],   
        [np.zeros((l, k)), np.diag(alpha[-l:])],
        [np.zeros((m-k-l, l)), np.zeros((m-k-l, l))]
    ])

    sigma2 = np.block([
        [np.zeros((l, k)), np.diag(beta[-l:])],
        [np.zeros((p-l, k)), np.zeros((p-l, l))]
    ])
else:
    sigma1 = np.block([
        [np.eye(k), np.zeros((k, m-k)), np.zeros((k, k+l-m))],  
        [np.zeros((m-k, k)), np.diag(alpha[k:m]), np.zeros((m-k, k+l-m))] 
    ])

    sigma2 = np.block([
        [np.zeros((m-k, k)), np.diag(beta[k:m]), np.zeros((m-k, k+l-m))],
        [np.zeros((k+l-m, k)), np.zeros((k+l-m, m-k)), np.eye(k+l-m)],
        [np.zeros((p-l, k)), np.zeros((p-l, m-k)), np.zeros((p-l, k+l-m))]
    ])

# Construct [0, R] matrix (r-by-n zero matrix with R)
print("\nR: ")
print(tabulate(R, tablefmt="fancy_grid"))
zero_block = np.zeros((r, n - r))
zero_R = np.hstack([zero_block, R])

# Compute reconstructed A and B
A_reconstructed = U @ sigma1 @ zero_R @ Q.T
B_reconstructed = V @ sigma2 @ zero_R @ Q.T

# Print outputs
print("\nAlpha:")
print(tabulate([[val] for val in alpha], tablefmt="fancy_grid"))

print("\nBeta:")
print(tabulate([[val] for val in beta], tablefmt="fancy_grid"))

print("\nU:")
print(tabulate(U, tablefmt="fancy_grid"))

print("\nV:")
print(tabulate(V, tablefmt="fancy_grid"))

print("\nQ:")
print(tabulate(Q, tablefmt="fancy_grid"))

print("\nR:")
print(tabulate(R, tablefmt="fancy_grid"))

print("\nsigma1:")
print(tabulate(sigma1, tablefmt="fancy_grid"))

print("\nsigma2:")
print(tabulate(sigma2, tablefmt="fancy_grid"))

print("\nReconstructed A:")
print(tabulate(A_reconstructed, tablefmt="fancy_grid"))

print("\n A:")
print(tabulate(A, tablefmt="fancy_grid"))

print("\nReconstructed B:")
print(tabulate(B_reconstructed, tablefmt="fancy_grid"))

print("\n B:")
print(tabulate(B, tablefmt="fancy_grid"))
