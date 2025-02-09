import os
import ctypes
import numpy as np

# Determine the path to the shared library.
# Here we assume the shared library (e.g. libgsvd.so) is located in the same directory as this __init__.py.
_lib_path = os.path.join(os.path.dirname(__file__), "libgsvd.so")
_lib = ctypes.CDLL(_lib_path)

# Define the argument and return types for the gsvd_wrapper function.
# The function prototype in C is:
#   int gsvd_wrapper(int m, int n, int p,
#                    double* A, double* B,
#                    double* alpha, double* beta,
#                    double* U, double* V, double* Q,
#                    int* k, int* l);
_lib.gsvd_wrapper.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, flags="F_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
_lib.gsvd_wrapper.restype = ctypes.c_int

def gsvd(A, B):
    """
    Compute the generalized singular value decomposition (GSVD) of two real matrices A and B.
    
    Parameters:
        A (ndarray): Real matrix of shape (m, n) stored in column-major order.
        B (ndarray): Real matrix of shape (p, n) stored in column-major order.
                    (The number of columns must match that of A.)
                    
    Returns:
        dict: A dictionary containing:
            - "alpha": Generalized singular values (array of length n).
            - "beta":  Generalized singular values (array of length n).
            - "U":     Orthogonal matrix of shape (m, m).
            - "V":     Orthogonal matrix of shape (p, p).
            - "Q":     Orthogonal matrix of shape (n, n).
            - "k":     Integer parameter from GSVD.
            - "l":     Integer parameter from GSVD.
            
    Raises:
        ValueError: If the number of columns in A and B differ.
        RuntimeError: If the GSVD computation fails.
    """
    # Convert inputs to NumPy arrays in Fortran (column-major) order.
    A = np.array(A, dtype=np.double, order='F')
    B = np.array(B, dtype=np.double, order='F')
    
    m, nA = A.shape
    p, nB = B.shape
    if nA != nB:
        raise ValueError("Matrices A and B must have the same number of columns.")
    n = nA

    # Allocate output arrays (in Fortran order)
    alpha = np.empty(n, dtype=np.double, order='F')
    beta  = np.empty(n, dtype=np.double, order='F')
    U = np.empty((m, m), dtype=np.double, order='F')
    V = np.empty((p, p), dtype=np.double, order='F')
    Q = np.empty((n, n), dtype=np.double, order='F')
    R = []
    k = ctypes.c_int()
    l = ctypes.c_int()

    # Call the C wrapper function.
    info = _lib.gsvd_wrapper(m, n, p,
                             A, B,
                             alpha, beta,
                             U, V, Q,
                             ctypes.byref(k), ctypes.byref(l))
    if info != 0:
        raise RuntimeError(f"GSVD computation failed with info = {info}")

    r = k.value + l.value
    m_k_l = m - k.value - l.value

    if m_k_l >= 0:
        R = A[:r, -r:].copy()
    else:
        R_except_R33 = A[:m, -r:].copy()
        R33 = B[m - k.value : m - k.value + l.value, n + m - k.value - l.value :]

        R = np.vstack([R_except_R33, np.hstack([np.zeros((R33.shape[0], r - R33.shape[1])), R33])])


    return {
        "alpha": alpha,
        "beta": beta,
        "U": U,
        "V": V,
        "Q": Q,
        "R": R,
        "k": k.value,
        "l": l.value
    }
