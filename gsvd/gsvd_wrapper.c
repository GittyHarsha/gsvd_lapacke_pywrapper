#include <lapacke.h>
#include <stdlib.h>

/* 
 * gsvd_wrapper
 *
 * Computes the GSVD of two real matrices A and B (stored in column-major order)
 * using the LAPACKE_dggsvd3 routine.
 *
 * Parameters:
 *   m    : Number of rows of A.
 *   n    : Number of columns of A (and B).
 *   p    : Number of rows of B.
 *   A    : Pointer to the m x n matrix A.
 *   B    : Pointer to the p x n matrix B.
 *   alpha: Output array of length n.
 *   beta : Output array of length n.
 *   U    : Output m x m orthogonal matrix.
 *   V    : Output p x p orthogonal matrix.
 *   Q    : Output n x n orthogonal matrix.
 *   k    : Output pointer for integer k.
 *   l    : Output pointer for integer l.
 *
 * Returns 0 on success, or a nonzero value if LAPACKE_dggsvd3 fails.
 */
int gsvd_wrapper( int m, int n, int p,
                  double* A, double* B,
                  double* alpha, double* beta,
                  double* U, double* V, double* Q,
                  int* k, int* l )
{
    /* Allocate workspace (iwork array of size at least n) */
    int* iwork = (int*)malloc(n * sizeof(int));
    if (!iwork) return -1;  // Memory allocation error

    int info = LAPACKE_dggsvd3(LAPACK_COL_MAJOR, 'U', 'V', 'Q',
                               m, n, p, k, l,
                               A, m, B, p, alpha, beta,
                               U, m, V, p, Q, n,
                               iwork);
    free(iwork);
    return info;
}
