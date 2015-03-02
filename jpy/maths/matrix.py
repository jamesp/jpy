import numpy as np

def tridiag(n, d, u, l):
    """Create an nxn tridiagonal matrix with `d` along diagonal, `u` on
    first upper diag and `l` on first lower"""
    M = np.zeros((n,n))
    i,j = np.indices((n,n))
    M[i == j] = d
    M[i == j-1] = u
    M[i == j+1] = l
    return M
