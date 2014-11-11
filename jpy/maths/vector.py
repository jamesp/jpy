import numpy as np

def vec_abs(x):
    """Returns the length of a vector (or list of vectors)"""
    return np.sqrt(np.sum(x**2, axis=x.ndim-1))

def normalise(u):
    return u / vec_abs(u)

def project(u, v):
    return u * (np.dot(u, v) / np.dot(u, u))

def gram_schmidt(vs, normalised=True):
    """Gram-Schmidt Orthonormalisation / Orthogonisation.
    Given a set of vectors, returns an orthogonal set of vectors
    spanning the same subspace.
    Set `normalised` to False to return a non-normalised set of vectors."""
    us = []
    for v in np.array(vs):
        u = v - np.sum([project(x, v) for x in us], axis=0)
        us.append(u)
    if normalised:
        return np.array([normalise(u) for u in us])
    else:
        return np.array(us)


def gram_schmidt_modified(vs, normalised=True):
    """Gram-Schmidt Orthonormalisation / Orthogonisation.
    Given a set of vectors, returns an orthogonal set of vectors
    spanning the same subspace.
    Set `normalised` to False to return a non-normalised set of vectors.

    (Modified version as per p.80 Practical Numerical Algorithms
        - Parker & Chua)"""
    us = []
    for v in np.array(vs):
        u = v
        for x in us:
            u = u - project(x, u)
        us.append(u)
    if normalised:
        return np.array([normalise(u) for u in us])
    else:
        return np.array(us)


if __name__ == '__main__':
    ## testing  Gram-Schmidt works as expected
    print("Testing Gram-Schmidt orthonormalisation.")

    randmat = np.random.random((3,3))
    print("Random matrix:")
    print(randmat)
    gsrm = gram_schmidt(randmat)
    print("Normalised:")
    print(gsrm)
    print("x1 . x2 = ", np.dot(gsrm[0], gsrm[1]))
    print("x1 . x3 = ", np.dot(gsrm[0], gsrm[2]))
    print("x2 . x3 = ", np.dot(gsrm[1], gsrm[2]))
    print("x2 . x2 = ", np.dot(gsrm[1], gsrm[1]))
