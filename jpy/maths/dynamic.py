"""Functions for examining the properties of dynamical systems."""

from functools import wraps

import numpy as np

from jpy.maths.diffeqns import RK4
from jpy.maths.vector import vec_abs, gram_schmidt

def augment_position(x, u):
    """Combine phase-space position and linearised vector matrix."""
    return np.append([x], u, axis=0)

def unaugment_position(xu):
    """Break apart an augmented position matrix into phase-space position and perturbation matrix."""
    return (xu[0], xu[1:])

def lyapunov_exponents(x0, u0, T, h, f):
    """Numerically calculate the Lyapunov Exponents of the augmented
    nonlinear system f.
    x0:  Initial Position in phase space.
    u0:  A list of initial perturbation vectors.
    f:  The system of nonlinear and linearised equations.
    Returns a list of Lyapunov Exponents."""
    x, u = x0, u0
    sums= [0,0,0]
    for i in range(int(T/h)):
        xu = augment_position(x, u)
        xu1 = RK4(xu, h, extended_lorenz)
        x, u = unaugment_position(xu1)
        v = gram_schmidt(u, normalised=False)
        u = gram_schmidt(v)
        sums += np.log(vec_abs(v))
    return sums / T

def kaplan_yorke_dimension(lyapunov_exponents):
    """Calculates the Kaplan Yorke Dimension of a list of Lyapunov exponents.
        D = k + sum[i = 1 to k](exponent[i] / abs(exponent[k+1])
    where k is the largest number of exponents that can be summed together to a total > 0.
    """
    lexp = np.sort(lyapunov_exponents)[::-1]
    k = np.sum(lexp.cumsum() > 0)
    return k + np.sum([e / np.abs(lexp[k]) for e in lexp[:k]])

### The Lorenz Equations

def lorenz(xyz, sigma, r, b):
    x, y, z = xyz
    xdot = sigma * (y - x)
    ydot = x * (r - z) - y
    zdot = x * y - b * z
    return np.array([xdot, ydot, zdot])

def fix_lorenz_params(sigma, r, b):
    """Fix the lorenz attractor with set of parameters.
       This can then be passed to a numerical integration method."""
    @wraps(lorenz)
    def fixed_lorenz(x):
        return lorenz(x, sigma, r, b)
    return fixed_lorenz

STD_LORENZ = fix_lorenz_params(10.0, 28.0, 8.0/3.0)


if __name__ == '__main__':
    T =  100
    h =  0.001
    x0 = x = np.array([5.2, 8.5, 27.0])
    u0 = np.identity(3)

    def extended_lorenz(xu, sigma=10.0, r=28.0, b=8.0/3.0):
        """Extended Lorenz system includes the linearised equations.
        Given a 4 vector list `xu`, where row 1 is the position in
        phase space and rows 2-4 are 3 perturbation vectors.
        Returns 4 vector list of the same format for the new position."""
        xyz, u = unaugment_position(xu)
        x,y,z = xyz
        xyz_ = lorenz(xyz, sigma, r, b)
        u_ = []
        for dx, dy, dz in u:
            dxdot = -sigma*dx + sigma*dy
            dydot = (r - z) * dx - dy - x * dz
            dzdot = y * dx + x * dy - b * dz
            u_.append([dxdot, dydot, dzdot])
        return augment_position(xyz_, u_)

    print("Calculating Lyapunov exponents of the Lorenz Equations")
    lyxp = lyapunov_exponents(x0, u0, T, h, extended_lorenz)

    print("Lyapunov Exponents:")
    for i, l in enumerate(lyxp):
        print("lambda %d: %.4f"% (i+1, l))

    # Lyapunov Exponents should sum to divergence  # (-13.6666...)
    print("Sum of exponents: %.5f (should match divergence: div(u) = -13.666...)" % sum(lyxp))

    print("Kaplan-Yorke dimension: %.3f" % kaplan_yorke_dimension(lyxp))
