#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerical Methods of Partial Differential Equations.

Provides integration methods and other utility functions such as the RA & RAW
time filters for numerical integration of PDEs.
"""

import numpy as np

def RA_filter(phi, epsilon=0.1):
    """Robert-Asselin-Williams time filter.

    phi: A tuple of phi at time levels (n-1), n, (n+1)

    epsilon: The RA filter weighting

    Takes variable phi at 3 timelevels (n-1), n, (n+1) and recouples the values
    at (n) and (n+1).
    φ_bar(n) = φ(n) + ϵ[ φ(n+1) - 2φ(n) + φ(n-1) ]
    """
    _phi, phi, phi_ = phi
    return (_phi, phi + epsilon*(_phi - 2.0 * phi + phi_), phi_)

def RAW_filter(phi, nu=0.2, alpha=0.53):
    """The RAW time filter, an improvement on RA filter.

    phi: A tuple of phi at time levels (n-1), n, (n+1)

    nu: Equivalent to 2*ϵ; the RA filter weighting

    alpha: Scaling factor for n and (n+1) timesteps.
           With α=1, RAW —> RA.

    For more information, see [Williams 2009].
    """
    _phi, phi, phi_ = phi
    d = nu*0.5*(_phi - 2.0 * phi + phi_)
    return (_phi, phi+alpha*d, phi_ + (alpha-1)*d)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # simple harmonic osicallator example from [Williams 2009]
    xt = lambda x,y,t,omega: -omega*y
    yt = lambda x,y,t,omega: omega*x
    x0, y0 = 1.0, 0.0
    dt = 0.2
    omega = 1.0
    alpha = 0.53  # RAW filter parameter
    t=0.0
    # initialise with a single euler step
    _x = x = x0
    _y = y = y0
    x = _x + dt*xt(x,y,t,omega)
    y = _y + dt*yt(x,y,t,omega)
    xs = [x0,x]
    ys = [y0,y]
    ts = [0, dt]
    # integrate forward using leapfrog method
    for t in np.arange(0+dt,100,dt):
        x_ = _x + 2*dt*xt(x,y,t,omega)
        y_ = _y + 2*dt*yt(x,y,t,omega)
        (_x,x,x_) = RAW_filter((_x,x,x_), alpha=alpha)
        (_y,y,y_) = RAW_filter((_y,y,y_), alpha=alpha)
        # step variables forward
        ts.append(t+dt)
        _x,x = x,x_
        _y,y = y,y_
        xs.append(x)
        ys.append(y)
    ts = np.array(ts)
    xs = np.array(xs)
    ys = np.array(ys)
    print np.array([ts,xs,ys])
    plt.subplot(211)
    plt.plot(ts,xs)
    plt.plot(ts, np.cos(ts), 'grey')
    plt.xlabel('x')
    plt.subplot(212)
    plt.plot(ts,ys)
    plt.plot(ts, np.sin(ts), 'grey')
    plt.ylabel('y')
    plt.show()

# [Williams 2009] - Paul Williams. A Proposed Modification to the Robert–Asselin Time Filter.
