#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerical differentiation."""

import numpy as np

from jpy.maths.matrix import tridiag

def make_stencil(n, i, i_, _i):
    """Create a tridiagonal stencil matrix of size n.
    Creates a matrix to dot with a vector for performing discrete spatial computations.
    i, i_ and _i are multipliers of the ith, i+1 and i-1 values of the vector respectively.
    e.g. to calculate an average at position i based on neighbouring values:
    >>> s = make_stencil(N, 0, 0.5, 0.5)
    >>> avg_v = np.dot(s, v)
    The stencil has periodic boundaries.

    Returns an nxn matrix.
    """
    m = tridiag(n, i, i_, _i)
    m[-1,0] = i_
    m[0,-1] = _i
    return m

def make_central_difference1(n, dx):
    """Returns a function dfdx that calculates the first derivative
    of a list of values discretised with n points of constant separation dx.
    >>> x = np.arange(0, 1, 0.01)
    >>> y = np.sin(2*np.pi*x)
    >>> dfdx = make_central_difference1(len(x), 0.01)
    >>> dfdx(y)  #=> ~ 2π cos(2πx)"""
    m = make_stencil(n, 0, 1, -1) / (2*dx)
    def dfdx(phi):
        return np.dot(m, phi)
    return dfdx

def make_central_difference2(n, dx):
    """Returns a function df2dx that calculates the second derivative
    of a list of values discretised with n points of constant separation dx.
    >>> x = np.arange(0, 1, 0.01)
    >>> y = np.sin(2*np.pi*x)
    >>> d2fdx = make_central_difference2(len(x), 0.01)
    >>> dfdx(y)  #=> ~ -4π² sin(2πx)"""
    m = make_stencil(n, -2, 1, 1) / (dx**2)
    def d2fdx(phi):
        return np.dot(m, phi)
    return d2fdx



if __name__ == '__main__':
    # test: plot sin(2pi x) and dy/dx = cos(2pi x) [normalised]
    import matplotlib.pyplot as plt
    X = 1.0
    dx = 0.01
    x = np.arange(0,X,dx)
    y = np.sin(2*np.pi*x/X)
    dfdx = make_central_difference1(len(x), dx)
    d2fdx = make_central_difference2(len(x), dx)
    plt.plot(x, y)
    plt.plot(x, dfdx(y) / (2*np.pi))
    plt.plot(x, d2fdx(y) / (4*np.pi*np.pi))
    plt.legend(['y=sin(x)','dy/dx','d2y/dx'])
    plt.show()
