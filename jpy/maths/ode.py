import numpy

def forward_euler(x,h,f):
    """Forward Euler integration method for an autonomous system f."""
    return x + h * f(x)

def RK4(x, h, f):
    """Runge-Kutta 4th-order integration method for an autonomous system f."""
    K1 = h*f(x)
    K2 = h*f(x+(K1 * 0.5))
    K3 = h*f(x+(K2 * 0.5))
    K4 = h*f(x+K3)
    return x+(1/6.0)*(K1 + 2*K2 + 2*K3 + K4)

def double_approx(x, h, f):
    """The method used by E. Lorenz in his original study of the equations."""
    x1 = forward_euler(x, h, f)
    x2 = forward_euler(x1, h, f)
    return 0.5*(x + x2)
