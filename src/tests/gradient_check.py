
import numpy as np


def numerical_gradient(f, x, eps=1e-3):
    return (f(x+eps) - f(x-eps)) / 2*eps


def gradient_checking(f, gx, x, eps=1e-3, atol=1e-5, rtol=1e-4):
    true_gx = numerical_gradient(f, x, eps)
    np.testing.assert_allclose(gx, true_gx, atol=atol, rtol=rtol)
