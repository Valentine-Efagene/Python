from sympy import fourier_series, pi, cos
from sympy.abc import x
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from sympy import *

s = fourier_series(x**2, (x, -pi, pi))
res = s.truncate(n=9)

print(res)
lam_f = lambdify(x, res)
print(lam_f(4))

plot(res)