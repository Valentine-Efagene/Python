from sympy import fourier_series, pi, cos
from sympy.abc import x
import sympy as sym

#sym.init_printing()

s = fourier_series(x**2, (x, -pi, pi))
res = s.truncate(n=3)

print(res)