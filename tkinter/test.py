from sympy.abc import x, y
from sympy import *

f = (x**2 - 1) / (1 + x**2) * 1 / (1 + x**4)
print integrate(f, x)
