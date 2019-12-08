from sympy.abc import x
from sympy import *

n = 1
T = n * pi
f = sin(x) ** 2
g = integrate(f, (x, 0, T))

print(g)