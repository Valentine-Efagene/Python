from sympy.abc import x
from sympy import *
from sympy.plotting import plot

n = 1
T = n * pi
f = sin(x) ** 2
g = integrate(f, (x, 0, T))

print(g)
plot(f, g, (x, 0, 4 * T))