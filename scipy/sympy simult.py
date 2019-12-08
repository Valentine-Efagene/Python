import numpy as np
from sympy import linsolve, Matrix, I
from sympy.abc import x, y, z
from sympy import init_printing
init_printing()

A = Matrix([[40, I*20], [I*30, 60]])
B = Matrix([100, 0])
res = linsolve( (A, B), [x, y] )
#res
print(res)