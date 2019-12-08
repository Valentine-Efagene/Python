from sympy.abc import x
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function

#f = implemented_function('f', lambda x: x**2)
#lam_f = lambdify(x, f(x))
#print(lam_f(4))

f = x**2
lam_f = lambdify(x, f)
print(lam_f(4))