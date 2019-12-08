from sympy import symbols, sin, cos, pi
from sympy.plotting import plot
from sympy.functions.special.delta_functions import DiracDelta
from matplotlib import style
style.use('ggplot')

x0 = 5
x = symbols('x')
y = 5 + sin(x) + 10 * sin(x + pi/2) + 6 * cos(pi * x)
p = plot(y)