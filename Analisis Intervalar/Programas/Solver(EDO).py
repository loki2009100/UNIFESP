# importando modulos necesarios
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import sympy 
from scipy import integrate

# imprimir con notación matemática.
sympy.init_printing(use_latex='mathjax')


x = sympy.Symbol('x')
y = sympy.Function('y')

# definiendo la ecuación
eq = 1

# Condición inicial
ics = {y(2): 1}

# Resolviendo la ecuación
edo_sol = sympy.dsolve(y(x).diff(x) - eq)
edo_sol

# Sustituyendo condiciones iniciales
C_eq = sympy.Eq(edo_sol.lhs.subs(x, 0).subs(ics), edo_sol.rhs.subs(x, 0))
C_eq

sympy.solve(C_eq)
