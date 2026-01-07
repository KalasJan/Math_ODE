# res ODR y' - y = 2x, y(0) = 2

import sympy as sym
import numpy as num
import matplotlib.pyplot as mal

#definujeme symboliku
x = sym.symbols('x')
y = sym.Function('y')

# rovnice a reseni 
rovnice = sym.Eq(sym.diff(y(x), x) - y(x), 2*x)
reseni = sym.dsolve(rovnice, ics={y(0): 2}) # pocatecni podminka
print (reseni.rhs) 

# graf: 
f = sym.lambdify(x, reseni.rhs, 'numpy')

hx = num.linspace(0, 0.1, 1000) #  hodnoty na ose x
hy = f(hx)

# samotne vykresleni:
mal.figure(figsize=(10, 6))
mal.plot(hx, hy, color="blue")
mal.axhline(0, color='black', linewidth=0.5, linestyle='--')
mal.axvline(0, color='black', linewidth=0.5, linestyle='--')
mal.title("Řešení diferenciální rovnice $y' - y = 2x$, $y(0) = 2$")
mal.xlabel("x")
mal.ylabel("y")
mal.grid(True)

mal.show()