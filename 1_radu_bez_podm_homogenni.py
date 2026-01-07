# res ODR y' - y = 0

import sympy as sym

# definujeme symboliku
x = sym.symbols('x')
y = sym.Function('y')

# rovnice
rovnice = sym.Eq(sym.diff(y(x), x) - y(x), 0)

# reseni 
reseni = sym.dsolve(rovnice)

# print (reseni) # Eq(y(x), C1*exp(x)) je vysledek

# upraveni zapisu vysledku
C = sym.symbols('C')
uprava = reseni.rhs.subs(sym.Symbol('C1'), C)
vysledek = f'y= {uprava}'
print(vysledek)

# graf
import numpy as num
import matplotlib.pyplot as mal

# prevod na numerickou 
nahrada = uprava.subs(C, 2) # pro C = 2

f = sym.lambdify(x, nahrada, 'numpy')

hx = num.linspace(0, 5, 1000) #  hodnoty na ose x
hy = f(hx)

# samotne vykresleni:
mal.figure(figsize=(10, 6))
mal.plot(hx, hy, label=r"$y = Ce^x$", color="blue")
mal.axhline(0, color='black', linewidth=0.5, linestyle='--')
mal.axvline(0, color='black', linewidth=0.5, linestyle='--')
mal.title("Řešení diferenciální rovnice $y' - y = 0$")
mal.xlabel("x")
mal.ylabel("y")
mal.legend()
mal.grid(True)

mal.show()
