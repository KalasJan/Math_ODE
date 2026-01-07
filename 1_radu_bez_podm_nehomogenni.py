# res ODR y' - y = 2x

import sympy as sym

# definujeme symboliku
x = sym.symbols('x')
y = sym.Function('y')

# rovnice
rovnice = sym.Eq(sym.diff(y(x), x) - y(x), 2*x)

# reseni 
reseni = sym.dsolve(rovnice)
# print(f'y = {reseni}')

# upraveni zapisu vysledku
C1 = sym.symbols('C1') # zachovani konstanty
vysledek = reseni.rhs # kvuli P strane rovnice
print(f'y = {vysledek}')

# graf
import numpy as num
import matplotlib.pyplot as mal

# prevod na numerickou 
p = 4
nahrada = vysledek.subs(C1, p) # nahrazeni konstanty cislem

f = sym.lambdify(x, nahrada, 'numpy')

hx = num.linspace(-5, 2, 1000) #  hodnoty na ose x
hy = f(hx)

# samotne vykresleni:
mal.figure(figsize=(10, 6))
mal.plot(hx, hy, color="blue")
mal.axhline(0, color='black', linewidth=0.5, linestyle='--')
mal.axvline(0, color='black', linewidth=0.5, linestyle='--')
mal.title(f"Řešení diferenciální rovnice $y' - y = 2x$ s parametrem C1 = {p}")
mal.xlabel("x")
mal.ylabel("y")
mal.grid(True)

mal.show()