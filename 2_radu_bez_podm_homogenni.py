# res ODR y'' + 2y' -y -3 = 0

import sympy as sym
import numpy as num
import matplotlib.pyplot as mal

#symbolika
x = sym.symbols('x')
y = sym.Function('y')

# rovnice
rovnice = sym.Eq(sym.diff(y(x), x, 2) + 2*sym.diff(y(x),x) - y(x) + 1, 0)

# vysledek
reseni = sym.dsolve(rovnice)
vysledek = reseni.rhs  # Pravá strana rovnice (obecné řešení)
print(f'Obecné řešení: y = {vysledek}')

# konstanty
uprava = vysledek.subs({sym.Symbol('C1'): 2, sym.Symbol('C2'): 3})
     # Nahrazení konstant
print(f'Specifické řešení: y = {uprava}')

# numerika
f = sym.lambdify(x, uprava, 'numpy')
hx = num.linspace(-1, 1, 1000)
hy = f(hx)

# samotne vykresleni:
mal.figure(figsize=(10, 6))
mal.plot(hx, hy, label=f"$y = {vysledek}$", color="blue")
mal.axhline(0, color='black', linewidth=0.5, linestyle='--')
mal.axvline(0, color='black', linewidth=0.5, linestyle='--')
mal.title("Řešení diferenciální rovnice $y'' + 2y' -y -3 = 0$")
mal.xlabel("x")
mal.ylabel("y")
mal.grid(True)

mal.show()


