# res ODR y'' + 2y' -y -3 = 5x^2+3x
# y(0) = 0, y(1) = 2

import sympy as sym
import numpy as num
import matplotlib.pyplot as mal

#symbolika
x = sym.symbols('x')
y = sym.Function('y')

# rovnice
rovnice = sym.Eq(sym.diff(y(x), x, 2) + 2*sym.diff(y(x),x) - y(x) + 1, 5*x**2+3*x)

# vysledek
reseni = sym.dsolve(rovnice)
vysledek = reseni.rhs  # Pravá strana rovnice (obecné řešení)
print(f'Obecné řešení: y = {vysledek}')

# konstanty
C1, C2 = sym.symbols('C1 C2')

# podminky
podminky = [
    sym.Eq(vysledek.subs(x, 0), 0),  # y(0) = 0
    sym.Eq(vysledek.subs(x, 1), 0)] # y(1) = 2

# hodnoty konstant podle podminek
konstanty = sym.solve(podminky, (C1, C2))
print (f'Konstanty jsou: {konstanty}')

#nahrazeni, substituce konstant
substituce = vysledek.subs(konstanty)
print(f'Konkretni reseni je y = {substituce}')

# numerika
f = sym.lambdify(x, substituce, 'numpy')
hx = num.linspace(0, 1, 1000)
hy = f(hx)

# samotne vykresleni:
mal.figure(figsize=(10, 6))
mal.plot(hx, hy, label=f"$y = {vysledek}$", color="blue")
mal.axhline(0, color='black', linewidth=0.5, linestyle='--')
mal.axvline(0, color='black', linewidth=0.5, linestyle='--')
mal.title("Řešení diferenciální rovnice $y'' + 2y' -y -3 = 5x^2+3x$")
mal.xlabel("x")
mal.ylabel("y")
mal.grid(True)

mal.show()
