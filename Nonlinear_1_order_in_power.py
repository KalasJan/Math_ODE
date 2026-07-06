# res ODR (y')^2+y = 3x^2+6x, y(0) = 0

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# definujeme symboliku
x = sym.symbols('x')
y = sym.Function('y')

# rovnice
rovnice = sym.Eq(sym.diff(y(x), x)**2 + y(x), 3*x**2 + 6*x)

# reseni 
reseni = sym.dsolve(rovnice)
# print(f'y = {reseni}')

# upraveni zapisu vysledku
C1 = sym.symbols('C1') # zachovani konstanty
vysledek = reseni.rhs # kvuli P strane rovnice
print(f'y = {vysledek}')


# Vyjadreni derivace: y' = sqrt (3x^2+6x)-y
def derivace(x, y):
    return ((3*x**2+6*x)-y)**(1/2)

# Nastavení okrajových podmínek a simulačního intervalu - 
x_eval = np.linspace(0, 3, 1000) # pocatecni podminka, do, pocet bodu
dx = x_eval[1] - x_eval[0]

# Eulerova metoda
y_result = np.zeros(len(x_eval))
y_result[0] = 0 #☻ pocatecni podminka y(0) = 0

for i in range(1, len(x_eval)):
    x_current = x_eval[i-1]
    y_current = y_result[i-1]
    # euler: y_current = y_puvodni + dx * derivace (x_current, y_current)
    y_result[i] = y_current + dx * derivace(x_current, y_current)
 
# vykresleni
plt.figure(figsize=(10, 6))
plt.plot(x_eval, y_result, color="blue", label=r"Numerické řešení $y(x)$")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(r"Řešení nelineární ODR $y \cdot (y')^2 + y = 3x^2+6x $ pro $y(0) = 0$", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper right")
plt.show()