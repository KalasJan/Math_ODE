# solve ODR with parametr k:  y'' + ky' + 3k = 6x-2

import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

# symbols
x = sm.Symbol('x')
k = sm.Symbol('k')
y = sm.Function('y')

# equation
equation = sm.Eq(sm.diff(y(x), x, 2) + k * sm.diff(y(x), x) + 3 * k, 6*x - 2)

# vysledek
solve = sm.dsolve(equation)
result = solve.rhs  # Non-homogenous part (right)
print(f'General solution: y = {result}')
latex_gen = sm.latex(result)

#constant
C1 = 1
C2 = 1
change = result.subs({sm.Symbol('C1'): C1, sm.Symbol('C2'): C2})     # Nahrazení konstant
print(f'Specific solution (with parametr k): y = {change}')

# plot
f = sm.lambdify((x,k), change, 'numpy')
hx = np.linspace(-2, 2, 1000)

plt.figure(figsize=(10, 6))

k_val = [0.4, 0.5, 1.0, 2.0, 3.0]
for val_k in k_val:
    if val_k == 0: 
        continue # k !=0 (divide)
    exact = change.subs(k, val_k) # exact k to Specific solution
    hy = f(hx, val_k)
    latex_sol = sm.latex(exact)
    plt.plot(hx, hy, label=f"$k = {val_k}: y = {latex_sol}$")

# graph
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(r"Solutions of ODE: $y'' + ky' + 3k = 6x-2$ with some parameters $k$"
          + '\n' + fr"General solution: $y = {latex_gen}$, $C_1 = {C1}$, $C_2 = {C2}$")
plt.xlabel('x') 
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.show()