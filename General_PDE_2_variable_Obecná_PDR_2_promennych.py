# Reste parcialni diferencialni rovnici
# df/dx + df/dy = x + y

import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

# 1) definice promennych
x, y= sm.symbols('x y')

# 2) definice reseni
u = sm.Function('u')(x, y)

# 3) zadani rovnice (musi byt rovno 0)
pde = sm.diff(u, x) + sm.diff(u, y)  - (x + y)

# 4) reseni
reseni = sm.pdsolve(pde)
u_expr = reseni.rhs

print(f"Obecné řešení rovnice je u(x,y) = {u_expr}")

# ====================================================
# s pocatecni podminkou u (0, y) = 0

# 5) vyjareni rovnice pro x = 0 a u = 0
# u_expr.subs(x, 0) nám dá výraz s funkcí F(x - y)
rovnice_podminky = sm.Eq(u_expr.subs(x, 0), 0)

# 6) nalezeni nezname funkce F
# AppliedUndef = aplikovaná neznámá funkce
funkce_F = list(reseni.atoms(sm.core.function.AppliedUndef))[0] # vrati F(x - y)
nazev_F = funkce_F.func  # Toto je samotný symbol funkce F

# 7) vyjaddreni F(-y) z podminky
vyjadrena_F_negativni = sm.solve(rovnice_podminky, nazev_F(-y))[0]

# substituce t = -y
t = sm.symbols('t')
predpis_F = vyjadrena_F_negativni.subs(y, -t)

# 8) konkretni reseni F(t)
konkretni_F = predpis_F.subs(t, x - y)

# konkretni reseni (v promennych x, y)
konkretni_reseni = u_expr.subs(funkce_F, konkretni_F)
konkretni_reseni_zjednodusene = sm.simplify(konkretni_reseni)

print(f"Obecné řešení rovnice s podmínkou u(0,y) je u(x,y) = {konkretni_reseni_zjednodusene}")

# ============================================================
# 9) vykesleni
u_konkretni_num = sm.lambdify((x, y), konkretni_reseni_zjednodusene, 'numpy')

# hodnoty
x_val = np.linspace(0, 5, 1000)
y_val = np.linspace(0, 5, 1000)
X, Y = np.meshgrid(x_val, y_val)
Z = u_konkretni_num(X, Y)

# iniciace 3D grafu
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-60)

# povrch
povrch = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# pocatecni podminka u(0, y) = 0
y_hrana = np.linspace(0, 5, 100)
x_hrana = np.zeros_like(y_hrana)
z_hrana = np.zeros_like(y_hrana)
ax.plot(x_hrana, y_hrana, z_hrana, color='red', linewidth=4, zorder=5, label='Podmínka $u(0, y) = 0$')

# Nastavení popisů os a vzhledu grafu
ax.set_title(rf'3D povrch řešení PDR: $u(x, y) = {sm.latex(konkretni_reseni_zjednodusene)}$', fontsize=14, pad=20)
ax.set_xlabel('Osa x', fontsize=11)
ax.set_ylabel('Osa y', fontsize=11)
ax.set_zlabel('u(x, y)', fontsize=11)

plt.legend()
plt.show()