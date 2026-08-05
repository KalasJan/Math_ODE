# Reste PDR: dx + dy + dz = x + y + z 
# i s okr. podminkou u(0, y, z) = 0

import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

# 1. Definice prostorových proměnných
x, y, z = sm.symbols('x y z', real=True)
C1, C2 = sm.symbols('C1 C2', real=True)

# ===========================================
# A) Obecne reseni

# 1) Definice charakteristických rovnic vyjádřených jako funkce parametru s
# dx/ds = 1, dy/ds = 1, dz/ds = 1
# df/ds = x + y + z
# Víme, že dy/dx = 1 => y = x - C1 => C1 = x - y
# Víme, že dz/dx = 1 => z = x - C2 => C2 = x - z
# => df/dx = x + y + z = 3*x - C1 - C2

pravo = 3 * x - C1 - C2

# 2) prevod na ODE
f_ode = sm.Function('f')(x)
odr = sm.Eq(sm.diff(f_ode, x), pravo)

reseni_odr = sm.dsolve(odr)
f_vyraz = reseni_odr.rhs

# 3) zpetna substituce charakteristik F(x-y, x-z)
vsechny_konstanty = f_vyraz.free_symbols - {x, C1, C2}
sympy_C1 = list(vsechny_konstanty)[0]

F = sm.Function('F')(x - y, x - z)

# 4) obecne reseni
obecne_reseni = f_vyraz.subs({C1: x - y, C2: x - z, sympy_C1: F})
obecne_reseni_zjednodusene = sm.simplify(obecne_reseni)

print(f"u(x, y, z) = {obecne_reseni_zjednodusene}")

# ===============================================================
# B) S podminkou u(0, y, z) = 0

# 5) sestaveni rovnice, x = 0, u = 0
rovnice_podminky = sm.Eq(obecne_reseni_zjednodusene.subs(x, 0), 0)

# vyjadreni F(-y, -z)
funkce_F = list(obecne_reseni_zjednodusene.atoms(sm.core.function.AppliedUndef))[0]
nazev_F = funkce_F.func
vyjadrena_F_negativni = sm.solve(rovnice_podminky, nazev_F(-y, -z))[0]

# substituce F(ar1, ar2)
arg1, arg2 = sm.symbols('arg1 arg2', cls=sm.Dummy)
konkretni_predpis_F = vyjadrena_F_negativni.subs({y: -arg1, z: -arg2})

# zpetna substituce
dosazeni_charakteristik = konkretni_predpis_F.subs({arg1: x - y, arg2: x - z})
konkretni_reseni = obecne_reseni_zjednodusene.subs(funkce_F, dosazeni_charakteristik)
konkretni_reseni_zjednodusene = sm.simplify(konkretni_reseni)

print (f"u(x, y, z) s podmínkou u(0, y, z) = 0 je {konkretni_reseni_zjednodusene}")

# =========================================================================
# C) Graf reseni s podminkou (konkretni hladina, z = cislo)
u_konkretni_num = sm.lambdify((x, y, z), konkretni_reseni_zjednodusene, 'numpy')

# iniciace 3D grafu
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# hodnoty
x_val = np.linspace(0, 5, 1000)
y_val = np.linspace(0, 5, 1000)
X, Y = np.meshgrid(x_val, y_val)
z0 = 3
Z = u_konkretni_num(X, Y, z0)

ax1.view_init(elev=22, azim=-125)

# povrch
povrch = ax1.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.9)

# pocatecni podminka u(0, y) = 0
y_hrana = np.linspace(0, 5, 100)
x_hrana = np.zeros_like(y_hrana)
z_hrana = np.zeros_like(y_hrana)
ax1.plot(x_hrana, y_hrana, z_hrana, color='red', linewidth=4, zorder=5, label=f'Podmínka $u(0, y, z = {z0}) = 0$')

# Nastavení popisů os a vzhledu grafu
ax1.set_title(rf'3D povrch řešení PDR: $u(x, y, z) = {sm.latex(konkretni_reseni_zjednodusene)}$', fontsize=14, pad=20)
ax1.set_xlabel('Osa x', fontsize=11)
ax1.set_ylabel('Osa y', fontsize=11)
ax1.set_zlabel(f'u(x, y, z={z0})', fontsize=11)

plt.legend(loc='upper left', fontsize=9)

# ===================================================================
# D) druhá možnost vykreslení

x_v = np.linspace(0, 4, 20)
y_v = np.linspace(0, 4, 20)
z_v = np.linspace(0, 4, 20)
X2, Y2, Z2 = np.meshgrid(x_v, y_v, z_v)

U = u_konkretni_num(X2, Y2, Z2) # 4D hodnoty

ax2.view_init(elev=22, azim=-125)

# mraky bodu
scat = ax2.scatter(X2, Y2, Z2, c=U, cmap='plasma', alpha=0.5, s=15)
fig.colorbar(scat, ax=ax2, shrink=0.6, aspect=12, label='Hodnota $u(x,y,z)$')

ax2.set_title('Celoprostorový mrak hodnot $u(x,y,z)$')
ax2.set_xlabel('Osa x')
ax2.set_ylabel('Osa y')
ax2.set_zlabel('Osa z')

plt.tight_layout()
plt.show()

