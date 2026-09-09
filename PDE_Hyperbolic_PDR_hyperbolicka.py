# Vykreslete Hyperbolickou PDE / vlnova rovnice
# Poissonova uloha d2u/dt2 - c2* L u = f
# podminka du/d omega =  0 // u(x,0) = g(x) (pevna hranice) // du/dt (x,) = h(x)

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron

# 1) 1D objekt (napr. tyc)

# 1) oblast, na ktere to delame
N = 100  # Počet vnitřních/celkových bodů mřížky
L_domain = 1.0
x = np.linspace(0, L_domain, N)
dx = x[1] - x[0]

f = np.zeros(N)  # libovolna funkce f, pro f = 0 je np.zeros(N) - žádné vnitřní napájení 

# 2) čas
t_max = 1 # kolik sec trvá simulace
dt = 0.0005  # velikost casoveho kroku, musi byt extremne maly
kroku = int(t_max / dt)

# rychlost sireni vlny
c = 1

# 3) Laplacian (L): Lu = d^2 u / dx^2 = 
# = (u_{i-1}-u_i - (u_i - u_{i+1}))/dx^2 = (u_{i+1} - 2u_i + u_{i-1}) / dx^2

main_diag = -2.0 * np.ones(N)
off_diag = np.ones(N - 1)
A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).tocsc()
A = (A / dx**2)

# 4) pocatecni podm. u(x,0) = g(x)
u = np.exp(-((x - 0.5) ** 2) / 0.005)
u_initial = u.copy()

# 1. casovy krok, du/dt = 0
d2u_dx2 = A.dot(u)
u_prev = u - 0.5 * dt**2 * (c**2 * d2u_dx2 + f)

# 5) prace s casem
for step in range(kroku):
  d2u_dx2 = A.dot(u)
  
  # vlnova rovnice: u_new = 2*u - u_prev + dt^2 * (c^2 * Lu + f)
  u_new = 2 * u - u_prev + (dt * c) ** 2 * d2u_dx2 + (dt**2) * f

  # posun v historii
  u_prev = u.copy()
  u = u_new.copy()
  
  # okrajova podminka - upevneni struny
  u[0] = 0.0
  u[-1] = 0.0

# vykresleni grafu
plt.figure(figsize=(8, 5))
plt.plot(x, u_initial,
         label='Počáteční stav $u(x, 0)$',
         color='gray', ls='--',lw=1.5,)
plt.plot(x,u,
    label=f'Stav po čase $t = {t_max}$ s',
    color='royalblue', lw=2.5,)
plt.title(
    '1D Hyperbolická rovnice (Vlnová rovnice)\nOdraz a šíření vln', fontsize=12
)
plt.xlabel('Pozice $x$')
plt.ylabel('Vychýlení $u(x, t)$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ================================================
# 2D Hyperbolicka (napr. vlnwni na blane / desce)

Nx, Ny = 50, 50 # na jake oblasti to delame
x2 = np.linspace(0, 1, Nx)
y2 = np.linspace(0, 1, Ny)
dx2 = x2[1] - x2[0]
dy2 = y2[1] - y2[0]

X, Y = np.meshgrid(x2, y2, indexing='ij')

# funkce f
f2d = np.zeros((Nx, Ny))
f_vec = f2d.flatten() # prevod 2D do 1D

# 2) čas
t2_max = 1 # kolik sec trvá simulace
dt2 = 0.0005 # velikost casoveho kroku
kroku2 = int(t2_max / dt2)

# koeficient vedeni tepla
c2 = 1

# 3) Laplacian - osy
Dx2 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Nx, Nx)).tocsc() / dx2**2
Dy2 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Ny, Ny)).tocsc() / dy2**2

Ix = eye(Nx)
Iy = eye(Ny)
A2 = kron(Dx2, Iy) + kron(Ix, Dy2)

# 4) Pocateni podminky (klepnuti do stredu desky)
U_init = np.exp(-(((X - 0.5) ** 2 + (Y - 0.5) ** 2)) / 0.02)
u_vec = U_init.flatten()

# okrajove podminky
is_boundary2 = (X == 0) | (X == x2[-1]) | (Y == 0) | (Y == y2[-1])
boundary_indices2 = np.where(is_boundary2.flatten())[0]

u_vec[boundary_indices2] = 0.0 # na okrajich nula

# 1. krok historie
Lu_vec = A2.dot(u_vec)
u_prev_vec = u_vec - 0.5 * dt2**2 * (c2**2 * Lu_vec + f_vec)
u_prev_vec[boundary_indices2] = 0.0

# 5) prace s casem
for step in range(kroku2):
  Lu_vec = A2.dot(u_vec)

  u_new_vec = 2.0 * u_vec - u_prev_vec + (dt2 * c2) ** 2 * Lu_vec # Eulerova metoda
 
  u_prev_vec = u_vec.copy()
  u_vec = u_new_vec.copy()

  u_vec[boundary_indices2] = 0.0 # okraj

U = u_vec.reshape((Nx, Ny)) # prevod 1D -> 2D
    
# graf
plt.figure(figsize=(7, 6))
plt.imshow(U.T, extent=[0, 1, 0, 1], origin='lower', cmap='seismic',
    aspect='auto', vmin=-0.5, vmax=0.5,)
plt.colorbar(label=f'Vychýlení $u(x, y)$ v t = {t2_max} s')
plt.title(
    f'2D Hyperbolická rovnice (Vlnění na bláně)\nStav po čase $t = {t2_max}$ s'
)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()

# ====================================================================
# 3D model

Kx, Ky, Kz = 50, 50, 50 # na jake oblasti to delame
x3 = np.linspace(0, 1, Kx)
y3 = np.linspace(0, 1, Ky)
z3 = np.linspace(0, 1, Kz)
dx3 = x3[1] - x3[0]
dy3 = y3[1] - y3[0]
dz3 = z3[1] - z3[0]

X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing='ij')

# funkce f
f3d = np.zeros((Kx, Ky, Kz))
f_pro = f3d.flatten() # prevod 3D do 1D

# 2) čas
t3_max = 1 # kolik sec trvá simulace
dt3 = 0.0005 # velikost casoveho kroku
kroku3 = int(t3_max / dt3)

# koeficient vedeni tepla
c3 = 1

# 3) Laplacian - osy
Dx3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Kx, Kx)).tocsc() / dx3**2
Dy3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Ky, Ky)).tocsc() / dy3**2
Dz3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Kz, Kz)).tocsc() / dz3**2

Jx = eye(Kx)
Jy = eye(Ky)
Jz = eye(Kz)
A3 = kron(kron(Dx3, Jy), Jz) + kron(kron(Jx, Dy3), Jz) + kron(kron(Jx, Jy), Dz3)

# 4) Pocateni podminky (klepnuti do stredu desky)
U_init3 = np.exp(-(((X3 - 0.5) ** 2 + (Y3 - 0.5) ** 2 + (Z3 - 0.5) ** 2)) / 0.02)
u_pro = U_init3.flatten()

# okrajove podminky
is_boundary3 = (X3 == 0) | (X3 == x3[-1]) | (Y3 == 0) | (Y3 == y3[-1]) | (Z3 == 0) | (Z3 == z3[-1])
boundary_indices3 = np.where(is_boundary3.flatten())[0]

u_pro[boundary_indices3] = 0.0 # na okrajich nula

# 1. krok historie
Lu_pro = A3.dot(u_pro)
u_prev_pro = u_pro - 0.5 * dt3**2 * (c3**2 * Lu_pro + f_pro)
u_prev_pro[boundary_indices3] = 0.0

# 5) prace s casem
for step in range(kroku3):
  Lu_pro = A3.dot(u_pro)

  u_new_pro = 2.0 * u_pro - u_prev_pro + (dt3 * c3) ** 2 * Lu_pro + (dt3**2) * f_pro # Eulerova metoda
 
  u_prev_pro, u_pro = u_pro, u_new_pro

  u_pro[boundary_indices3] = 0.0 # okraj

U3 = u_pro.reshape((Kx, Ky, Kz)) # prevod 1D -> 3D
    
# graf (4D)
# vykreslime kazdy 2. bod (step_v)
step_v = 2
x_v = x3[::step_v]
y_v = y3[::step_v]
z_v = z3[::step_v]
X_g, Y_g, Z_g = np.meshgrid(x_v, y_v, z_v, indexing='ij')

# Vytahneme data z vysledne matice
U_sparse = U3[::step_v, ::step_v, ::step_v].flatten()

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=-135)

# Vykresleni 3D prostoru, kde 4. dimenze (akustický tlak) je barevne
scat = ax.scatter(X_g.flatten(), Y_g.flatten(), Z_g.flatten(), 
                  c=U_sparse, cmap='seismic', vmin=-0.2, vmax=0.2, alpha=0.4, s=15)

cbar = fig.colorbar(scat, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Akustické vychýlení / Tlak $u(x,y,z)$', fontsize=10)

plt.suptitle('3D Hyperbolická vlnová rovnice', fontsize=14, weight='bold', y=0.95)
plt.title(f'Volumetrické pole šíření vln v čase $t = {t3_max}$ s', fontsize=11, pad=10)
ax.set_xlabel('Osa X', fontsize=10)
ax.set_ylabel('Osa Y', fontsize=10)
ax.set_zlabel('Osa Z', fontsize=10)

plt.tight_layout()
plt.show()
