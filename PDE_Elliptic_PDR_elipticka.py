# Vykreslete Eliptickou PDE 
# Poissonova uloha -L u = f
# podminka du/dn =  g(x)

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

# 1) 1D objekt (napr. tyc)

# oblast, na ktere to delame
N = 100  # Počet vnitřních/celkových bodů mřížky
L_domain = 1.0
x = np.linspace(0, L_domain, N)
dx = x[1] - x[0]

f = np.sin(x)  # libovolna funkce f, pro f = 0 je np.zeros(N) 

# Laplacian (L): Lu = d^2 u / dx^2 = 
# = (u_{i-1}-u_i - (u_i - u_{i+1}))/dx^2 = (u_{i+1} - 2u_i + u_{i-1}) / dx^2

main_diag = 2.0 * np.ones(N)
off_diag = -np.ones(N - 1)
A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).tocsc()
A = (A / dx**2)

# okrajova podminka du/dn = g(x)
# v 1D: x=0 => n = -1, x=L -> n = 1

# Hodnota na zacatku (x=0)
A[0, :] = 0
A[0, 0] = 1.0
f[0] = 0.0 # du/dn = 0

# hodnota na konci (x=L)
A[-1, :] = 0
A[-1, -1] = 1.0
f[-1] = 1.0 # du/dn = 0 (jine kladne cislo - tyc je na konci zahrata)

# soustava rovnic (Au=f)
u = spsolve(A, f)

# vykresleni grafu
plt.figure(figsize=(8, 5))
plt.plot(x, u, label='Řešení $u(x)$', color='b', lw=2)
# plt.plot(x, f, label='Pravá strana $f(x)$', color='r', ls='--', alpha=0.7)
plt.title(
    '1D Poissonova úloha (Eliptická PDE)\n$-Lu = f$',
    fontsize=12,
)
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ================================================
# 2D Elipticke (napr. rozmisteni tepla v plose)

Nx, Ny = 10, 10 # na jake oblasti to delame
x2 = np.linspace(0, 1, Nx)
y2 = np.linspace(0, 1, Ny)
dx2 = x2[1] - x2[0]
dy2 = y2[1] - y2[0]

X, Y = np.meshgrid(x2, y2, indexing='ij')

# funkce f
f2d = np.sin(np.pi * X) * np.sin(np.pi * Y)
f_vec = f2d.flatten() # prevod 2D do 1D

# Laplacian - osy
Dx2 = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Nx, Nx)).tocsc() / dx2**2
Dy2 = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Ny, Ny)).tocsc() / dy2**2

# Laplacian a Kroneckeruv soucit (zrychleni vypoctu matice)
Ix = eye(Nx)
Iy = eye(Ny)
A2 = kron(Dx2, Iy) + kron(Ix, Dy2)
A2 = A2.tolil() # Převedeme na LIL pro bezpečné úpravy řádků

# okrajove podminky
is_boundary = (X == 0) | (X == x2[-1]) | (Y == 0) | (Y == y2[-1])
boundary_indices = np.where(is_boundary.flatten())[0]

for idx in boundary_indices:
    A2[idx, :] = 0.0
    A2[idx, idx] = 1.0
    f_vec[idx] = 0.0  # u = 0 na hranici

    
# samotna soustava (A * u = f_vec)
A2 = A2.tocsc()
u_vec = spsolve(A2, f_vec)
U = u_vec.reshape((Nx, Ny)) # zpetny prevod 1D -> 2D

# graf
plt.figure(figsize=(7, 6))
plt.imshow(U.T, # transpozice matice U
           extent=[0, 1, 0, 1], # osy jsou x 0-1, y 0-1
           origin='lower', # kde je poatek (0,0)
           cmap='plasma',
           aspect='auto',) # automaticky pomer stran
plt.colorbar(label='Hodnota $u(x, y)$')
plt.title(
    '2D Poissonova rovnice (Eliptická PDE)\n$-Lu = f$')
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
f3d = np.exp(-(((X3 - 0.5) ** 2 + (Y3 - 0.5) ** 2 + (Z3 - 0.5) ** 2)) / 0.02)
f_pro = f3d.flatten() 

# 3) Laplacian - osy
Dx3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Kx, Kx)).tocsc() / dx3**2
Dy3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Ky, Ky)).tocsc() / dy3**2
Dz3 = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Kz, Kz)).tocsc() / dz3**2

Jx = eye(Kx)
Jy = eye(Ky)
Jz = eye(Kz)

A3 = -(kron(kron(Dx3, Jy), Jz) + kron(kron(Jx, Dy3), Jz) + kron(kron(Jx, Jy), Dz3))
A3 = A3.tolil()


# okrajove podminky (Dirichlet)
is_boundary3 = (X3 == 0) | (X3 == x3[-1]) | (Y3 == 0) | (Y3 == y3[-1]) | (Z3 == 0) | (Z3 == z3[-1])
boundary_indices3 = np.where(is_boundary3.flatten())[0]

# okrajova podminka
A3[boundary_indices3, :] = 0.0
f_pro[boundary_indices3] = 0.0

# maticovy vypocet
for idx in boundary_indices3:
    A3[idx, idx] = 1.0

A3 = A3.tocsc()

# Au = f_pro
u_pro = spsolve(A3, f_pro)

U3 = u_pro.reshape((Kx, Ky, Kz)) # prevod z 1d do 3D

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

# Vykresleni 3D prostoru, kde 4. dimenze (Teplota) je barevne
scat = ax.scatter(X_g.flatten(), Y_g.flatten(), Z_g.flatten(), 
                  c=U_sparse, cmap='plasma', vmin=0.0, vmax=0.05, alpha=0.4, s=15)

cbar = fig.colorbar(scat, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Potenciál pole $u(x,y,z)$', fontsize=10)

plt.suptitle('3D Eliptická rovnice (Poissonova)', fontsize=14, weight='bold', y=0.95)
plt.title(r'Ustálené stacionární rozložení potenciálu $- \Lambda u = f(x,y,z)$', fontsize=11, pad=10)
ax.set_xlabel('Osa X', fontsize=10)
ax.set_ylabel('Osa Y', fontsize=10)
ax.set_zlabel('Osa Z', fontsize=10)

plt.tight_layout()
plt.show()