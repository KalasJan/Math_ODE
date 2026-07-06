# res ODR (y'')^2 +2y'- y = 5*sin(e^x), y(0) = 1, y'(1) = 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# definujeme soustavu rovnice: y'' = (5*sin(e^x)-2y'+y ) ^ (1/2)
def soustava (x,y):
    y1, y2 = y
    y1_der = y2
    
    # Vyjádření zrychlení y'' z rovnice (s ošetřením záporného vnitřku odmocniny)
    pravo = 5 * np.sin(np.exp(x)) - 2 * y2 + y1
    y2_der = np.sqrt(np.maximum(0, pravo))
    
    return np.vstack((y1_der, y2_der))
 
# okrajove podminky
def podminky(ya, yb):
    return np.array([ya[0] - 1, yb[1] - 1]) # ya = y(0), yb = y'(1)

# nastaveni grafu
x_eval = np.linspace(0, 1, 100)

# Prvotní odhar - prevod na Pocatecní ulohu
y_odhad = np.zeros((2, x_eval.size))
y_odhad[0, :] = 1.0  # Konstantní odhad pro polohu
y_odhad[1, :] = 1.0  # Konstantní odhad pro rychlost
 
# vypocty
sol = solve_bvp(soustava, podminky, x_eval, y_odhad)
     
# vykresleni
plt.figure(figsize=(10, 6))
plt.plot(sol.x, sol.y[0], color="red", label=r"Numerické řešení $y(x)$")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(r"Řešení nelineární ODR $(y'')^2 +2y'- y = 5*sin(e^x), y(0) = 1 ,  y'(1) = 1$", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0.7, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper left")
plt.show()