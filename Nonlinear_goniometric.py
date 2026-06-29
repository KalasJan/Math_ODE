# res ODR y*y' = sin(cos(x))
# Vzhledem k extremni slozitosti integrace pravé strany volíme robustní numerické řešení.

import numpy as np
import matplotlib.pyplot as plt

# Vyjadreni derivace: y' = sin(cos(x)) / y
def derivace(x, y):
    return np.sin(np.cos(x)) / y

# Nastavení okrajových podmínek a simulačního intervalu - 
x_eval = np.linspace(2, 10, 1000) # pocatecni podminka, do, pocet bodu
dx = x_eval[1] - x_eval[0]

# Eulerova metoda
y_result = np.zeros(len(x_eval))
y_result[0] = 3 #☻ pocatecni podminka y(2) = 3

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
plt.title(r"Řešení nelineární ODR $y \cdot y' = \sin(\cos(x))$ pro $y(0) = 2$", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper right")
plt.show()

