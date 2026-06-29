# res ODR y*y' = e^sin(x)
# Vzhledem k extremni slozitosti integrace pravé strany volíme robustní numerické řešení.

import numpy as np
import matplotlib.pyplot as plt

# Vyjadreni derivace: y' = e^sin(x) / y
def derivace(x, y):
    return np.exp(np.sin(x)) / y

# Nastavení okrajových podmínek a simulačního intervalu - 
x_eval = np.linspace(1, 10, 1000) # pocatecni podminka, do, pocet bodu
dx = x_eval[1] - x_eval[0]

# Eulerova metoda
y_result = np.zeros(len(x_eval))
y_result[0] = 1 #☻ pocatecni podminka y(1) = 1

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
plt.title(r"Řešení nelineární ODR $y \cdot y' = e^{\sin(x)} $ pro $y(1) = 1$", fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper right")
plt.show()

