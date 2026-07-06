# res ODR (y'')^3 = e^sin(ln(x)), y(e) = 1, y'(e) = 1

import numpy as np
import matplotlib.pyplot as plt


# Vyjadreni soustavy: y'' = e^sin(ln(x))**(1/3)
def soustava(x, y1, y2):
    y1_der = y2 #prvni derivace
    y2_der = np.exp((np.sin(np.log(x)))) **(1/3) #derivce
    return y1_der, y2_der

# Nastavení okrajových podmínek a simulačního intervalu - 
x_eval = np.linspace(np.e, 5, 1000) # pocatecni podminka, do, pocet bodu
dx = x_eval[1] - x_eval[0]

# Eulerova metoda
y_result = np.zeros(len(x_eval))
dy_result = np.zeros(len(x_eval))

# pocatecni podminky
y_result[0] = 1.0 # y(e) = 1
dy_result[0] = 1.0 # y'(e)= 1


for i in range(1, len(x_eval)):
    x_current = x_eval[i-1]
    y_current = y_result[i-1]
    dy_current = dy_result[i-1]
    
# obe derivace naraz
    y1_der, y2_der = soustava(x_current, y_current, dy_current)
    
# Simultánní posun polohy i rychlosti o krok dx
    y_result[i] = y_current + dx * y1_der
    dy_result[i] = dy_current + dx * y2_der
 
# vykresleni
plt.figure(figsize=(10, 6))
plt.plot(x_eval, y_result, color="green", label=r"Numerické řešení $y(x)$")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title(r"Graf řešení nelineární ODR $(y'')^3 = e^{sin(ln(x))}, y(e) = 1 $ a $y'(e) = 1$", fontsize=12)
plt.xlabel("x")
plt.xlim(2.5, 5.1) # posun obrazku
plt.ylabel("y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper left")
plt.show()