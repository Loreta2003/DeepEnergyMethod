import numpy as np
import matplotlib.pyplot as plt


def u(x, F, EI, L):
    u = -pow(F/EI, 2)*(x**3)*(((x**2)/40) - (L*x)/8 + ((L**2)/6))
    return u 

def w(x, F, EI, L):
    return -(F/EI)*(x**2)*((x/6) - (L/2))  
    
F = 1.0
# Beam properties
L = 1            # Length [m]
W = 0.02           # Width [m]
H = 0.02           # Height [m]
E = 50e6           # Young's modulus [Pa]

# Cross-section parameters (assuming rectangular cross-section)
A = W * H
I = H ** 3 * W / 12

x = np.linspace(0, L, 500)

u_values = u(x, F, E*I, L)
w_values = w(x, F, E*I, L)

x_deformed = x + u_values
y_deformed = w_values

plt.figure(figsize=(12, 6))

plt.plot(x, np.zeros_like(x), label='Original Beam', linestyle='--')
plt.plot(x_deformed, y_deformed, label='Deformed Beam')

plt.xlabel('Horizontal Position (x)')
plt.ylabel('Vertical Position (y)')
plt.title('Deformed Shape of the Beam')
plt.legend()
plt.grid(True)

plt.show()
