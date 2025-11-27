import matplotlib.pyplot as plt
import math
import numpy as np
import time

hbar = 1.0545718e-34 # J·s         
k = 100*1.60218e17   # N/m
m = 1.6765e-27       # kg
omega = np.sqrt(k/m)

b = int(input("Enter the value of b(0,10,30): "))*1.60218e32
r_min = 0.0001e-15    # m
r_max = 5e-15     # m
n = 1000

r = np.linspace(r_min, r_max, n)
dr = r[1] - r[0]

V = (0.5 * m*omega**2* r**2) + ((1/3)*b*r**3)

alpha = (hbar ** 2) / (2 * m * dr**2)

T = np.zeros((n, n))
for i in range(n):
    T[i, i] = 2 * alpha
    if i > 0:
        T[i, i - 1] = -alpha
    if i < n - 1:
        T[i, i + 1] = -alpha

H = T + np.diag(V)

eigenvalues, eigenvectors = np.linalg.eigh(H)
eigenvalues_MeV = eigenvalues / 1.60218e-13
print("Eigenvalues (MeV):")
print(eigenvalues_MeV[:10])  # First 10 eigenvalues


# Normalize wavefunctions and add boundary points (0 at ends)
wavefunctions = []
num_states_to_plot = int(input("how many do you want plotted?"))
for i in range(num_states_to_plot):
    psi = eigenvectors[:, i]
    norm = np.sqrt(np.sum(psi**2) * dr)
    psi = psi / norm

    if psi[np.argmax(np.abs(psi))]<0:
        psi*= -1
    wavefunctions.append(psi)
    
num_cols = 2
num_rows = (num_states_to_plot + num_cols - 1) // num_cols  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows))
axes = axes.flatten()

for i in range(num_states_to_plot):
    psi = wavefunctions[i]
    ax = axes[i]
    ax.plot(r * 1e15, psi, label=f"n={i}")
    ax.set_ylabel(f"ψ_{i}(r)")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.legend()
    ax.grid(True)

# Set xlabel only on bottom row
for ax in axes[-num_cols:]:
    ax.set_xlabel("Radius r (fm)")

plt.suptitle("Individual Wavefunctions for Quantum Anharmonic Oscillator")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

