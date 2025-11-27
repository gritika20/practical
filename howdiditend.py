import matplotlib.pyplot as plt
import math
import numpy as np
import time
# Use natural units: ħ = c = 1
hbar_c = 197.       # MeV·fm          # unitless
k = 100         # MeV/fm²
m = 940         # MeV/c²

r_min = 0    # fm
r_max = 5     # fm
n = 1000

r = np.linspace(r_min, r_max, n)
dr = r[1] - r[0]

V = 0.5 * k * r**2

alpha = hbar_c**2 / (2 * m * dr**2)

T = np.zeros((n, n))
for i in range(n):
    T[i, i] = 2 * alpha
    if i > 0:
        T[i, i - 1] = -alpha
    if i < n - 1:
        T[i, i + 1] = -alpha

H = T + np.diag(V)

eigenvalues, eigenvectors = np.linalg.eigh(H)
print("Eigenvalues (MeV):")
print(eigenvalues[:10])  # First 10 eigenvalues


# Normalize wavefunctions and add boundary points (0 at ends)
wavefunctions = []
num_states_to_plot = int(input("how many do you want plotted?"))  # plot only first 5 states
for i in range(num_states_to_plot):
    psi = eigenvectors[:, i]
    norm = np.sqrt(np.sum(psi**2) * dr)
    psi = psi / norm

    if psi[np.argmax(np.abs(psi))]<0:
        psi*= -1
    wavefunctions.append(psi)
 
#plot
offset = 0
for i in range(num_states_to_plot):
    psi = wavefunctions[i]
    plt.plot(r, psi + i * offset, label=f"n={i}")
    plt.axhline(i * offset, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
#plt.xlim(0, 0.3)
plt.title("Wavefunctions and Energy Levels for Quantum Harmonic Oscillator")
plt.xlabel("Radius r (fm)")
plt.ylabel("Wavefunction + Energy (eV)")
plt.legend()
#plt.grid(True)
plt.show()

fig, axes = plt.subplots(num_states_to_plot,1, figsize=(8, 3*num_states_to_plot), sharex=True)
