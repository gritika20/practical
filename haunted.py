import matplotlib.pyplot as plt
import math
import numpy as np
import time


e = 1.60217662e-19       # elementary charge (C)
epsilon0 = 8.854187817e-12 # vacuum permittivity (F/m)
r_min = 1e-11              # Start just above zero to avoid singularity, meters
hbar = 1.0545718e-34     # Reduced Planck's constant in J·s
m = 9.10938356e-31       # Mass of electron in kg

r_max = float(input("Enter the maximum radius in Angstroms (e.g. 200): ")) * 1e-10
n = int(input("Number of grid points (e.g. 3000): "))

print("Schrodinger equation by finite difference")

# Step size
r = np.linspace(r_min, r_max, n)
dr = r[1] - r[0]

#potential
V = - e**2 / (4 * np.pi * epsilon0 * r)

#kinetic energy matrix
alpha = (hbar**2) / (2*m*dr**2)

# Construct tridiagonal kinetic energy matrix
T = np.zeros((n, n))
for i in range(n):
    T[i, i] = 2 * alpha
    if i > 0:
        T[i, i-1] = -alpha
    if i < n-1:
        T[i, i+1] = -alpha

V_matrix = np.diag(V)    #   Make it a diagonal matrix

H = T + V_matrix             # Total Hamiltonian = Kinetic + Potential

# Solve for eigenvalues (energies) and eigenvectors (wavefunctions)
eigenvalues, eigenvectors = np.linalg.eigh(H)
eigenval = eigenvalues/e
print(eigenval)

# Normalize wavefunctions and add boundary points (0 at ends)
wavefunctions = []
for i in range(n):
    psi = eigenvectors[:, i]
    norm = np.sqrt(np.sum(psi**2) * dr)
    psi = psi / norm

    if psi[np.argmax(np.abs(psi))]<0:
        psi*= -1
    wavefunctions.append(psi)
R = [i/r for i in wavefunctions]

#plot

num_states_to_plot = int(input("how many do you want plotted?"))
offset = 5
for i in range(num_states_to_plot):
    plt.plot(r*1e10, (wavefunctions[i])**2, label=f"n={i+1}")

plt.xlim(0, 20) 
plt.title("Radial wavefunctions and Energy Levels for Hydrogen (l=0)")
plt.xlabel("Radius r (Angstroms)")
plt.ylabel("psi**2")
plt.legend()
plt.grid(True)
plt.show()

print("Computed vs Exact Energy levels (eV):")
for i in range(5):
    print(f"n={i+1}: Computed = {eigenval[i]:.4f} eV, Exact = {-13.6/(i+1)**2:.4f} eV")

