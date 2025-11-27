import matplotlib.pyplot as plt
import math
import numpy as np
import time

print("Solving Schrodinger equation for particle in a box by finite difference method")

# Limits
a = float(input("enter the upper limit: "))
print("the lower limit is 0 by default.")

# Step size
n = int(input("number of values of solutions:"))
x = np.linspace(0, a*10e-10, n)
s = x[1]-x[0]

#potential
V = np.zeros(n)

#kinetic energy matrix
hbar = 1.0545718e-34     # Reduced Planck's constant in J·s
m = 9.10938356e-31       # Mass of electron in kg
alpha = (hbar**2) / (2*m*s**2)

# Construct tridiagonal kinetic energy matrix
T = np.zeros((n-2, n-2))  #to exclude the two boundary points

for i in range(n-2):
    T[i, i] = 2 * alpha
    if i > 0:
        T[i, i-1] = -alpha
    if i < n-3:
        T[i, i+1] = -alpha
V_cut = V[1:-1]              # Remove boundary points
V_matrix = np.diag(V_cut)    # Make it a diagonal matrix

H = T + V_matrix             # Total Hamiltonian = Kinetic + Potential

# Solve for eigenvalues (energies) and eigenvectors (wavefunctions)
eigenvalues, eigenvectors = np.linalg.eigh(H)
eigenval = []
for i in eigenvalues:
    j = i*6.2415e+18
    eigenval.append(j)
print("first few energy values:", eigenval[:10])
# Normalize wavefunctions and add boundary points (0 at ends)
wavefunctions = []

for i in range(n-2):  # Loop through all eigenstates
    psi = eigenvectors[:, i]

    # Flip sign
    if psi[np.argmax(np.abs(psi))] < 0:
        psi *= -1

    # Normalize
    psi = psi / np.sqrt(np.sum(psi**2) * s)

    # Add boundary points
    psi_full = np.zeros(n)
    psi_full[1:-1] = psi
    wavefunctions.append(psi_full)

#plot
num_states_to_plot = int(input("how many do you want plotted?"))  # plot only first 5 states
for i in range(num_states_to_plot):
    plt.plot(x, wavefunctions[i] + eigenvalues[i], label=f'n={i+1}')

plt.title("Wavefunctions and Energy Levels for Particle in a Box")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid(True)
plt.show()
