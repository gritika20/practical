import numpy as np
import matplotlib.pyplot as plt

print("Particle in a Box – Finite Difference Solver")

# --- Inputs ---
a = float(input("Enter the box length (upper limit): "))
n = int(input("Number of grid points: "))

# --- Grid ---
x = np.linspace(0, a*1e-9, n)
dx = x[1] - x[0]

# --- Potential (zero inside box) ---
V = np.zeros(n-2)

# --- Constants ---
hbar = 1.0545718e-34
m = 9.10938356e-31
alpha = hbar**2 / (2*m*dx**2)

# --- Hamiltonian (tridiagonal) ---
diag = 2*alpha + V
off  = -alpha * np.ones(n-3)
H = np.diag(diag) + np.diag(off,1) + np.diag(off,-1)

# --- Solve ---
E, psi = np.linalg.eigh(H)
E_ev = E * 6.2415e18   # convert J → eV

print("First few energies (eV):", E_ev[:5])

# --- Normalize & add boundary zeros ---
psi_full = np.zeros((n, n-2))
psi = psi / np.sqrt(np.sum(psi**2, axis=0) * dx)
psi_full[1:-1, :] = psi

# --- Plot ---
num = int(input("How many states to plot? "))
for i in range(num):
    plt.plot(x, psi_full[:, i] + E[i], label=f"n={i+1}")

plt.title("Particle in a Box – Wavefunctions and Energy Levels")
plt.xlabel("x (m)")
plt.ylabel("ψ(x)")
plt.grid()
plt.legend()
plt.show()
