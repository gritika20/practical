import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
e = 1.60217662e-19
eps0 = 8.854187817e-12
hbar = 1.0545718e-34
m = 9.10938356e-31

# --- Inputs ---
r_max = float(input("Max radius (Å): ")) * 1e-10
n = int(input("Grid points: "))

# --- Grid ---
r = np.linspace(1e-11, r_max, n)
dr = r[1] - r[0]

# --- Potential ---
V = - e**2 / (4*np.pi*eps0*r)

# --- Kinetic (tridiagonal) ---
alpha = hbar**2 / (2*m*dr**2)
diag = 2*alpha + V
off  = -alpha * np.ones(n-1)

# Construct Hamiltonian using vectorized diagonals
H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

# --- Solve eigenproblem ---
E, psi = np.linalg.eigh(H)
E_ev = E / e

# --- Normalize wavefunctions ---
psi = psi / np.sqrt(np.sum(psi**2, axis=0) * dr)

# --- Plot ---
num = int(input("How many states to plot? "))
for i in range(num):
    plt.plot(r*1e10, psi[:, i]**2, label=f"n={i+1}")

plt.xlim(0, 20)
plt.title("Hydrogen Radial Probability (ℓ=0)")
plt.xlabel("r (Å)")
plt.ylabel("|ψ|²")
plt.legend()
plt.grid()
plt.show()

# --- Compare energies ---
print("\nComputed vs Exact (eV):")
for i in range(5):
    print(f"n={i+1}: {E_ev[i]:.4f}  vs  {-13.6/(i+1)**2:.4f}")
