import numpy as np
import matplotlib.pyplot as plt

# --- Natural units ---
hbar_c = 197     # MeV·fm
k = 100          # MeV/fm²
m = 940          # MeV/c²

# --- Grid ---
r = np.linspace(0, 5, 1000)
dr = r[1] - r[0]

# --- Potential ---
V = 0.5 * k * r**2

# --- Kinetic (tridiagonal) ---
alpha = hbar_c**2 / (2*m*dr**2)
diag = 2*alpha + V
off  = -alpha * np.ones(len(r)-1)
H = np.diag(diag) + np.diag(off,1) + np.diag(off,-1)

# --- Solve ---
E, psi = np.linalg.eigh(H)
print("First 10 Eigenvalues (MeV):")
print(E[:10])

# --- Normalize ---
psi = psi / np.sqrt(np.sum(psi**2, axis=0) * dr)

# --- Plot ---
num = int(input("How many states to plot? "))
scale = 0.2
for i in range(num):
    wf = psi[:, i]
    if wf[np.argmax(np.abs(wf))] < 0:
        wf *= -1
    plt.plot(r, scale*(wf + E[i]), label=f"n={i}")

plt.title("Quantum Harmonic Oscillator – Wavefunctions")
plt.xlabel("r (fm)")
plt.ylabel("ψ(r) + E")
plt.legend()
plt.show()
