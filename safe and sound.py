import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# --- Associated Legendre polynomial ---
def assoc_legendre(l, m, x):
    m = abs(m)
    # Base P_m^m
    if m == 0:
        Pmm = np.ones_like(x)
    else:
        Pmm = (-1)**m * np.prod(np.arange(1, 2*m, 2)) * (1 - x**2)**(m/2)
    if l == m:
        return Pmm
    # P_{m+1}^m
    Pm1m = x * (2*m + 1) * Pmm
    if l == m + 1:
        return Pm1m
    # Recurrence for l > m+1
    for ll in range(m + 2, l + 1):
        Plm = ((2*ll - 1)*x*Pm1m - (ll + m - 1)*Pmm) / (ll - m)
        Pmm, Pm1m = Pm1m, Plm
    return Plm

# --- Real Spherical Harmonics ---
def Y_real(l, m, theta, phi):
    P = assoc_legendre(l, abs(m), np.cos(theta))
    norm = np.sqrt((2*l + 1)/(4*np.pi) * factorial(l - abs(m))/factorial(l + abs(m)))
    if m == 0:
        return norm * P
    elif m > 0:
        return np.sqrt(2) * norm * P * np.cos(m * phi)
    else:
        return np.sqrt(2) * norm * P * np.sin(-m * phi)

# --- Spherical to Cartesian ---
def sph_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# --- Grid ---
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
TH, PH = np.meshgrid(theta, phi)

# --- List of (l, m) pairs to visualize ---
pairs = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]

# --- Plot ---
for l, m in pairs:
    Y = Y_real(l, m, TH, PH)
    P = Y**2
    P /= P.max()  # normalize for visualization

    r = 1 + 0.5 * P  # radius modulated by probability density
    X, Yc, Z = sph_to_cart(r, TH, PH)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Yc, Z, facecolors=plt.cm.viridis(P),
                    linewidth=0, antialiased=False)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    ax.set_title(f"Spherical Harmonic |Y_{{{l},{m}}}|²", pad=10)

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(P)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label="|Y|² (normalized)")

plt.show()
print("All selected spherical harmonics plotted successfully!")
