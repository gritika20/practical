import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# --- Grid ---
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)
TH, PH = np.meshgrid(theta, phi)
# --- Choose (l, m) values to plot ---
pairs = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]

# --- Function to convert spherical to cartesian ---
def sph_to_cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return x, y, z

for l, m in pairs:
    Y = sph_harm(m, l, PH, TH)
    P = np.abs(Y)**2
    P /= P.max()       # scale to 0–1

    r = 1 + 0.6 * P    # modulation for shape
    X, Yc, Z = sph_to_cart(r, TH, PH)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Yc, Z, facecolors=plt.cm.viridis(P),
                    linewidth=0, antialiased=False)

    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    ax.set_title(f"|Y({l},{m})|²")

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(P)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)

plt.show()
