import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def calculate_rectangular_velocity(width, height, length, delta_p, viscosity, resolution=50):
    """
    Calculates the 2D velocity profile for laminar flow in a rectangular duct.
    Using the analytical Fourier series solution.
    """
    # Create grid for the cross-section
    y = np.linspace(-width/2, width/2, resolution)
    z = np.linspace(0, height, resolution)
    Y, Z = np.meshgrid(y, z)

    # Initialize velocity field
    u = np.zeros_like(Y)

    # Pre-factor
    prefactor = (4 * (height**2) * delta_p) / (np.pi**3 * viscosity * length)

    # Fourier series with progress bar
    n_values = range(1, 40, 2)

    for n in tqdm(n_values, desc="Computing velocity field"):
        term_n = (1/n**3) * (
            1 - (np.cosh(n * np.pi * Y / height) /
                 np.cosh(n * np.pi * width / (2 * height)))
        ) * np.sin(n * np.pi * Z / height)

        u += term_n

    return Y, Z, u * prefactor

# --- Parameters (Standard Microfluidic Scale) ---
w = 200e-6      # 200 microns width
h = 100e-6      # 100 microns height
L = 0.01        # 1 cm length
dP = 500        # 500 Pa pressure drop
mu = 1e-3       # 1 mPa·s (Viscosity of water/culture media)

# Calculate
Y, Z, U = calculate_rectangular_velocity(w, h, L, dP, mu)

# --- Visualization ---
plt.figure(figsize=(10, 5))
contour = plt.contourf(Y * 1e6, Z * 1e6, U, levels=50, cmap='viridis')
plt.colorbar(contour, label='Velocity (m/s)')
plt.title(f'Laminar Velocity Profile (Rectangular Duct: {int(w*1e6)}x{int(h*1e6)} µm)')
plt.xlabel('Width (µm)')
plt.ylabel('Height (µm)')
plt.show()

# Print Max Velocity
print(f"Maximum Velocity: {np.max(U):.4f} m/s")
