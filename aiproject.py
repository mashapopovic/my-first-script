import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- STEP 1: DATA LOADING ---
# Load data from Ansys Workbench parameter CSV file
# Replace 'ansys_results.csv' with your actual file path
file_path = 'ansys_results.csv'
df = pd.read_csv(file_path)

# Preview the data
print("Loaded data preview:")
print(df.head())

# Assuming the CSV has columns: 'Insulation_Thickness', 'Heat_Flux', 'Skin_Temperature'
# Adjust column names if different
X = df[['Insulation_Thickness', 'Heat_Flux']]
y = df['Skin_Temperature']

print("Data successfully loaded from CSV!")

# --- STEP 2: AI MODEL TRAINING --- [cite: 22]
X = df[['t', 'Q']]
y = df['T_skin']

# Using a Random Forest Regressor as the Surrogate Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# --- STEP 3: VISUALIZATION (DESIGN SPACE) --- [cite: 24, 31]
insulation_thickness_range = np.linspace(1, 5, 50)
heat_flux_range = np.linspace(0.1, 2.0, 50)
T_grid = np.zeros((50, 50))

for i, t_val in enumerate(insulation_thickness_range):
    for j, Q_val in enumerate(heat_flux_range):
        T_grid[j, i] = model.predict(np.array([[t_val, Q_val]]))[0]

plt.figure(figsize=(8, 6))
cp = plt.contourf(insulation_thickness_range, heat_flux_range, T_grid, cmap='RdYlBu_r', levels=20)
plt.colorbar(cp, label='Skin Temperature (°C)')
# Draw the safety constraint line at 43.5°C [cite: 26]
cs = plt.contour(insulation_thickness_range, heat_flux_range, T_grid, levels=[43.5], colors='red', linestyles='--')
plt.clabel(cs, inline=True, fontsize=10, fmt='Safety Limit (43.5°C)')
plt.xlabel('Insulation Thickness (t) [mm]')
plt.ylabel('Device Heat Flux (Q) [W]')
plt.title('Design Space: Performance vs. Safety')
plt.savefig('design_space.png')
print("Plot saved as design_space.png")

# --- STEP 4: OPTIMIZATION TASK --- [cite: 26]
# Finding the max Q for each thickness that stays under 43.5°C
best_Q = 0
best_t = 0

for t_val in np.linspace(1, 5, 100):
    for Q_val in np.linspace(2.0, 0.1, 100): # Start from max power
        temp_pred = model.predict(np.array([[t_val, Q_val]]))[0]
        if temp_pred <= 43.5:
            if Q_val > best_Q:
                best_Q = Q_val
                best_t = t_val
            break # Found the max Q for this thickness

print(f"--- Final Recommendation ---")  # [cite: 33]
print(f"Recommended Insulation Thickness: {best_t:.2f} mm")
print(f"Maximum Allowable Power Output: {best_Q:.2f} Watts")