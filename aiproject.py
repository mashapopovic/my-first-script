import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- STEP 1: DATA LOADING ---
file_path = 'ansys_results.csv'
df = pd.read_csv(file_path)

# Assign inputs and output [cite: 22]
X = df[['t', 'Q']]
y = df['Tskin']

# VALIDATION: Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset Size: {len(df)} cases (Training: {len(X_train)}, Testing: {len(X_test)})")

# --- STEP 2: AI MODEL TRAINING ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # Train ONLY on the training set

# --- STEP 3: ERROR ANALYSIS (Deliverable) ---
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
accuracy = r2_score(y_test, y_pred_test)

print("-" * 30)
print(f"MODEL PERFORMANCE (Validation)")
print(f"RMSE: {rmse:.4f} °C")
print(f"R² Accuracy: {accuracy:.4f}")
print("-" * 30)

# --- STEP 4: VISUALIZATION (DESIGN SPACE) [cite: 24, 31] ---
t_range = np.linspace(1, 5, 50)
Q_range = np.linspace(0.1, 2.0, 50)
T_grid = np.zeros((50, 50))

for i, t_val in enumerate(t_range):
    for j, Q_val in enumerate(Q_range):
        T_grid[j, i] = model.predict(np.array([[t_val, Q_val]]))[0]

plt.figure(figsize=(8, 6))
cp = plt.contourf(t_range, Q_range, T_grid, cmap='RdYlBu_r', levels=20)
plt.colorbar(cp, label='Skin Temperature (°C)')
# Draw the safety constraint line at 43.5°C [cite: 26]
cs = plt.contour(t_range, Q_range, T_grid, levels=[43.5], colors='red', linestyles='--')
plt.clabel(cs, inline=True, fontsize=10, fmt='Safety Limit (43.5°C)')
plt.xlabel('Insulation Thickness (t) [mm]')
plt.ylabel('Device Heat Flux (Q) [W]')
plt.title('Design Space: Performance vs. Safety')
plt.savefig('design_space.png')

# --- STEP 5: OPTIMIZATION TASK [cite: 26] ---
best_Q = 0
best_t = 0

for t_val in np.linspace(1, 5, 100):
    for Q_val in np.linspace(2.0, 0.1, 100):
        temp_pred = model.predict(np.array([[t_val, Q_val]]))[0]
        if temp_pred <= 43.5:
            if Q_val > best_Q:
                best_Q = Q_val
                best_t = t_val
            break 

print(f"--- Final Recommendation [cite: 33] ---")
print(f"To maximize performance, we recommend an insulation thickness of {best_t:.2f} mm.")
print(f"This allows for a maximum power output of {best_Q:.2f} Watts while maintaining skin safety.")