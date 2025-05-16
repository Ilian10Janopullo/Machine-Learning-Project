import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ====== Load Data & Timer ======
start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {msg}")

file_path = 'Source/processed_ev_data.csv'
log(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path)
log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

y = df['Electric Range (km)'].values

# ====== Baseline Linear Regression ======
baseline_features = ['Model Year', 'Base MSRP', 'Vehicle Age', 'Is Luxury', 'Is Full Electric']
log("Training baseline LinearRegression...")
X_baseline = df[baseline_features].values
baseline_model = LinearRegression()
baseline_pred = baseline_model.fit(X_baseline, y).predict(X_baseline)
log("Baseline training complete.")

df['residual'] = y - baseline_pred

# ====== Manual Cross-Validation for Residual Model ======
residual_features = [c for c in df.columns if c not in ['Electric Range (km)', 'residual']]
X_residual = df[residual_features].values
y_residual = df['residual'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
residual_pred_cv = np.zeros_like(y_residual)

log("Starting manual CV for residual model (5 folds)...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_residual), 1):
    log(f"Fold {fold}: training on {len(train_idx)} samples...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_residual[train_idx], y_residual[train_idx])
    log(f"Fold {fold}: predicting {len(test_idx)} samples...")
    residual_pred_cv[test_idx] = model.predict(X_residual[test_idx])
    log(f"Fold {fold} complete.")

# ====== Hybrid Predictions & Evaluation ======
hybrid_pred = baseline_pred + residual_pred_cv
log("Evaluating hybrid model...")
rmse = np.sqrt(mean_squared_error(y, hybrid_pred))
mae = mean_absolute_error(y, hybrid_pred)
r2 = r2_score(y, hybrid_pred)
print(f"Hybrid Model Performance:\n  RMSE: {rmse:.2f} km\n  MAE: {mae:.2f} km\n  R²: {r2:.3f}")

# ======= Total Elapsed Time ========
log(f"Total elapsed time: {time.time() - start_time:.1f}s")

min_val = min(y.min(), hybrid_pred.min())
max_val = max(y.max(), hybrid_pred.max())

plt.figure()
plt.scatter(y, hybrid_pred)
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("Actual Electric Range (km)")
plt.ylabel("Predicted Electric Range (km)")
plt.title("Actual vs Predicted EV Range")
plt.show()
