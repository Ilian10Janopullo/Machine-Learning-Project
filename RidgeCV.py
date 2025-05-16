import pandas as pd
import numpy as np
import time
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt    

# ====== TIMER SETUP ======
start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {msg}")

# ====== Load Data ======
file_path = 'Source/processed_ev_data.csv'
log(f"Loading dataset from '{file_path}'...")
df = pd.read_csv(file_path)
log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ====== Prepare Features & Target ======
y = df['Electric Range (km)'].values
feature_cols = [col for col in df.columns if col != 'Electric Range (km)']
X = df[feature_cols].values

# ====== Cross-Validation & Model Setup ======
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 13)
log("Initializing RidgeCV...")
ridge_cv = RidgeCV(alphas=alphas, cv=kf)

# ====== Alpha Selection ======
log("Selecting optimal alpha via full-data fit...")
ridge_cv.fit(X, y)
log(f"Selected alpha: {ridge_cv.alpha_:.3e}")

# ====== Cross-Validated Evaluation ======
log("Performing cross-validated predictions...")
cv_preds = cross_val_predict(ridge_cv, X, y, cv=kf, n_jobs=-1)
log("Prediction complete.")

# ====== Metrics ======
rmse = np.sqrt(mean_squared_error(y, cv_preds))
mae = mean_absolute_error(y, cv_preds)
r2 = r2_score(y, cv_preds)
print(f"Improved Simple Model (Ridge) Performance:\n  RMSE: {rmse:.2f} km\n  MAE: {mae:.2f} km\n  R²: {r2:.3f}")

# ====== Final Training & Saving ======
log(f"Model saved; total elapsed time: {time.time() - start_time:.1f}s")

log("Plotting Actual vs Predicted EV Range...")
min_val = min(y.min(), cv_preds.min())
max_val = max(y.max(), cv_preds.max())

plt.figure()
plt.scatter(y, cv_preds)
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("Actual Electric Range (km)")
plt.ylabel("Predicted Electric Range (km)")
plt.title("RidgeCV: Actual vs Predicted EV Range")
plt.tight_layout()
plt.show()