import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg'
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- TIMER SETUP -----
start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {msg}")

# ----- LOAD DATA -----
file_path = 'Source/processed_ev_data.csv'
log(f"Loading dataset from '{file_path}'...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    log(f"Error: File '{file_path}' not found.")
    exit()
log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----- PREPARE FEATURES & TARGET -----
target = 'Electric Range (km)'
if target not in df.columns:
    log(f"Error: Target column '{target}' not found in dataset.")
    exit()

y = df[target].values
feature_cols = [col for col in df.columns if col != target]
X = df[feature_cols].values

log(f"Features: {len(feature_cols)}, Target: {target}")

# ----- TRAIN/TEST SPLIT -----
log("Splitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ----- FEATURE SCALING -----
log("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log("Feature scaling complete.")

# ----- FIND OPTIMAL K -----
# log("Finding optimal K for KNN...")
# k_range = range(1, 31) #T Test for k values from 1 to 30
# k_rmses = []
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#
# for k_val in k_range:
#     knn = KNeighborsRegressor(n_neighbors=k_val, n_jobs=-1)
#     mse_scores = cross_val_score(knn, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')
#     rmse_scores = np.sqrt(-mse_scores)
#     k_rmses.append(rmse_scores.mean())
#     log(f"  k={k_val}, Mean CV RMSE: {rmse_scores.mean():.2f}")
#
# optimal_k = k_range[np.argmin(k_rmses)]
# log(f"Optimal K: {optimal_k} with Mean CV RMSE: {min(k_rmses):.2f}")
optimal_k = 2 # Optimal k for the processed dataset found to be 2.

# ----- Plot k vs. RMSE -----
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, k_rmses, marker='o', linestyle='-')
# plt.title('K vs. Cross-Validated RMSE')
# plt.xlabel('Number of Neighbors (K)')
# plt.ylabel('Root Mean Squared Error (RMSE)')
# plt.xticks(k_range)
# plt.grid(True)
# plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K: {optimal_k}')
# plt.legend()
# plt.show()

# ----- TRAIN KNN REGRESSOR -----
log(f"Training KNN Regressor with K={optimal_k}...")
knn_model = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
log("Training complete.")

# ----- EVALUATE MODEL -----
log("Evaluating model on test set...")
y_pred_test = knn_model.predict(X_test_scaled)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nKNN Regressor (k={optimal_k}) Performance on Test Set:")
print(f"  RMSE: {rmse_test:.2f} km")
print(f"  MAE:  {mae_test:.2f} km")
print(f"  R²:   {r2_test:.3f}")

# ----- Actual results vs. Predicted -----
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
plt.title('Actual vs. Predicted Electric Range (KNN)')
plt.xlabel('Actual Electric Range (km)')
plt.ylabel('Predicted Electric Range (km)')
plt.grid(True)
plt.show()

# ----- TOTAL ELAPSED TIME -----
log(f"Total elapsed time: {time.time() - start_time:.1f}s")