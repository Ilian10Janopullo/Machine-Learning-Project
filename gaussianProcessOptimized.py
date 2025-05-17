import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg'
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

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

# Drop rows where the target variable is NaN
log(f"Initial rows: {len(df)}")
df.dropna(subset=[target], inplace=True)
log(f"Rows after dropping NaNs in target ('{target}'): {len(df)}")

if df.empty:
    log(f"Error: No data remaining after dropping NaNs in target column '{target}'.")
    exit()

y = df[target].values

# Select only numeric features for X
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
if target in numeric_features:
    numeric_features.remove(target)

if not numeric_features:
    log("Error: No numeric features found in the dataset to use for X after excluding the target.")
    log(f"Available columns: {df.columns.tolist()}")
    log(f"Numeric columns found: {df.select_dtypes(include=np.number).columns.tolist()}")
    exit()

X_numeric = df[numeric_features].values
feature_cols = numeric_features # Store feature names for potential later use

log(f"Features selected (numeric only): {len(feature_cols)}, Target: {target}")
if len(feature_cols) < 10: # Log feature names if there are few
    log(f"Feature names: {feature_cols}")

# ----- IMPUTE MISSING VALUES IN FEATURES X -----
log("Imputing missing values in features (X) using mean strategy...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)
log("Missing value imputation for X complete.")

# Check if y still contains NaNs (should not happen if dropna worked, but good for sanity check)
if np.isnan(y).any():
    log("Error: Target variable y still contains NaNs after attempting to drop them. Please check data.")
    exit()


# ----- TRAIN/TEST SPLIT -----
log("Splitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
log(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ----- FEATURE SCALING -----
log("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log("Feature scaling complete.")

# ----- Gaussian Process Kernel Setup -----
log("Defining Gaussian Process kernel...")
# Adjusted bounds based on ConvergenceWarnings and typical practice
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
log(f"Kernel defined: {kernel}")

# ----- Training Gaussian Process -----
log("Training Gaussian Process...")
# Consider alpha for numerical stability if convergence issues persist
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-5)

# Reduce training samples if dataset is very large to manage computation time
MAX_TRAIN_SAMPLES = 5000 # Adjust as needed based on your system's capability
if X_train_scaled.shape[0] > MAX_TRAIN_SAMPLES:
    log(f"Reducing training samples from {X_train_scaled.shape[0]} to {MAX_TRAIN_SAMPLES} for faster training.")
    indices = np.random.choice(X_train_scaled.shape[0], MAX_TRAIN_SAMPLES, replace=False)
    X_train_subset = X_train_scaled[indices]
    y_train_subset = y_train[indices]
    gp_model.fit(X_train_subset, y_train_subset)
else:
    gp_model.fit(X_train_scaled, y_train)

log("Gaussian Process training complete.")
log(f"Trained kernel: {gp_model.kernel_}")

# ----- Evaluate Model -----
log("Evaluating model on test set...")
y_pred_test = gp_model.predict(X_test_scaled)

# Gaussian Process can predict negative values; clip them if inappropriate for the target
# y_pred_test = np.maximum(0, y_pred_test)
# log("Negative predictions set to 0 (if any).")

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nGaussian Process Regressor Performance on Test Set:")
print(f"  RMSE: {rmse_test:.2f} km")
print(f"  MAE:  {mae_test:.2f} km")
print(f"  R²:   {r2_test:.3f}")

# ----- Plotting Predictions vs Actual -----
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)

# Ensure y_test is not empty before calling min/max
if len(y_test) > 0 and len(y_pred_test) > 0:
    # Plot a diagonal line y=x for reference
    # Use the overall min and max of actual and predicted values for the line range
    # to ensure the line covers the entire scatter plot range.
    plot_min = min(min(y_test), min(y_pred_test))
    plot_max = max(max(y_test), max(y_pred_test))
    plt.plot([plot_min, plot_max], [plot_min, plot_max], '--r', linewidth=2)
elif len(y_test) == 0:
    log("Warning: y_test is empty, cannot plot actual vs predicted values.")
else: # y_pred_test might be empty if y_test was not, though unlikely with GPR
    log("Warning: y_pred_test is empty, cannot plot actual vs predicted values accurately.")


plt.title('Gaussian Process: Actual vs. Predicted Electric Range')
plt.xlabel('Actual Electric Range (km)')
plt.ylabel('Predicted Electric Range (km)')
plt.grid(True)
plt.show()

# ----- TOTAL ELAPSED TIME -----
log(f"Total elapsed time: {time.time() - start_time:.1f}s")