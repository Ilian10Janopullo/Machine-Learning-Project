import matplotlib
matplotlib.use('TkAgg')
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
file_path = 'Source/Electric_Vehicle_Population_Data.csv'
log(f"Loading dataset from '{file_path}'...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    log(f"Error: File '{file_path}' not found.")
    exit()
log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----- PREPARE FEATURES & TARGET -----
target = 'Electric Range'
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
log(f"Features selected: {len(numeric_features)}, Target: {target}")
if len(numeric_features) < 10:
    log(f"Feature name: {numeric_features}")

# ----- Imputing missing values in X -----
log("Imputing missing values in features (X) using mean strategy ...")
imputer = SimpleImputer(strategy='mean')
X_numeric_imputed = imputer.fit_transform(X_numeric)
log("Imputation complete.")

if np.isnan(y).any():
    log("Error: Target variable 'y' contains NaN values after imputation.")
    exit()

# ----- TRAIN/TEST SPLIT -----
log("Splitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X_numeric_imputed, y, test_size=0.2, random_state=42)
log(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ----- FEATURE SCALING -----
log("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log("Feature scaling complete.")

# ----- Gaussian Process Kernel Setup -----
log("Defining Gaussian Process kernel...")
# Adjusted bounds based on ConvergenceWarnings
kernel = C(1.0, (1e-3, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e2))
log(f"Kernel defined: {kernel}")

# ----- Training Gaussian Process -----
log("Training Gaussian Process...")
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-5)

# Reduce training samples for faster training
MAX_TRAIN_SAMPLES = 5000
if X_train_scaled.shape[0] > MAX_TRAIN_SAMPLES:
    log(f"Reducing training samples from {X_train_scaled.shape[0]} to {MAX_TRAIN_SAMPLES} for faster training.")
    indices = np.random.choice(X_train_scaled.shape[0], MAX_TRAIN_SAMPLES, replace=False)
    X_train_scaled = X_train_scaled[indices]
    y_train = y_train[indices]

gp_model.fit(X_train_scaled, y_train)
log("Gaussian Process training complete.")
log(f"Trained kernel: {gp_model.kernel_}")

# ----- Evaluate Model -----
log("Evaluating model on test set...")
y_pred_test = gp_model.predict(X_test_scaled)

#Turning negative predictions to 0
y_pred_test = np.maximum(0, y_pred_test)
log("Negative predictions set to 0.")

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Gaussian Process Performance on Test Set:")
print(f"  RMSE: {rmse_test:.2f} km")
print(f"  MAE: {mae_test:.2f} km")
print(f"  R²: {r2_test:.3f}")

# ----- Plotting Predictions vs Actual -----
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
if len(y_test) > 0:
    # Ensure y_pred_test is not empty if y_test is not, before calculating min/max
    if len(y_pred_test) > 0:
        min_val = min(min(y_test), min(y_pred_test))
        max_val = max(max(y_test), max(y_pred_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    else:
        log("Warning: y_pred_test is empty, cannot plot min/max line accurately.")
else:
    log("Warning: y_test is empty, cannot plot min/max line.")
plt.title('Gaussian Process: Actual vs Predicted')
plt.xlabel('Actual Electric Range (km)')
plt.ylabel('Predicted Electric Range (km)')
plt.grid(True)
plt.show()

# ----- Total Elapsed Time -----
log(f"Total elapsed time: {time.time() - start_time:.1f}s")