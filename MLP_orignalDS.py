import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the data
df = pd.read_csv('Electric_Vehicle_Population_Data.csv')
# 2. Drop the VIN column (no predictive value)
df.drop(columns=['VIN (1-10)'], inplace=True)
# 3. **Drop rows where the target is missing**
df = df.dropna(subset=['Electric Range'])

# 4. Define target and raw features
y = df['Electric Range'].values
X_raw = df.drop(columns=['Electric Range'])

# 5. One‐hot encode all object‐dtype columns
X = pd.get_dummies(X_raw, drop_first=True)

# 6. Impute any remaining missing feature values with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# 7. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 9. Instantiate the MLPRegressor
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=20,
    verbose=True,
    random_state=42
)

# 10. Train the model
mlp.fit(X_train_scaled, y_train)

# 11. Make predictions
y_pred = mlp.predict(X_test_scaled)

# 12. Compute metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE  (km): {mae:.2f}")
print(f"RMSE (km): {rmse:.2f}")
print(f"R²       : {r2:.3f}")

# 13. Plot Actual vs. Predicted Range
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.xlabel('Actual Range (km)')
plt.ylabel('Predicted Range (km)')
plt.title('MLPRegressor: Actual vs. Predicted EV Range')
plt.grid(True)
plt.ylim(bottom=0)   
plt.show()
