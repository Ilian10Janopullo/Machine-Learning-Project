import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# 1. Load and sanitize column names
df = pd.read_csv('processed_ev_data.csv')
clean = {c: re.sub(r'[^0-9A-Za-z_]+', '_', c) for c in df.columns}
df.rename(columns=clean, inplace=True)

# 2. Identify sanitized target name
orig = 'Electric Range (km)'
target = clean[orig]               

# 3. Define target & features
y = df[target]
X = df.drop(columns=[target])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Instantiate the LGBMRegressor
model = LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=32,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

# 6. Fit with callbacks logging both RMSE & MAE
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse','mae'],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# 7. Predict on the test set
y_pred = model.predict(X_test)

# 8. Compute metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE  (km): {mae:.2f}")
print(f"RMSE (km): {rmse:.2f}")
print(f"R²        : {r2:.3f}")

# 9. Plot Actual vs. Predicted
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, alpha=0.7, color='orange')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'b--', lw=2
)
plt.xlabel('Actual Range (km)')
plt.ylabel('Predicted Range (km)')
plt.title('LightGBM: Actual vs. Predicted EV Range')
plt.grid(True)
plt.show()
