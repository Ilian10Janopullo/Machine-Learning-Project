import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data
df = pd.read_csv('processed_ev_data.csv')

# 2. Define target and features
y = df['Electric Range (km)'].values
X = df.drop(columns=['Electric Range (km)'])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. Build the MLP
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=[
        keras.metrics.MeanAbsoluteError(name='mae'),
        keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)

# 6. Train with early stopping
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=2
)

# 7. Predict & compute metrics
y_pred = model.predict(X_test_scaled).flatten()
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE  (km): {mae:.2f}")
print(f"RMSE (km): {rmse:.2f}")
print(f"R²        : {r2:.3f}")

# 8. Plot Actual vs. Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.xlabel('Actual Range (km)')
plt.ylabel('Predicted Range (km)')
plt.title('MLP: Actual vs. Predicted EV Range')
plt.grid(True)
plt.ylim(bottom=0)
plt.show()

