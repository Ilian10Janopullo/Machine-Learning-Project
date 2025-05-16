import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt


# ====== TIMER SETUP ======
start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {msg}")

# ====== Load Data ======
log("Loading dataset from 'Source/processed_ev_data.csv'...")
df = pd.read_csv('Source/processed_ev_data.csv')
log(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# ====== Prepare Features & Target ======
X = df.drop(columns=['Electric Range (km)']).values
y = df['Electric Range (km)'].values

# ====== Train/Test Split ======
log("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
log(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ====== Feature Scaling ======
scaler = StandardScaler()
log("Fitting scaler on training data...")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====== Build Model ======
input_dim = X_train.shape[1]
log(f"Building NN model with input dim = {input_dim}...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# ====== Training ======
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
log("Starting training with early stopping on validation loss...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
log("Training complete.")

# ====== Evaluation ======
log("Evaluating model on test set...")
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Deep Model Performance on Test Set:\n  RMSE: {rmse:.2f} km\n  MAE: {mae:.2f} km\n  R²: {r2:.3f}")

# ======= Total Elapsed Time ========
log(f"Total elapsed time: {time.time() - start_time:.1f}s")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("Actual Electric Range (km)")
plt.ylabel("Predicted Electric Range (km)")
plt.title("Deep Model: Actual vs Predicted EV Range")
plt.tight_layout()
plt.show()
