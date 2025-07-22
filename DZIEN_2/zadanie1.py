import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generowanie danych: y = 3x + szum
np.random.seed(42)
X = np.linspace(-2, 2, 200).reshape(-1, 1)
noise = np.random.normal(0, 0.5, X.shape[0])
y = 3 * X.squeeze() + noise

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Budowa i trening modelu (Keras/TensorFlow)
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(1,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)  # wyjście 1D
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

# 3. Predykcja na danych testowych
y_pred = model.predict(X_test).squeeze()

# 4. Wizualizacja wyników
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, label='Rzeczywiste', color='blue')
plt.scatter(X_test, y_pred, label='Predykcja (sieć NN)', color='red', alpha=0.6)
plt.title("Predykcja sieci neuronowej vs rzeczywiste dane")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 5. Prezentacja skuteczności modelu
mse = mean_squared_error(y_test, y_pred)
print(f"Średni błąd kwadratowy (MSE): {mse:.3f}")

# Co można poprawić?
print("""
Co można poprawić?
- Zwiększyć liczbę neuronów lub warstw (bardziej złożona sieć)
- Trenować dłużej (więcej epok)
- Użyć innej funkcji aktywacji lub optymalizatora
- Zwiększyć zbiór treningowy lub zmienić rozkład szumu
""")
