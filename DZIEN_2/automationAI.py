# 1. Importy i generowanie sztucznych danych
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import matplotlib.pyplot as plt
import datetime

# Symulacja danych: 500 punktów co minutę
np.random.seed(42)
time = pd.date_range("2025-07-22 08:00:00", periods=500, freq="T")
cpu_load = np.sin(np.linspace(0, 20, 500)) * 10 + 50 + np.random.normal(0, 2, 500)
df = pd.DataFrame({'timestamp': time, 'cpu_load': cpu_load})

# 2. Wykrywanie anomalii (Isolation Forest)
iso = IsolationForest(contamination=0.03)
df['anomaly'] = iso.fit_predict(df[['cpu_load']])
anomalies = df[df['anomaly'] == -1]

# 3. Predykcja przyszłego obciążenia (Prophet)
df_prophet = df[['timestamp', 'cpu_load']].rename(columns={'timestamp':'ds', 'cpu_load':'y'})
model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=60, freq='T')
forecast = model.predict(future)

# 4. Symulacja decyzji: czy skalować?
# Załóżmy: jeśli przewidywany CPU > 60 przez kolejne 10 minut → skaluj!
window = forecast[['ds', 'yhat']].tail(15)
should_scale = (window['yhat'] > 60).sum() >= 10

# 5. Komunikat diagnostyczny (symulowany ChatOps)
if should_scale:
    msg = f"[{datetime.datetime.now()}] [ChatOps BOT]: Skalowanie! Prognozowane obciążenie CPU powyżej 60%."
else:
    msg = f"[{datetime.datetime.now()}] [ChatOps BOT]: System OK. Skalowanie nie jest wymagane."
print(msg)

# 6. Wizualizacja
plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['cpu_load'], label='CPU Load')
plt.scatter(anomalies['timestamp'], anomalies['cpu_load'], color='red', label='Anomaly')
plt.plot(forecast['ds'], forecast['yhat'], color='green', label='Forecast')
plt.legend()
plt.xlabel("Czas")
plt.ylabel("Obciążenie CPU (%)")
plt.title("Wykrywanie anomalii i prognozowanie obciążenia CPU")
plt.tight_layout()
plt.show()
