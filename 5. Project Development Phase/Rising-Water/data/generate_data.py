"""
Generates a synthetic but realistic historical weather dataset for flood
prediction. Features mirror real meteorological flood-risk drivers:
annual rainfall, seasonal (monsoon) rainfall, cloud visibility, humidity,
river discharge level, and elevation/drainage proxy.

FLOOD is a binary label derived from a physically-motivated rule + noise,
so the relationship is learnable by tree-based and distance-based models
alike.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

# --- Core meteorological features ---
annual_rainfall = np.random.normal(2200, 650, N).clip(400, 4500)          # mm/year
seasonal_rainfall = annual_rainfall * np.random.uniform(0.45, 0.75, N)    # monsoon share, mm
cloud_visibility = np.random.normal(6, 3, N).clip(0.2, 15)                # km (lower = denser storm cloud)
humidity = np.random.normal(75, 12, N).clip(30, 100)                      # %
river_discharge = np.random.normal(1500, 800, N).clip(50, 6000)           # cumecs
drainage_index = np.random.uniform(0, 10, N)                              # 0=poor drainage,10=excellent
temperature = np.random.normal(28, 4, N).clip(15, 42)                     # deg C

# --- Physically-motivated flood risk score ---
risk_score = (
    0.0009 * seasonal_rainfall +
    0.00045 * annual_rainfall +
    -0.06 * cloud_visibility +        # poor visibility (storm clouds) raises risk
    0.018 * humidity +
    0.0006 * river_discharge +
    -0.18 * drainage_index +
    0.01 * temperature +
    np.random.normal(0, 0.35, N)      # noise (kept modest so signal dominates)
)

threshold = np.percentile(risk_score, 62)  # ~38% flood-positive class balance
flood = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "ANNUAL_RAINFALL": annual_rainfall.round(1),
    "SEASONAL_RAINFALL": seasonal_rainfall.round(1),
    "CLOUD_VISIBILITY": cloud_visibility.round(2),
    "HUMIDITY": humidity.round(1),
    "RIVER_DISCHARGE": river_discharge.round(1),
    "DRAINAGE_INDEX": drainage_index.round(2),
    "TEMPERATURE": temperature.round(1),
    "FLOOD": flood
})

df.to_csv("flood_data.csv", index=False)
print("Dataset shape:", df.shape)
print(df["FLOOD"].value_counts(normalize=True))
print(df.head())