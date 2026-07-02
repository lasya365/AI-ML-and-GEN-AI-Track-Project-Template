# 🌊 Rising Waters — ML-Powered Flood Prediction System

Rising Waters is a machine learning flood early-warning system trained on
historical weather data. It compares four classification algorithms
(Decision Tree, Random Forest, K-Nearest Neighbours, and XGBoost), saves
the best-performing model, and serves predictions through a Flask web app
so disaster management teams and meteorologists can assess flood risk in
real time.

## Problem Statement

Floods are among the most devastating natural disasters, claiming
thousands of lives and displacing millions every year. Conventional
forecasting methods often fall short in predicting floods early enough
for authorities and communities to respond. This project builds a
machine learning-powered flood prediction system trained on historical
weather data — analysing meteorological features such as temperature,
humidity, cloud cover, and seasonal rainfall patterns to predict the
likelihood of a flood event.

## Features

- Trains and compares **Decision Tree, Random Forest, KNN, and XGBoost**
- Automatically selects and saves the best-performing model
- Handles class imbalance (rare flood events) via class weighting
- Flask web app with a prediction form and a model-performance dashboard
- Returns flood probability and a risk level (Low / Moderate / High / Severe)
- JSON API endpoint (`/predict`) for integration with other tools
- Designed for deployment on cloud platforms (e.g. IBM Cloud, Render, Railway)

## Demo Scenarios

1. **Early Flood Warning & Evacuation Planning** — a meteorologist enters
   current rainfall and cloud cover readings for a district; the model
   flags high flood probability, enabling early evacuation advisories.
2. **Disaster Response & Resource Allocation** — a coordinator monitors
   multiple regions during monsoon season, using instant risk
   classifications to prioritise resource deployment.
3. **Model Validation** — analysts compare model accuracy, precision,
   recall, and F1 score on the `/about` page to confirm reliability.

## Project Structure

```
rising_waters/
├── data/
│   ├── generate_data.py       # builds a synthetic dataset (optional/demo)
│   └── flood_data.csv         # historical weather dataset (real or synthetic)
├── model/
│   ├── train.py               # trains & compares all 4 classifiers
│   ├── best_model.pkl         # saved best-performing model (generated)
│   ├── scaler.pkl             # saved StandardScaler (generated)
│   └── results.json           # accuracy/precision/recall/F1 per model (generated)
├── templates/
│   ├── base.html
│   ├── index.html             # prediction form + result
│   └── about.html             # model performance comparison page
├── app.py                     # Flask application
├── requirements.txt
└── README.md
```

## Dataset

The model expects a CSV with these columns:

| Column | Description |
|---|---|
| `Temp` | Temperature (°C) |
| `Humidity` | Humidity (%) |
| `Cloud Cover` | Cloud cover (%) |
| `ANNUAL` | Annual rainfall (mm) |
| `Jan-Feb` | Jan–Feb rainfall (mm) |
| `Mar-May` | Mar–May rainfall (mm) |
| `Jun-Sep` | Jun–Sep (monsoon) rainfall (mm) |
| `Oct-Dec` | Oct–Dec rainfall (mm) |
| `avgjune` | Average June rainfall (mm) |
| `sub` | Sub-regional rainfall (mm) |
| `flood` | Target label (1 = flood, 0 = no flood) |

> Note: with a small/imbalanced dataset, the best model is selected by
> **F1 score**, not raw accuracy — a model that always predicts "no
> flood" can score high accuracy while being operationally useless.

## Setup & Usage

### 1. Clone and set up environment
```bash
git clone https://github.com/<your-username>/rising_waters.git
cd rising_waters
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your dataset
Place your dataset as `data/flood_data.csv` with the columns listed above.
(Or run `python data/generate_data.py` from inside `data/` to create a
synthetic demo dataset instead.)

### 3. Train and compare models
```bash
cd model
python train.py
cd ..
```
This prints accuracy/precision/recall/F1 for all four models and saves
the best one as `model/best_model.pkl`.

### 4. Run the web app
```bash
python app.py
```
Visit `http://localhost:5000` to use the prediction form, or
`http://localhost:5000/about` for the model comparison dashboard.

### 5. API usage
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Temp":30,"Humidity":78,"Cloud Cover":42,"ANNUAL":3300,"Jan-Feb":20,"Mar-May":150,"Jun-Sep":2200,"Oct-Dec":600,"avgjune":250,"sub":500}'
```

## Deployment

This is a standard Flask app. For production:
```bash
gunicorn app:app
```
Deployable on IBM Cloud Code Engine, Render, Railway, Heroku, or any
platform supporting Python buildpacks/containers. Set the `PORT`
environment variable if your platform requires a specific binding.

## Tech Stack

- **Python**, **scikit-learn**, **XGBoost**, **pandas**, **NumPy**
- **Flask** for the web application
- **joblib** for model persistence

## License

This project is provided as-is for educational and prototype purposes.

## Acknowledgements

Built as a disaster-management decision-support prototype demonstrating
how classical ML models can power accessible, real-time flood
early-warning tools.
