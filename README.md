# AI-ML-and-GEN-AI-Track-Project-Template
# 🌊 Rising Water – ML-Powered Flood Prediction System

## 📖 Overview

**Rising Water** is a Machine Learning-based Flood Prediction System designed to provide early flood risk assessment using historical weather and rainfall data. The system compares multiple classification algorithms, selects the best-performing model, and provides real-time flood predictions through an interactive Flask web application.

The project demonstrates how Artificial Intelligence can support disaster management by helping authorities and communities prepare for potential flood events.

---

## 🚨 Problem Statement

Floods are among the most destructive natural disasters, causing loss of life, property damage, and economic disruption every year. Traditional flood forecasting methods often struggle to provide timely and accurate predictions.

This project uses Machine Learning techniques to analyze historical rainfall and weather parameters such as cloud cover, seasonal rainfall, and annual rainfall to predict whether flood conditions are likely to occur.

---

## 🎯 Objectives

- Predict flood occurrence using historical weather data.
- Compare multiple Machine Learning algorithms.
- Select the best-performing model automatically.
- Provide a user-friendly web interface for prediction.
- Support early disaster preparedness and response.

---

## ✨ Features

- 🌧 Predicts flood occurrence using Machine Learning
- 🤖 Compares four ML algorithms:
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbours (KNN)
  - XGBoost
- 📊 Automatically selects the best-performing model
- ⚖ Handles class imbalance using class weighting
- 🌐 Flask-based web application
- 📈 Displays model performance metrics
- 🚀 Fast and easy prediction interface
- ☁ Ready for cloud deployment (Render, Railway, IBM Cloud)

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Model Saving | Joblib |
| Web Framework | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Version Control | Git & GitHub |

---

## 📂 Repository Structure

```
Rising_Water/

├── 1. Brainstorming & Ideation
├── 2. Requirement Analysis
├── 3. Project Design Phase
├── 4. Project Planning Phase
├── 5. Project Development Phase
│   └── Rising-Water/
│       ├── app.py
│       ├── floods.save
│       ├── transform.save
│       ├── templates/
│       ├── static/
│       ├── requirements.txt
│
├── 6. Project Testing
├── 7. Project Documentation
├── 8. Project Demonstration
└── README.md
```

---

## 📊 Dataset

The Machine Learning model is trained using historical weather and rainfall data containing the following attributes:

| Feature | Description |
|---------|-------------|
| Cloud Cover | Percentage of cloud cover |
| Annual Rainfall | Total annual rainfall |
| Jan-Feb Rainfall | Rainfall during Jan-Feb |
| Mar-May Rainfall | Rainfall during Mar-May |
| Jun-Sep Rainfall | Monsoon rainfall |
| Oct-Dec Rainfall | Rainfall during Oct-Dec |
| Flood | Target label (1 = Flood, 0 = No Flood) |

---

## 🧠 Machine Learning Models

The following classification algorithms were evaluated:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbours (KNN)
- XGBoost Classifier

The model with the highest **F1 Score** is selected and saved for prediction.

---

## ⚙ Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Feature Selection
4. Train Multiple Models
5. Evaluate Model Performance
6. Save Best Model
7. Integrate with Flask
8. Predict Flood Risk

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Rising_Water.git
```

### Navigate to Project

```bash
cd Rising_Water
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The model with the best F1 Score is deployed in the application.

---

## 💻 Application Features

- Home Page
- Prediction Form
- Instant Flood Prediction
- Model Performance Dashboard
- Responsive User Interface

---

## 🧪 Testing

The application has been tested using:

- Functional Testing
- Integration Testing
- User Interface Testing
- Performance Testing (Apache JMeter)

---

## 🌍 Deployment

The Flask application can be deployed on:

- Render
- IBM Cloud
- Railway
- Heroku

---

## 📸 Screenshots

Add screenshots of:

- Home Page
- Prediction Form
- Prediction Result
- Performance Dashboard

---

##  Future Enhancements

- Real-time Weather API Integration
- Live Rainfall Monitoring
- SMS & Email Alerts
- Interactive Flood Maps
- Mobile Application
- Deep Learning Models

---

## 👥 Team Members

- Annam Lasya
- Guravaiah Katta
- Batta Kusumamba

---

## 📚 Documentation

This repository contains complete project documentation, including:

- Brainstorming & Ideation
- Requirement Analysis
- Project Design
- Project Planning
- Development
- Testing
- Documentation
- Demonstration

---

##  License

This project is developed for academic and educational purposes.

---

##  Acknowledgements

- Kaggle Dataset
- Scikit-learn
- XGBoost
- Flask
- Python Community
- Open Source Community
