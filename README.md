# 🌟 Acne and Pimple Prediction & Clinical Intelligence Platform

An end-to-end **Machine Learning & Dermatological Analytics Platform** that predicts acne likelihood, classifies severity, clusters skin health archetypes, mines behavioral association rules, and provides personalized skincare routines through an interactive modern web dashboard.

🔗 **Google Survey Form:** [Access Research Survey](https://docs.google.com/forms/d/e/1FAIpQLSfjeGZ3jZvv9TccYCv8z8vkDUB4ZrOzeWbiL35kd5cfYivKOQ/viewform?usp=dialog)

---

## 🚀 Key Features

### 1. 📋 Interactive Skin Assessment Quiz
- **15 Clinical Questions**: Directly aligned with the research survey covering Demographics, Skin Type, UV Exposure, Sunscreen, Hygiene, Water Intake, Diet, Stress, Genetics, Face Touching, and Sleep.
- **Multi-Step Stepper**: Smooth animated sections with instant validation.

### 2. 🩺 Comprehensive AI Diagnostic Report
- **Acne Likelihood Index (0–100%)**: Calibrated probability scoring powered by Random Forest and Gradient Boosting models.
- **Severity Prediction**: Multiclass classification (*Clear / Mild / Moderate / Severe*).
- **Skin Cluster Archetype**: Unsupervised K-Means persona assignment (*e.g., "Stress & Diet Vulnerable", "Protective Routine Maintainer", "Barrier & UV Challenged"*).
- **Multi-Dimensional Risk Breakdown**: Visual meters across 5 lifestyle & biological dimensions (Diet, Hydration, Stress, Skincare Defense, Sleep).
- **Mined Association Rules Matcher**: Identifies specific behavioral trigger combinations mined via the Apriori algorithm.
- **Personalized 24-Hour Regimen**: Customized Morning (AM) protocol, Night (PM) active treatments, and lifestyle interventions.
- **Print / PDF Export**: One-click printable medical report layout.

### 3. 📊 ML Research & Analytics Dashboard
- **Algorithm Benchmark Suite**: Side-by-side performance metrics across 7 models.
- **Feature Importance Ranker**: Identifies primary predictive drivers (*Skin Type, Stress, Age, Face Touching, Sleep, etc.*).
- **2D PCA Cluster Map**: Interactive scatter visualization of survey respondents across 3 principal clusters.
- **Apriori Rules Explorer**: Searchable table displaying Antecedents, Consequents, Support, Confidence, and Lift.

### 4. 🎛️ Real-Time "What-If" Lifestyle Simulator
- Interactive sliders (Water, Sugar, Sleep, Stress, Sunscreen, Face Touching) allow users to simulate habit modifications and observe projected risk reductions in real time.

### 5. 🗄️ Survey & Data Hub
- Schema explorer and direct download links for the dataset and precomputed ML insight artifacts.

---

## 🤖 Machine Learning Model Benchmarks

| Model | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **81.33%** | **88.24%** | 84.6% | 92.2% |
| **Random Forest** | **80.67%** | **88.40%** | 83.2% | 94.3% |
| **SVM (RBF Kernel)** | 78.67% | 87.55% | 81.9% | 94.0% |
| **Naive Bayes** | 77.33% | 86.35% | 81.6% | 91.8% |
| **Decision Tree** | 76.00% | 84.55% | 82.8% | 86.4% |
| **Logistic Regression** | 75.00% | 84.79% | 80.4% | 89.6% |
| **KNN (K=5)** | 73.67% | 83.84% | 80.0% | 88.1% |

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, Vanilla CSS (Glassmorphism & Dark Mode), Modern JavaScript (ES6+), [Chart.js](https://www.chartjs.org/), FontAwesome Icons
- **Backend / API**: Python 3, Flask, Gunicorn
- **Machine Learning & Analytics**: Scikit-learn, Pandas, NumPy, MLxtend (Apriori), Matplotlib, Seaborn
- **Deployment Ready**: Configured for **Render** (`Procfile`, `requirements.txt`) and **Vercel** (`vercel.json`)

---

## 📁 Project Structure

```plaintext
├── Acne_dataset.csv         # Research dataset containing survey responses
├── app.py                   # Flask server with REST API (/api/predict, /api/insights)
├── train_and_export.py      # Script to train ML models & export ml_insights.json
├── main.py                  # CLI exploratory analysis & visualization script
├── ml_insights.json         # Precomputed model weights, cluster centroids & rules
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment configuration for Render / Heroku
├── vercel.json              # Serverless configuration for Vercel
├── static/
│   ├── index.html           # Main frontend interface & dashboard
│   ├── style.css            # Dark glassmorphic design system
│   ├── app.js               # Client-side prediction engine & chart logic
│   └── Acne_dataset.csv     # Public download copy of dataset
└── README.md                # Project documentation
```

---

## 💻 How to Run Locally

### Option A: Via Python Flask Server (Recommended)
1. Clone the repository:
   ```bash
   git clone https://github.com/Nishtha56/Acne-and-Pimple-Prediction.git
   cd Acne-and-Pimple-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   python app.py
   ```
4. Open your browser at **`http://localhost:5000`**.

### Option B: Standalone in Browser
Open `static/index.html` directly in any web browser. The built-in client-side ML engine operates seamlessly offline.

### Option C: Retrain Models with New Data
Whenever new survey records are collected, run:
```bash
python train_and_export.py
```

---

## ☁️ Free Deployment Guide

### Deploy on Render (Recommended for Flask API + Frontend)
1. Push this repository to your GitHub account.
2. Go to [render.com](https://render.com) and create a new **Web Service**.
3. Select your repository.
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn app:app`
6. Choose the **Free** instance type and click **Deploy**.

### Deploy on Vercel
1. Go to [vercel.com](https://vercel.com) and import your repository.
2. The included `vercel.json` will automatically configure the routing.
3. Click **Deploy**.

---

## 📜 License & Disclaimer
This project is for educational and research purposes. Medical advice or treatment should always be confirmed with a licensed dermatologist.
