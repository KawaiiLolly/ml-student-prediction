# Student Performance Prediction ML Project

A simple end-to-end Machine Learning web application built using **Python, Flask, and Scikit-learn** to predict student mathematics performance based on academic and demographic features.

---

## 📌 Project Overview

This project predicts a student's **Math Score** using machine learning models trained on student exam performance data.

The system:

* Uses multiple regression models for training
* Compares model performance using **R² score**
* Selects the best-performing model automatically
* Accepts user input through a web interface
* Predicts mathematics score instantly

---

## 🚀 Features

* Home page with project overview
* Predict student performance using form input
* View full dataset in browser
* Rounded prediction output up to 2 decimal places
* Modern responsive UI using HTML + CSS
* Flask-based routing system

---

## 📂 Project Structure

```bash
ML-PROJECT/
│── artifacts/
│── logs/
│── notebook/
│── src/
│── templates/
│   │── index.html
│   │── home.html
│   │── dataset.html
│── app.py
│── requirements.txt
│── setup.py
│── README.md
```

---

## ⚙️ Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS

---

## 📊 Input Features Used

The prediction model uses:

* Gender
* Race / Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course
* Reading Score
* Writing Score

---

## 🧠 Model Training

Several regression models were trained and evaluated.

Best model selected using:

* **R² Score**

---

## 🌐 Routes

### Home Page

```bash
/
```

Project introduction page

---

### Prediction Page

```bash
/predictdata
```

Input student details and predict maths score

---

### Dataset Page

```bash
/dataset
```

View complete dataset in scrollable table

---

## ▶️ Run Locally

### 1. Clone repository

```bash
git clone <your-repository-link>
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
```

---

### 3. Activate environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run Flask app

```bash
python app.py
```

---

## 🌍 Open in Browser

```bash
http://127.0.0.1:5000
```

---

## 📌 Future Improvements

* Add charts and visual analytics
* Add downloadable prediction reports

---
