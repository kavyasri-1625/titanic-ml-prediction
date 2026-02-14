# Titanic Survival Prediction using Machine Learning

This project builds a complete machine learning pipeline to predict passenger survival on the Titanic.

## 🎯 Objective
Predict whether a passenger survived based on demographic and travel information.

---

## 📊 Dataset
**Titanic Dataset**
- Source: Public dataset
- Records: ~891 passengers
- Target Variable: `Survived` (0 = No, 1 = Yes)

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing
- Handled missing values (Age, Embarks)
- Created new features:
  - FamilySize
  - IsAlone

### 2️⃣ Feature Engineering
- Converted categorical data using One-Hot Encoding
- Scaled numerical features

### 3️⃣ Train-Test Split
- 80% training
- 20% testing

### 4️⃣ Models Used
- Logistic Regression
- Random Forest Classifier

### 5️⃣ Model Evaluation
Metrics used:
- Accuracy
- F1 Score
- Classification Report

---

## 📈 Results

Random Forest performed better with improved prediction accuracy.

**Key Predictors of Survival:**
- Gender
- Passenger Class
- Fare Paid

---

## 📊 Visualizations
- Feature Importance Chart
- Model Performance Output

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

---

## 📂 Project Structure
