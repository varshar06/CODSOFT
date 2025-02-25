# CODSOFT
This repository contains all my Codsoft Internship projects.
# 🚀 CodSoft Internship - Data Science Tasks


## 📌 Task 1: Titanic Survival Prediction  

# 🚢 Titanic Survival Prediction

This project predicts whether a passenger on the Titanic survived or not using Machine Learning models. The dataset includes passenger details such as age, gender, fare, class, and family size.

## 📂 Project Structure
📁 CODSOFT
├── 📄 Titanic-Dataset.csv
├── 📄 titanic_survival.py
├── 📄 README.md
The main script (`titanic_survival.py`) performs **data preprocessing, feature engineering, and model training**.

---

## 📊 Features & Enhancements
✅ **New Feature:** Family Size (SibSp + Parch + 1)  
✅ **Unique Scaling:** MinMaxScaler instead of StandardScaler  
✅ **Custom Encoding:** 'Sex' manually mapped to binary values  
✅ **Improved Data Handling:** 'Embarked' missing values replaced with 'U'  
✅ **Different Model:** Logistic Regression instead of Random Forest  
✅ **Visualization:** Confusion matrix & Family Size vs. Survival Rate  

---

## 📂 Dataset Information
The dataset **Titanic-Dataset.csv** contains:  
- `Survived`: 0 = No, 1 = Yes  
- `Pclass`: Ticket class (1st, 2nd, 3rd)  
- `Sex`: Male or Female  
- `Age`: Passenger’s age  
- `SibSp`: No. of siblings/spouses aboard  
- `Parch`: No. of parents/children aboard  
- `Fare`: Ticket fare  
- `Embarked`: Port of Embarkation (C, Q, S)  

---

## 🛠 Setup & Installation

### **🔹 Step 1: Install Dependencies**
Before running the script, install required libraries:  
```sh
pip install pandas numpy seaborn matplotlib scikit-learn







