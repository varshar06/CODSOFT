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

---

## 📌 **Task 2: Movie Rating Prediction**

# 🎬 Movie Rating Prediction - CodSoft Internship Task 2

This project is part of my **CodSoft Data Science Internship**, where I built a Machine Learning model to **predict IMDb movie ratings** based on factors like **genre, director, and actors.**  

## 📂 Project Structure
📁 Task2_Movie_Rating_Prediction
├── 📄 IMDb Movies India.csv (Dataset)
├── 📄 movie_rating.py (Python Script)
├── 📄 README.md (Project Documentation)


---

## 📊 Features & Enhancements  
✅ **Uses Linear Regression** to predict movie ratings  
✅ **Encodes categorical data** (Genre, Director, Actors)  
✅ **Removes missing values & cleans data**  
✅ **Feature scaling for better predictions**  
✅ **Visualizations: Scatter plot comparing actual vs predicted ratings**  

---

## 📂 Dataset Information  
The dataset **IMDb Movies India.csv** contains:  
- `Name` – Movie title  
- `Year` – Release year  
- `Duration` – Movie duration  
- `Genre` – Movie genre(s)  
- `Rating` – IMDb rating (Target variable 🎯)  
- `Votes` – Number of votes  
- `Director` – Movie director  
- `Actor 1, Actor 2, Actor 3` – Main actors  

---

## 🛠 Setup & Installation  

### **🔹 Step 1: Install Dependencies**  
Before running the script, install the required libraries:  
```sh
pip install pandas numpy seaborn matplotlib scikit-learn





