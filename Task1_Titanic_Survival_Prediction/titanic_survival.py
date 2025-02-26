import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load  the dataset
dataset_path = r"C:\Users\Arunesh_27\Documents\CODSOFT\Task1_Titanic_Survival_Prediction\Titanic-Dataset.csv"

try:
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: {dataset_path} not found. Please check the file path.")
    exit()

# Display basic info about the dataset
print("\nDataset Overview:")
print(df.info())

# Drop irrelevant columns 
df.drop(["Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing values with custom logic
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna("U", inplace=True)  # 'U' stands for Unknown
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Encode categorical variables 
encoder = LabelEncoder()
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Custom mapping
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# Create a new feature: Family Size 
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1  # Including self

# Drop Name column since it's text-based
df.drop("Name", axis=1, inplace=True)

# Splitting dataset into features and target variable
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Splitting into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Feature scaling using MinMaxScaler instead of StandardScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix visualization
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Data Visualization: Family Size vs Survival Rate 
plt.figure(figsize=(6, 4))
sns.barplot(x=df["Family_Size"], y=df["Survived"], ci=None, palette="magma")
plt.title("Survival Rate Based on Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Probability")
plt.show()
