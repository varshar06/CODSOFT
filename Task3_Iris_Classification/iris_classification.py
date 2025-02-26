import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
dataset_path = r"C:\Users\Arunesh_27\Documents\CODSOFT\Task3_Iris_Classification\IRIS.csv"
 # Make sure the dataset is in the same folder
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Oops! Couldn't find the dataset '{dataset_path}'. Make sure it's in the right place.")

# Let's take a quick look at the dataset
print("Here's a sneak peek at the data:")
print(df.head())

# Checking if any values are missing
print("\nAre there any missing values?")
print(df.isnull().sum())

# Convert the species column into numbers so the model can understand it
label_enc = LabelEncoder()
df["species"] = label_enc.fit_transform(df["species"])

# Splitting the dataset into features (X) and target (y)
X = df.drop("species", axis=1)  # Features
Y = df["species"]  # Target variable

# Standardizing the data to improve model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Let's train our model using Logistic Regression!
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Checking how well our model performed
accuracy = accuracy_score(y_test, y_pred)
print("\nAwesome! Our model's accuracy is:", accuracy)
print("\nHere's a detailed report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix - helps visualize how well the model predicted
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix - Model Performance")
plt.show()

# Scatter plot to visualize Sepal Length vs Sepal Width for different species
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["sepal_length"], y=df["sepal_width"], hue=df["species"], palette="coolwarm")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width - Classification View")
plt.show()
