import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# This file contains information about movies like genre, director, actors, and ratings
dataset_path = r"C:\Users\Arunesh_27\Documents\CODSOFT\Task2_Movie_Rating_Prediction\IMDb Movies India.csv"


def load_data(path):
    try:
        df = pd.read_csv(path, encoding="ISO-8859-1")  # Read the dataset with correct encoding
        print("✅ Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {path} not found.")
        exit()

df = load_data(dataset_path)

# Drop some columns that are not useful for predicting ratings
df.drop(["Name", "Year", "Duration"], axis=1, inplace=True)

# Remove rows where ratings are missing (since that's our target)
df.dropna(subset=["Rating"], inplace=True)

# Fill missing values in other columns with "Unknown"
df.fillna("Unknown", inplace=True)
df["Votes"] = df["Votes"].astype(str).str.replace(",", "").astype(float)

# Convert text data (like genre, director, actors) into numbers
encoder = LabelEncoder()
categorical_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
for col in categorical_features:
    df[col] = encoder.fit_transform(df[col])

# Define input features (X) and target (y)
X = df.drop("Rating", axis=1)  # All columns except "Rating" are input features
y = df["Rating"]  # The target we are predicting

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features (scaling data so that big numbers don’t dominate small ones)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a simple Linear Regression model (predicts rating based on input features)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Model makes predictions on test data

# Evaluate the model - Check how well it performed
mse = mean_squared_error(y_test, y_pred)  # Lower is better
r2 = r2_score(y_test, y_pred)  # Closer to 1 is better

print(f"\n📊 Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-Squared Score: {r2:.4f}")

# Visualize how well the predictions match the actual ratings
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show(block=True)
