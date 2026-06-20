import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Extract ZIP file
with ZipFile(r"C:\Users\USER\Downloads\archive (3).zip", "r") as zip_ref:
    zip_ref.extractall()

# Load dataset
df = pd.read_csv("Advertising.csv")

# Remove unnecessary index column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Display first rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Advertising impact
print("\nAdvertising Impact")
print("TV Coefficient:", model.coef_[0])
print("Radio Coefficient:", model.coef_[1])
print("Newspaper Coefficient:", model.coef_[2])

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
