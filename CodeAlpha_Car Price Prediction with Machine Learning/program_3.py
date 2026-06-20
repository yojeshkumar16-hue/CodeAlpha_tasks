import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Extract ZIP file
with ZipFile(r"C:\Users\USER\Downloads\archive (2).zip", "r") as zip_ref:
    zip_ref.extractall()

# Load dataset
df = pd.read_csv("car data.csv")

# Display first rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Create car age feature
current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df = df.drop(["Car_Name", "Year"], axis=1)

# Convert categorical columns into numbers
le = LabelEncoder()

df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le.fit_transform(df["Selling_type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(by="Importance", ascending=False))
