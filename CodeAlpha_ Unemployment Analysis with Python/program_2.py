import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Extract ZIP file
with ZipFile(r"C:\Users\USER\Downloads\archive (1).zip", "r") as zip_ref:
    zip_ref.extractall()

# Load dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Overall average unemployment rate
avg_rate = df["Estimated Unemployment Rate (%)"].mean()
print("\nAverage Unemployment Rate:", round(avg_rate, 2), "%")

# Average unemployment by state
region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate by Region:")
print(region_avg.sort_values(ascending=False))

# Plot 1: Overall unemployment trend
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Estimated Unemployment Rate (%)"])
plt.title("Unemployment Rate Trend in India (2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Top 10 regions with highest unemployment
top10 = region_avg.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top10.plot(kind="bar")
plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Plot 3: Monthly unemployment trend
monthly_avg = df.groupby(df["Date"].dt.month)["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10, 5))
monthly_avg.plot(kind="bar")
plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Covid impact analysis
covid_period = df[df["Date"] >= "2020-03-01"]

covid_avg = covid_period["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate During Covid Period:")
print(round(covid_avg, 2), "%")
