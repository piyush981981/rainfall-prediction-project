# Rainfall Prediction Project using Synthetic Austin Weather Data

# ----------------------------
# Step 1: Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

# ----------------------------
# Step 2: Create Synthetic Dataset (Simulating austin_weather.csv)
# ----------------------------
# Generate a small dataset of 100 days with random weather attributes
np.random.seed(42)
n = 100

data = {
    'Date': pd.date_range(start='2020-01-01', periods=n, freq='D'),
    'TempAvgF': np.random.uniform(45, 100, n),
    'HumidityAvgPercent': np.random.uniform(20, 90, n),
    'DewPointAvgF': np.random.uniform(30, 75, n),
    'SeaLevelPressureAvgInches': np.random.uniform(29.5, 30.5, n),
    'VisibilityAvgMiles': np.random.uniform(5, 10, n),
    'WindAvgMPH': np.random.uniform(0, 20, n),
    'PrecipitationSumInches': np.abs(np.random.normal(0.05, 0.1, n))  # Mostly small rain values
}

df = pd.DataFrame(data)

# Introduce a few "T" and "-" values in precipitation (simulating trace amounts/missing)
df.loc[[5, 12, 22], 'PrecipitationSumInches'] = 'T'
df.loc[[3, 30], 'PrecipitationSumInches'] = '-'

# ----------------------------
# Step 3: Data Cleaning
# ----------------------------

# Replace "T" and "-" with 0
df['PrecipitationSumInches'] = df['PrecipitationSumInches'].replace(['T', '-'], 0)

# Convert Precipitation column to numeric
df['PrecipitationSumInches'] = pd.to_numeric(df['PrecipitationSumInches'])

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# ----------------------------
# Step 4: Exploratory Data Analysis
# ----------------------------

# Plot precipitation over time
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['PrecipitationSumInches'], marker='o')
plt.title('Daily Precipitation Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation (inches)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# ----------------------------
# Step 5: Prepare Data for Modeling
# ----------------------------

features = ['TempAvgF', 'HumidityAvgPercent', 'DewPointAvgF', 
            'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH']
target = 'PrecipitationSumInches'

X = df[features]
y = df[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ----------------------------
# Step 6: Linear Regression Model
# ----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ----------------------------
# Step 7: Evaluation
# ----------------------------

print("Model Coefficients:")
for f, c in zip(features, model.coef_):
    print(f"{f}: {c:.4f}")

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.6f}")

# Scatter plot of predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.title('Actual vs Predicted Precipitation')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()
