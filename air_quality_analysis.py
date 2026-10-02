import pandas as pd

import seaborn as sns
import numpy as np


df = pd.read_excel("data/AirQualityUCI.xlsx")


df = df.replace(-200, np.nan)


df = df.dropna(subset=["CO(GT)"])


features = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "T",
    "RH",
    "AH"
]


df = df[features + ["CO(GT)"]]


df = df.dropna()

print("Clean dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())



import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(8, 5))
sns.histplot(df["CO(GT)"], bins=30, kde=True)

plt.title("Distribution of Carbon Monoxide (CO)")
plt.xlabel("CO(GT) concentration")
plt.ylabel("Number of observations")

plt.tight_layout()
plt.savefig("images/co_distribution.png")
plt.show()





plt.figure(figsize=(10, 7))

correlation = df.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Between Air Quality Variables")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()




plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="NOx(GT)",
    y="CO(GT)"
)

plt.title("CO Concentration vs NOx")
plt.xlabel("NOx(GT)")
plt.ylabel("CO(GT)")

plt.tight_layout()
plt.savefig("images/co_vs_nox.png")
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Features and target
X = df[features]
y = df["CO(GT)"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("\nLinear Regression Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs Predicted CO Concentration")

plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png")
plt.show()


from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Get feature importance
importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)


# Plot feature importance
plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance for CO Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()
