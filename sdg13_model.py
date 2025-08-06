import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Generate synthetic data
np.random.seed(42)
years = np.arange(1990, 2021)
co2 = 22 + (years - 1990) * 0.3 + np.random.normal(0, 1, len(years))
data = pd.DataFrame({'Year': years, 'CO2_Emissions': co2})

# Train/test split
X = data[['Year']]
y = data['CO2_Emissions']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
# Plot
plt.scatter(X, y, label='Actual CO2')
plt.plot(X, model.predict(X), color='red', label='Predicted CO2')
plt.title('CO2 Emissions Forecast (1990–2020)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.grid(True)
plt.savefig("demo_screenshot.png")
plt.show()
