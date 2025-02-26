# 📈 Demand Forecasting

## 📌 Introduction
Demand forecasting is a crucial aspect of business strategy, helping companies predict future sales based on historical data. In this project, I trained a **RandomForestRegressor** model using the **train.csv** dataset to predict units sold. I also explored performance metrics like **Root Mean Squared Error (RMSE)** and used **GridSearchCV** to optimize the model. Additionally, I introduced new features such as **price_difference** and **price_increase_pct** to improve the prediction accuracy. Let's dive into the details! 🚀

## 🔄 train_test_split
To evaluate our model effectively, I split the dataset into training and testing sets. I used an 80-20 ratio, ensuring that the model learns from a majority of the data while being tested on unseen samples:
```python
from sklearn.model_selection import train_test_split

x, y = data_frame.drop("units_sold", axis=1), data_frame["units_sold"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
This step prevents overfitting and allows to assess how well our model generalizes to new data.

## 🏆 How the model scored
Once the model was trained, I evaluated its performance using the `.score()` function, which calculates the **coefficient of determination (R² score)**:
```python
model.score(x_test, y_test)  # Performance on test data
model.score(x_train, y_train)  # Performance on training data
```
A high R² score indicates that the model explains most of the variance in the data, whereas a low score suggests that it may not be a good predictor.

## 📉 Root Mean Squared Error (RMSE)
RMSE is a popular metric for regression tasks as it measures the average error magnitude between predicted and actual values. Lower RMSE values indicate better model accuracy:
```python
from sklearn.metrics import root_mean_squared_error

y_pred = model.predict(x_test)
rmse = root_mean_squared_error(y_pred, y_test)
print("RMSE:", rmse)
```
By minimizing RMSE, I ensured our model makes more precise predictions.

## 📊 Creating the Scatterplot (Matplotlib)
To visualize the model's performance, I created a scatter plot comparing predicted vs. actual values. A perfect model would have all points on the red diagonal line:
```python
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(y_pred, y_test)
x_values = np.linspace(y_pred.min(), y_pred.max(), 100)
plt.plot(x_values, x_values, color="red")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Predicted vs Actual Scatter Plot")
plt.show()
```
This plot helps in identifying trends, patterns, and potential outliers in the predictions.

## 🌲 Using a RandomForestRegressor
The **RandomForestRegressor** is a powerful ensemble learning method that builds multiple decision trees to improve accuracy and reduce overfitting:
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
By setting `n_jobs=-1`, I utilized all available CPU cores for faster model training.

## 🛠️ Using GridSearchCV
To fine-tune the **RandomForestRegressor**, I employed **GridSearchCV**, which performs an exhaustive search over specified hyperparameters:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [10, 20],
    "min_samples_split": [2, 3]
}

model = RandomForestRegressor(n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, verbose=2, cv=3)
grid_search.fit(x_train, y_train)
```
This process ensures I found the optimal combination of parameters for improved model performance.

## 🔢 Adding price_difference and price_increase_pct (percentage)
Feature engineering plays a vital role in improving model accuracy. I introduced two new features:
1. **price_difference**: Measures the absolute difference between `total_price` and `base_price`.
2. **price_increase_pct**: Calculates the percentage increase in price.
```python
data_frame["price_difference"] = data_frame["total_price"] - data_frame["base_price"]

data_frame["price_increase_pct"] = ((data_frame["total_price"] - data_frame["base_price"]) / data_frame["base_price"]) * 100
```
These features provide valuable insights into pricing trends and their impact on demand.

## 🎯 Conclusion
This project demonstrated the power of **RandomForestRegressor** in demand forecasting. By applying **train-test splitting, model evaluation, hyperparameter tuning, and feature engineering**, a predictive model was built that helps businesses make informed decisions. 📊💡

Future improvements could include:
- Exploring other regression models (e.g., **XGBoost, Lasso Regression**)
- Testing on a larger dataset
- Incorporating time-series forecasting techniques