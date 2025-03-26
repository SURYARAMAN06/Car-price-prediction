# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
car_data = pd.read_csv("data/car_data.csv")

# Explore the dataset
print(car_data.shape)
print(car_data.columns)
print(car_data.dtypes)

# Preprocess the data (Convert categorical variables to numerical)
car_data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_data.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_data.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Define features (X) and target (y)
X = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_data["Selling_Price"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train a Linear Regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

# Predict on training data and calculate R-squared error
train_predictions = lin_reg_model.predict(X_train)
train_error_score = metrics.r2_score(y_train, train_predictions)
print(f"R_squared error (Train): {train_error_score}")

# Visualize Actual vs Predicted prices (Training Data)
plt.scatter(y_train, train_predictions)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price (Training)')
plt.savefig('images/actual_vs_predicted_train.png')  # Save plot
plt.show()

# Predict on test data and calculate R-squared error
test_predictions = lin_reg_model.predict(X_test)
test_error_score = metrics.r2_score(y_test, test_predictions)
print(f"R_squared error (Test): {test_error_score}")

# Visualize Actual vs Predicted prices (Testing Data)
plt.scatter(y_test, test_predictions)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price (Testing)')
plt.savefig('images/actual_vs_predicted_test.png')  # Save plot
plt.show()
