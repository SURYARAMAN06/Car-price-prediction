Car Price Prediction using Machine Learning

Project Overview
This project aims to predict the price of a used car based on its features such as mileage, brand, year, fuel type, and more. Various machine learning models were implemented, and Linear Regression showed promising results.

Dataset
The dataset used for this project is available on Kaggle - Vehicle Dataset from Cardekho. It contains features such as:

Car Name
Year
Selling Price
Present Price
Kms Driven
Fuel Type (Petrol, Diesel, CNG)
Seller Type (Dealer, Individual)
Transmission (Manual, Automatic)

Owner

Project Structure
bash
Copy
Edit
/car-price-prediction/
│
├── README.md               # Project overview and instructions
├── car_price_prediction.py  # Python script for the project
├── data/
│   └── car_data.csv         # Dataset used
├── images/
│   └── actual_vs_predicted_train.png   # Visualization (training set)
│   └── actual_vs_predicted_test.png    # Visualization (testing set)
└── requirements.txt         # List of dependencies

Libraries Used
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn


Model and Results

We used Linear Regression to predict car prices. The model was trained on 90% of the dataset and tested on 10%. The evaluation metric used was R-squared error.

Training R-squared: 0.879

Testing R-squared: 0.836

Visualizations
Actual vs Predicted Prices (Training Data)

Actual vs Predicted Prices (Testing Data)


Conclusion
The Car Price Prediction project shows that machine learning can be effectively applied to predict used car prices based on historical data. Linear Regression performed well with an R-squared error of 0.836 on the test data.
