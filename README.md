# Car Price Prediction Project

This repository contains a project for predicting car prices using various machine learning algorithms. The project includes data preprocessing, exploratory data analysis, and model training with different regression models.

## Data Preprocessing

The data preprocessing steps include:

1. Dropping irrelevant columns.
2. Handling missing values.
3. Converting categorical data to numerical data using one-hot encoding.
4. Feature scaling using `StandardScaler`.

## Exploratory Data Analysis

Exploratory data analysis (EDA) includes:

1. Visualization of relationships between features and the target variable `selling_price` using Seaborn joint plots.
2. Filtering outliers in the dataset.

## Model Training

Four models were trained and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Random Forest Regressor with hyperparameter tuning

## Results

The performance of the models is evaluated using Mean Absolute Error (MAE) and cross-validation score. Here are the results:

- **Linear Regression**
  - MAE: 262061.13
  - Cross-validation score: 0.67

- **Decision Tree Regressor**
  - MAE: 129729.04
  - Cross-validation score: 0.82

- **Random Forest Regressor**
  - MAE: 105018.74
  - Cross-validation score: 0.909

- **Random Forest with hyperparameter tuning**
  - MAE: 100736.04
  - Cross-validation score: 0.915


## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Seaborn

## Acknowledgements

- Data sourced from [Cardekho](https://www.cardekho.com).
