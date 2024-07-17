import numpy as np 
import pandas as pd 

orig_data = pd.read_csv('C:/ml/IPDproject/data/Cardekho_Extract.csv')
data = orig_data.copy()

data.drop(columns=['new-price','full_name','owner_type','seller_type','Source.Name','web-scraper-order', 'web-scraper-start-url'], axis=1, inplace=True)
data = data.dropna(inplace = False)

data = data.reset_index(drop = True)

data['selling_price'] = data['selling_price'].str.replace('*','')
data['selling_price'] = data['selling_price'].str.replace(',','')
for i in range(data.shape[0]):
    try:
        price = float(data['selling_price'][i].split(' ')[0])
        digit = data['selling_price'][i].split(' ')[1]
        if digit == 'Lakh':
            price = price * 100000
            data['selling_price'][i] = price
        elif digit == 'Cr':
            price = price * 10000000
            data['selling_price'][i] = price
    except:
        data['selling_price'][i] = float(price)

# kilometer driven
data['km_driven'] = data['km_driven'].str.split(' ', n=1, expand=True)[0]
data['km_driven'] = data['km_driven'].str.replace(',','')
# Mileage
data['mileage'] = data['mileage'].str.split(' ', expand=True)[0].str.split('e', expand=True)[2]
# Engine
data['engine'] = data['engine'].str.split(' ', expand=True)[0].str.split('e',expand=True)[1]
# Max Power
data['max_power'] = data['max_power'].str.split(' ', expand=True)[1].str.split('r',expand=True)[1]
# Seats 
data['seats'] = data['seats'].str.split('s', expand=True)[1]

data['selling_price'] = pd.to_numeric(data['selling_price'], errors='coerce').astype('Float64')
data['km_driven'] = pd.to_numeric(data['km_driven'], errors='coerce').astype('Float64')
data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce').astype('Float64')
data['engine'] = pd.to_numeric(data['engine'], errors='coerce').astype('Int64')
data['max_power'] = pd.to_numeric(data['max_power'], errors='coerce').astype('Float64')
data['seats'] = pd.to_numeric(data['seats'], errors='coerce').astype('Int64')
      

import seaborn as sn
sn.jointplot(x= 'km_driven', y= 'selling_price', data = data)
sn.jointplot(x= 'mileage', y= 'selling_price', data = data)
sn.jointplot(x= 'engine', y= 'selling_price', data = data)
sn.jointplot(x= 'max_power', y= 'selling_price', data = data)
sn.jointplot(x= 'seats', y= 'selling_price', data = data)

data = data[data['selling_price'] < 20000000]
data = data[data['km_driven'] < 1000000]
data = data[data['mileage'] < 40]
data = data[data['seats'] < 11]
data = data.reset_index(drop=True)
data.engine = data.engine.fillna(data.engine.median())
data.max_power = data.max_power.fillna(data.max_power.median())

sn.jointplot(x= 'km_driven', y= 'selling_price', data = data)
sn.jointplot(x= 'mileage', y= 'selling_price', data = data)
sn.jointplot(x= 'engine', y= 'selling_price', data = data)
sn.jointplot(x= 'max_power', y= 'selling_price', data = data)
sn.jointplot(x= 'seats', y= 'selling_price', data = data)

data = pd.get_dummies(data=data, columns=['fuel_type','transmission_type'], drop_first=True)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

x = data.iloc[:,1:]
y = data['selling_price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=25)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def do_prediction(classifier):
    
    # training the classifier on the dataset
    classifier.fit(x_train, y_train)
    
    #Do prediction and evaluting the prediction
    prediction = classifier.predict(x_test)
    cross_validation_score = cross_val(x_train, y_train, classifier)
    error = mean_absolute_error(y_test, prediction)
    
    return error, cross_validation_score

def cross_val(x_train, y_train, classifier):
    
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
    return accuracies.mean()

data2 = data.describe()

model_1 = LinearRegression()
error, score = do_prediction(model_1)

print('Linear Regression MAE: {}'.format(round(error,2)))
print('Cross validation score: {}'.format(round(score,2)))

model_2 = DecisionTreeRegressor()
error, score = do_prediction(model_2)

print('Decision Tree Regressor MAE: {}'.format(round(error,2)))
print('Cross validation score: {}'.format(round(score,2)))

model_3 = RandomForestRegressor()
error, score = do_prediction(model_3)

print('Random Forest Regressor MAE: {}'.format(round(error,2)))
print('Cross validation score: {}'.format(round(score,3)))

model_4 = RandomForestRegressor(n_estimators=400,
                                  min_samples_split=10,
                                  min_samples_leaf=1,
                                  max_features='sqrt',
                                  max_depth=60,
                                  bootstrap=False)

error, score = do_prediction(model_4)
print('Random Forest with hyperparameter tuning MAE: {}'.format(round(error,2)))
print('Cross validation score: {}'.format(round(score,3)))