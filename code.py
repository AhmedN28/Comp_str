#importing libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# reading the data
data = pd.read_excel("D:\Learnbay\Dataset\ML\Concrete_Data.xls")

# simplying column names
req_col_names = ["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer","CoarseAggregate", "FineAggregate", "Age", "CC_Strength"]

curr_col_names = list(data.columns)

mapp = {}
for i,name in enumerate(curr_col_names):
  mapp[name] = req_col_names[i]

data = data.rename(columns=mapp)

# Separating Input Features and Target Variable

X = data.iloc[:,:-1] 
# Features - All columns but last
y = data.iloc[:,-1]    
# Target - Last Column

#Splitting data into Training and Test splits
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# model building
lr = LinearRegression()
# Linear Regression

lr.fit(x_train, y_train)
# fitting the linear regression model

# Saving model to disk
pickle.dump(lr,open('comp_str_57.pkl','wb'))

# Loading the model to compare the results
model = pickle.load(open('comp_str_57.pkl','rb'))

