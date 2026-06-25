import numpy as np
import pandas as pd
from eda import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['mainroad','guestroom','basement','hotwaterheating' ,'airconditioning','prefarea' ,'furnishingstatus']:
    data[col]=le.fit_transform(data[col])

x=data[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']]
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)