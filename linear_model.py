import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from train_test_split import x_test,x_train,y_test,y_train
from sklearn.model_selection import cross_val_score


# in this module we will train and test the model
#  for that first we will apply k fold cross validation

lr=LinearRegression()

scores= cross_val_score(lr,x_train,y_train,cv=3,scoring='r2')

print(scores.mean())

model_train=lr.fit(x_train,y_train)

print(x_test)
print(y_test)

pred=lr.predict(x_test.iloc[[1]])
print(pred)