import numpy as np
import pandas as pd
from train_test_split import x_train,y_train,x_test,y_test
import matplotlib.pyplot as plt
import seaborn as sns

# make a new column area_parking by adding both values and then see if price is highly or low coorrelated with it
# we will do this for train and test data seperatly to void data leakage
# training data 
x_train['area_parking'] = x_train['area'] + x_train['parking']

train_data = x_train.copy()
train_data['price'] = y_train


corr = train_data.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Checking Correlation")
plt.show()
# test data
x_test['area_parking'] = x_test['area'] + x_test['parking']

test_data = x_test.copy()
test_data['price'] = y_test


corr_test = test_data.corr()

sns.heatmap(corr_test, annot=True, cmap="coolwarm")
plt.title("Checking Correlation")
plt.show()

# our new column is highly coorelated so we are using it IN'SHA'ALLAH
