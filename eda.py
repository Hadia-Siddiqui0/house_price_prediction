import numpy as np
import pandas as pd

data=pd.read_csv("Housing.csv")
coulumn_name=data.columns
data_description= data.describe()
data_median=data.median()
miss_value=data.isna().sum()
shape=data.shape
size=data.size
print(coulumn_name)
print(data_description)
print(data_median)
print(miss_value)
print(size)
print(shape)


# right skew very few high prices following up
#  will do outliers after train test split
#  having no missing values
# 13 columns and 545 rows all filled
#  clean and good data
