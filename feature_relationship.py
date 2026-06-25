import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eda import data
import seaborn as sns

df=data.corr()
sns.heatmap(df,annot=True,cmap="coolwarm")
plt.title("visualizing correlation of data")
plt.show()

plt.scatter(data['area'],data['price'])
plt.title("area scatter plot")
plt.show()

sns.boxplot(x=data['bedrooms'],y=data['price'])
plt.title("bedrooms boxplot plot")
plt.show()

sns.boxplot(x=data['bathrooms'],y=data['price'])
plt.title("bathrooms boxplot plot")
plt.show()

sns.boxplot(x=data['stories'],y=data['price'])
plt.title("stories boxplot plot")
plt.show()

sns.boxplot(x=data['parking'],y=data['price'])
plt.title("parking boxplot plot")
plt.show()

# price is linearly dependent on price
# price is impact with everything but the highest impact on prices is of the area
# there are outliers in bathrooms, price, bedrooms and eery input features means some have more bathrooms, bedroms etc than normaly
# by which there is also some outliers on prices on the above side means high
