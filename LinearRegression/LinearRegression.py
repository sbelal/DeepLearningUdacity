from sklearn.datasets import california_housing as calHousingDataSet
from sklearn.linear_model import LinearRegression as LinearRegression

housingData = calHousingDataSet.fetch_california_housing()
model = LinearRegression()
model.fit(housingData.data, housingData.target)



print(housingData.data)

print(housingData.feature_names)
print("Hello world2")