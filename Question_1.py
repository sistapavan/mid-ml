import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('california_housing_random.csv')
X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']
model = LinearRegression()
model.fit(X, y)
population_coefficient = model.coef_[3]  
print(f"Coefficient for population: {population_coefficient}")
