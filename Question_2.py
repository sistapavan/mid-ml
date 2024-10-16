import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('california_housing_random.csv')
X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']
model = LinearRegression()
model.fit(X, y)
test_data = [[30, 2463, 444, 1000, 455, 4.7]]
predicted_value = model.predict(test_data)
print(f"Predicted median house value: {predicted_value[0]}")
