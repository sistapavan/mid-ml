import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
df = pd.read_csv('california_housing_random.csv')
X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']
knn_model = KNeighborsRegressor(n_neighbors=20)
knn_model.fit(X, y)
test_data = [[30, 2463, 444, 1000, 455, 4.7]]
knn_predicted_value = knn_model.predict(test_data)
print(f"kNN predicted median house value: {knn_predicted_value[0]}")
