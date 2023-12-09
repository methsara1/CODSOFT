import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = pd.read_csv("imdb_india_movies.csv")
data.dropna(subset=["Year", "Genre", "Rating"], inplace=True)
genre_encoder = OneHotEncoder(sparse=False)
data_encoded = pd.concat([
    data.drop(["Genre"], axis=1),
    pd.DataFrame(genre_encoder.fit_transform(data[["Genre"]])),], axis=1)
scaler = StandardScaler()
data_encoded[["Year", "Duration", "Votes"]] = scaler.fit_transform(data_encoded[["Year", "Duration", "Votes"]])
X = data_encoded.drop("Rating", axis=1)
y = data_encoded["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
r2 = model.score(X_test, y_test)
print(f"R-squared: {r2:.4f}")
