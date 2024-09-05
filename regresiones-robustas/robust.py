import pandas as pd
import sklearn as sk

from sklearn.linear_model import (
    HuberRegressor, RANSACRegressor 
)

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

if __name__ == "__main__":
    dataset = pd.read_csv('../data/felicidad_corrupt.csv')
    print(dataset.describe())

    x = dataset.drop(['country', 'score'], axis=1)
    y = dataset[['score']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'Huber': HuberRegressor(epsilon=1.35),
        'RANSAC': RANSACRegressor() # meta estimador
    }