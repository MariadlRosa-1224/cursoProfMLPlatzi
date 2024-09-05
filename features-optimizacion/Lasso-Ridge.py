# Importar librerias importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Importar los modulos de Lasso e Rigge para el estudio de componentes principales

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Haremos una comparacion de los resultados de estos dos metodos con un modelo de regresion lineal

from sklearn.linear_model import LinearRegression


# Importar las librerias de particion de datos
from sklearn.model_selection import train_test_split

# Importar libreria de metrica de medicion
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    
    dt = pd.read_csv('../data/felicidad.csv')

    print(dt.describe())

# # Separamos los features de las etiquetas
    x = dt[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dt[['score']]

# # Dividimos los datos en entrenamiento y prueba
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)   

# # creamos variable del metodo de Lasso

    lasso = Lasso().fit(x_train, y_train)

    y_pred_lasso = lasso.predict(x_test)

# # creamos variable del metodo de Ridge

    ridge = Ridge()

    ridge.fit(x_train, y_train)

    y_pred_ridge = ridge.predict(x_test)

# # creamos variable del metodo de regresion lineal

    lr = LinearRegression()

    lr.fit(x_train, y_train)

    y_pred_lr = lr.predict(x_test)

# # Revisamos el error cuadratico medio de los modelos


    print("MSE LASSO: ", mean_squared_error(y_test, y_pred_lasso))

    print("MSE RIDGE: ", mean_squared_error(y_test, y_pred_ridge))  

    print("MSE LINEAR REGRESSION: ", mean_squared_error(y_test, y_pred_lr))

# # Revisamos los coeficientes de los modelos

    print('='*32)
    print("COEFICIENTES LASSO: ", lasso.coef_)
    print('='*32)
    print("COEFICIENTES RIDGE: ", ridge.coef_)