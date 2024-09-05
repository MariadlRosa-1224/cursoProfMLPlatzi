# Importar librerias importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Importar los modulos de PCA e IPCA para el estudio de componentes principales

from sklearn.decomposition import KernelPCA


# Haremos una comparacion de los resultados de estos dos metodos con un modelo logistico

from sklearn.linear_model import LogisticRegression

# Importar las librerias de preprocesamiento de datos

from sklearn.preprocessing import StandardScaler


# Importar las librerias de particion de datos
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    dt_heart = pd.read_csv('../data/heart.csv')

    print(dt_heart.head(5))

# Separamos los features de las etiquetas

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

# Normalizamos los datos

    dt_features = StandardScaler().fit_transform(dt_features)

# Dividimos los datos en entrenamiento y prueba

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)


# revisamos las dimensiones de los datos

    print(X_train.shape)
    print(y_train.shape)

# Creamos variable del metodo de KPCA

    kpca = KernelPCA(n_components=3)

    kpca.fit(X_train)

# Aplicamos el metodo de KPCA en los datos de entrenamiento y prueba

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)


# Aplicamos un modelo de regresion logistica

    logistic = LogisticRegression(solver='lbfgs')


# aplicamos la regresion logistica
    logistic.fit(dt_train, y_train)

# Revisamos el score del modelo
    print("SCORE KPCA: ", logistic.score(dt_test, y_test))

