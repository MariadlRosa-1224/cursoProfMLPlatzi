# Importar librerias importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Importar los modulos de PCA e IPCA para el estudio de componentes principales

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

# creamos variable del metodo de PCA

    pca = PCA(n_components=3)

    pca.fit(X_train)

# creamos variable del metodo de IPCA

    ipca = IncrementalPCA(n_components=3, batch_size=10)

    ipca.fit(X_train)

# usamos matplotlib para graficar el peso de las componentes

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)

    plt.show()

# Aplicamos un modelo de regresion logistica

    logistic = LogisticRegression(solver='lbfgs')

# Entrenamos el modelo con PCA

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)

# aplicamos la regresion logistica
    logistic.fit(dt_train, y_train)

# Revisamos el score del modelo
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

# Entrenamos el modelo con IPCA
    
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    
    # aplicamos la regresion logistica

    logistic.fit(dt_train, y_train)

    # Revisamos el score del modelo
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    