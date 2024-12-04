import pandas as pd 

# gradient boosting esta basado en arboles de decision

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_hear = pd.read_csv('../data/heart.csv')
    print(dt_hear['target'].describe())

    x = dt_hear.drop(['target'], axis=1)
    y = dt_hear['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=50).fit(x_train, y_train) 
    # n_estimators es el numero de arboles que vamos a utilizar
    # Construye arboles pequeños y los va a ir sumando para mejorar la prediccion
    boost_predict = boost.predict(x_test)
    print('='*64)
    print(accuracy_score(boost_predict, y_test))
    # El resultado deberia ser mas preciso que el de bagging
    # debido a que se va a ir ajustando a los errores que se vayan cometiendo