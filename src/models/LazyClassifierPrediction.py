import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from collections import Counter

def lazy_prediction_models(df):
    """
    Define de forma rápida e automatizada os resultados e métricas de cada modelo de machine learning para os dados fornecidos. É utilizado para definir inicialmente e de forma rápida os modelos que apresentam melhores desempenhos com dados considerados 'crus'.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
         
    """

    #Divisão do dataset em features e target
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    print('Distribuição original das classes:', Counter(y))

    #Dividir o dataset em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Inicializar o LazyClassifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

    #Treinar e avaliar os modelos
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    #Mostrar os resultados
    with open("./assets/models_evaluation/LazyPredict.txt", "w") as file:
        file.write(models.to_string(index=True))
