import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from evaluation import evaluation

def pipeline_models_evaluations(df, model, path, name):
    """
    Faz o treinamento do modelo com Stratified KFold, cross_val_score e plota a matriz de correlação e a curva ROC. Se o modelo for o KNearest Neighbors, plota ainda o grafico do Elbow Method para achar o valor ideal de K.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      model (SciKit-Learn Model): Modelo a ser avaliado.
      path (string): Caminho onde o arquivo será salvo
      name (string): nome do arquivo a ser salvo
         
    """
    
    #Definição das features e target
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    #Binarizar a saída
    classes = np.unique(y)

    #Criação do pipeline
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
    ])

    #Plotagem da Curva ROC
    evaluation.plot_multiclass_roc(X, y, classes, pipeline, path, name, test_size=0.2, random_state=42)

    #Avaliação do modelo com validação cruzada (cross_val_score)
    cross_val_acc = training_with_cross_val_score(pipeline, X, y)

    #Avaliação do modelo com validação cruzada (Stratified KFold)
    stratified_kfold_acc = training_with_Stratified_KFold(pipeline, X, y, path)
    
    #Divisão dos dados em treino e teste por meio do train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Predição simples dos dados
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    #Avaliação de métricas de desempenho
    evaluation.evaluate_all_metrics(y_test, y_pred, cross_val_acc, stratified_kfold_acc, path)

    #Plotagem da matriz de confusão
    evaluation.plot_confusion_matrix(y_test, y_pred, model, path)
    
    #Se o modelo for o KNearest Neighbors, plota o Elbow Method para definir o valor ideal de K
    if model.__class__.__name__ == "KNeighborsClassifier":
        knn_elbow_method(X, path)

def training_with_Stratified_KFold(model, X, y, path):
    """
    Treina o modelo passado como argumento com o uso de Stratified KFold e depois salva os resultados em um arquivo.
    
    Args:
      model (SciKit-Learn Model): modelo a ser treinado
      X (pd.DataFrame): features a serem treinadas
      y (pd.DataFrame): target do dataset
      path (string): Caminho em que o arquivo será salvo
         
    """

    #Stratified KFold com 10 divisões
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    #Adequação dos tipos de dados
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    #Definição dos arrays para armazenar o resultado de cada Folder
    scores_acc = []
    scores_f1 = []
    scores_recall = []
    scores_precision = []

    #Treinamento do modelo para cada folder e armazenamento dos resultados
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        scores_acc.append(accuracy_score(y_test, y_pred))
        scores_f1.append(f1_score(y_test, y_pred, average='weighted'))
        scores_recall.append(recall_score(y_test, y_pred, average='weighted'))
        scores_precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=True))

    #Predição simples e plotagem da matriz de confusão
    y_pred = model.predict(X_test)
    evaluation.plot_confusion_matrix(y_test, y_pred, model, path)

    return np.mean(scores_acc), np.mean(scores_precision), np.mean(scores_recall), np.mean(scores_f1)


def training_with_cross_val_score(model, X, y):
    """
    Faz o treinamento do modelo com o uso de cross_val_score (validação cruzada), observando a acurácia.

    Args:
      model (SciKit-Learn Model): modelo a ser treinado
      X (pd.DataFrame): features a serem treinadas
      y (pd.DataFrame): target do dataset
         
    """
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    score_sum = 0

    for i in range (1, 6):
        score_sum = score_sum + score[i - 1]
    score_sum = score_sum / 5

    return score_sum

def knn_elbow_method(X, path):
    """
    Plota o gráfico do Elbow Method para definir o valor ideal de K para o caso de o modelo ser o KNearest Neighbors
    Args:
      model (SciKit-Learn Model): modelo a ser treinado
      X (pd.DataFrame): features a serem treinadas
      y (pd.DataFrame): target do dataset
         
    """
    erro = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        erro.append(kmeans.inertia_)

    #Plotagem da figura
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), erro, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de clusters')
    plt.ylabel('Erro')
    plt.title('Elbow Method')
    plt.xticks(range(1, 11))
    plt.grid(True)

    #Salvar figura
    plt.savefig(path + "KNN_Elbow_Method.png")
    plt.close()