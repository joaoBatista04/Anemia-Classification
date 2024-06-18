import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

def plot_multiclass_roc(X, y, classes, pipeline, path, name, test_size=0.2, random_state=42):
    """
    Plota a curva ROC para classificação multiclasse usando um pipeline.

    Parâmetros:
    - X: DataFrame ou array das features
    - y: Array das labels
    - classes: Array das classes únicas
    - pipeline: Pipeline a ser usado (o último passo deve ser um classificador)
    - path: Caminho onde o arquivo será salvo
    - name: Nome do arquivo a ser salvo
    - test_size: Proporção do conjunto de teste
    - random_state: Estado aleatório para reprodutibilidade
    """

    #Binarizar a saída
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]

    #Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=test_size, random_state=random_state)

    # Treinar o pipeline usando One-vs-Rest
    clf = OneVsRestClassifier(pipeline)
    clf.fit(X_train, y_train)

    #Prever as probabilidades
    y_score = clf.predict_proba(X_test)

    #Calcular a curva ROC e a AUC para cada classe
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #Plotar a curva ROC para cada classe
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Classe {0} (área = {1:0.2f})'.format(classes[i], roc_auc[i]))

    #Detalhes da figura
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC Multiclasse')
    plt.legend(loc="lower right")

    #Salvar a figura
    plt.savefig(path + name)
    plt.close()


def evaluate_all_metrics(y_test, y_pred, cross_val_acc, stratified_kfold_acc, path):
    """
    Avalia as métricas de desempenho de um modelo, calculando acurácia, precisão, recall e F1-Score

    Parâmetros:
    - y_test: Valores verdadeiros dos rótulos
    - y_pred: Valores preditos para os rótulos
    - cross_val_acc: Valor da acurácia do modelo para validação cruzada com cross_val_score
    - stratified_kfold_acc: Valor da acurácia do modelo para validação cruzada com Stratified KFold
    - path: Caminho onde o arquivo deverá ser salvo
    """

    #Cálculo das métricas de desempenho
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    comment = f"Accuracy: {100 * acc:.2f}%, Precision: {100 * pre:.2f}%, F1-Score: {100* f1:.2f}%, Recall: {100 * rec:.2f}%"

    #Salvando o arquivo com as informações
    with open(path + "Metrics.txt", "w") as file:
        file.write(comment)
        file.write(f"\nAccuracy with cross validation with cross_val_score: {cross_val_acc}%")
        file.write(f"\nAccuracy with cross validation with Stratified KFold: {stratified_kfold_acc}%")

def plot_confusion_matrix(y_test, y_pred, model, path):
    """
    Plota uma matriz de confusão.

    Parâmetros:
    - y_test: Valores verdadeiros dos rótulos
    - y_pred: Valores preditos para os rótulos
    - model: Modelo a ser utilizado para a construção da matriz de confusão
    - path: Caminho onde o arquivo deverá ser salvo
    """

    #Calculando a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    #Normalizando a matriz de confusão para porcentagens
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    #Configurando a visualização
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Rótulo Previsto')
    plt.ylabel('Rótulo Verdadeiro')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')

    #Salvar a figura
    plt.savefig(path + f'ConfusionMatrix{model.__class__.__name__}.png')
    plt.close()