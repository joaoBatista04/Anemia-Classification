import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import models_train

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MultiModelEvaluate:
    """
    Classe para avaliação de múltiplos modelos e busca pelos melhores e mais otimizados hiperparâmetros para esses modelos
         
    """
    
    def __init__(self, df, path):
        self.path = path
        
        self.X = df.drop(columns=["Diagnosis"])
        self.y = df["Diagnosis"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

        self.models = models_train.models
        self.param_grids = models_train.param_grids

        self.pipelines = self.create_pipelines()

        self.best_models = {}

    def create_pipelines(self):
        """
        Cria o pipeline padrão com Standard Scaler e os modelos que foram definidos no arquivo de modelos e métricas
            
        """
        return {
            name: Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            for name, model in self.models.items()
        }
    
    def find_best_models(self):
        """
        Aplicação direta do GridSearchCV para definir os melhores hiperparâmetros para cada modelo
            
        """
        with open(self.path + "best_models.txt", "w") as file:
            file.write("Results GridSearchCV:\n")

        #Aplicação do GridSearchCV
        with open(self.path + "best_models.txt", "a") as file:
            for name, pipeline in self.pipelines.items():
                grid_search = GridSearchCV(
                    pipeline,
                    self.param_grids[name],
                    scoring='accuracy',
                    refit='mse',
                    cv=KFold(n_splits=2,shuffle=True,random_state=42),
                    error_score='raise'
                )

                grid_search.fit(self.X_train, self.y_train)
                self.best_models[name] = grid_search.best_estimator_
                
                #Registro de resultados em arquivo
                file.write(str(self.best_models[name]) + "\n\n")

    def evaluate_models(self):
        """
        Avaliação de métricas como acurácia, precisão, recall e F1-Score para os diferentes modelos de predição
            
        """
        results = []
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

        for name, model in self.best_models.items():
            predicted_values = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, predicted_values)
            precision = precision_score(self.y_test, predicted_values, average='weighted', zero_division=1)
            recall = recall_score(self.y_test, predicted_values, average='weighted')
            f1 = f1_score(self.y_test, predicted_values, average='weighted')

            results.append([name, accuracy, precision, recall, f1])

        with open(self.path + "evaluations.txt", "w") as file:
            file.write(tabulate(results, headers=headers, tablefmt="pretty"))