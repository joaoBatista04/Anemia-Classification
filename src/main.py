import pandas as pd

from pipelines import SearchHyperParameters, initial
from models import model_pipeline, LazyClassifierPrediction
from models import models_train

#LEITURA DOS DADOS
df = pd.read_csv('data/diagnosed_cbc_data_v4.csv')

initial_analysis = initial.Initial_Exploratory_Analysis(df)
df = initial_analysis.run_initial_analysis()

#Verificação de possíveis melhores modelos de forma rápida e autmotizada
LazyClassifierPrediction.lazy_prediction_models(df)

#Busca dos melhores hiperparâmetros para os modelos escolhidos
mult_model_eval = SearchHyperParameters.MultiModelEvaluate(df, "./assets/models_evaluation/")
mult_model_eval.find_best_models()
mult_model_eval.evaluate_models()

#Importação dos modelos a serem treinados
models = models_train.best_models

#Treinamento e avaliação de cada modelo
for name, model in models.items():
    model_pipeline.pipeline_models_evaluations(df, model, "./assets/models_evaluation/" + name.rstrip() + "/", name + ".png")