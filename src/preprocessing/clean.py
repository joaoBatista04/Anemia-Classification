import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder

def missing_values_plot_describe(df):
    """
    Descreve se o dataframe possui valores nulos ou NaN
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
         
    """
   
    with open("./assets/preprocessing_info/missing_data.txt", "w") as file:
        file.write(tabulate(pd.DataFrame(df.isna().sum())))
        file.write(tabulate(pd.DataFrame(df.isnull().sum())))

def describe_data(df, filename):
    """
    Descreve os dados de uma forma geral, como parâmetros como máximo, mínimo, média, quartis, desvio padrão, tipo, etc.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      filename (string): Nome do arquivo a ser salvo.
         
    """
    with open(f"./assets/preprocessing_info/{filename}", "w") as file:
        for key, value in df.dtypes.items():
            file.write(f"{key}: {value}\n")
        
        file.write("\n")
        for item in df.columns:
           file.write(item + "\t\t")
        file.write("\n")

        file.write(tabulate(df.describe()))

def remove_negative_values(df, columns):
    """
    Remove os valores negativos das colunas que forem passadas como argumento da função
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      columns (list): Lista de colunas a serem removidas por possuírem valores negativos
         
    """
    for column in columns:
        df = df.drop(df[df[column] < 0].index[0])

    return df

def remove_duplicated_data(df):
    """
    Remove valores duplicados do dataset
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
         
    """
    data = df.drop_duplicates()
    return data

def encoding_categorical_to_numeric(df, target):
    """
    Faz a transformação da variável target de valor categórico para valor numérico
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      target (string): Nome da coluna a ser transformada de categórica para numérica
         
    """
    
    #Dicionário de transformação
    mapping_dict = {}

    #Utilização de LabelEncoder
    encoder = LabelEncoder()
    encoded_data = encoder.fit_transform(df[target])
    mapping_dict[target] = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
    df[target] = encoded_data
    
    return df, mapping_dict

def remove_outliers(df):
    """
    Remove valores do dataset que sejam maiores do que 3 vezes o valor do desvio padrão da coluna onde estão armazenados. Dados nessas situações são considerados pela literatura como outliers.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
         
    """

    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < 3).all(axis=1)
    filtered_data = df[filtered_entries]
    filtered_data

    return filtered_data