import pandas as pd

from preprocessing import clean
from plots import plots

class Initial_Exploratory_Analysis:
    def __init__(self, df):
        self.dataframe = df

    def run_initial_analysis(self):
        #DESCRIÇÃO DOS DADOS E VERIFICAÇÃO DE VALORES NULOS
        clean.missing_values_plot_describe(self.dataframe)
        clean.describe_data(self.dataframe, "data_info.txt")

        #REMOÇÃO DE VALORES DUPLICADOS
        self.dataframe = clean.remove_duplicated_data(self.dataframe)

        #REMOÇÃO DE COLUNAS COM VALORES NEGATIVOS
        columns_with_negative = ["HGB", "MCV"]
        self.dataframe = clean.remove_negative_values(self.dataframe, columns_with_negative)

        #LABEL ENCODING PARA TRANSFORMAR VARIÁVEIS CATEGÓRICAS PARA VARIÁVEIS NUMÉRICAS
        self.dataframe, mapping_dict = clean.encoding_categorical_to_numeric(self.dataframe, "Diagnosis")

        #ANÁLISE DE DISTRIBUIÇÃO DOS DADOS
        columns = self.dataframe.columns.to_list()
        columns.remove('Diagnosis')
        #plots.plot_distribution_grid(self.dataframe, columns, "./assets/distributions/", "data_distribution_before.png", bins=50)
        #plots.plot_boxplot_grid(self.dataframe, columns, "./assets/distributions/", "data_quartiles_before.png")

        #ANÁLISE DE CORRELAÇÃO
        #plots.correlation_matrix(self.dataframe, "./assets/distributions/", "correlation_matrix.png")

        #DISTRIBUIÇÃO DE CADA COLUNA DO DATASET
        targets = self.dataframe["Diagnosis"].unique()
        #for target in targets:
            #plots.dist_for_each_anamemia_type(self.dataframe, columns, target, "./assets/distributions/", bins=50)

        #DISTRIBUIÇÃO DA VARIÁVEL TARGET
        plots.plot_distribution_target(self.dataframe, "Diagnosis", "./assets/distributions/")

        #PLOTAGEM DE BOXPLOTS PARA VERIFICAÇÃO DE OUTLIERS
        plots.boxplot_to_visualize_outliers(self.dataframe, "Diagnosis", "Outliers_Before", "./assets/distributions/")

        #REMOÇÃO DE OUTLIERS
        self.dataframe = clean.remove_outliers(self.dataframe)

        #VERIFICAÇÃO DOS DADOS APÓS REMOÇÃO DOS OUTLIERS
        plots.boxplot_to_visualize_outliers(self.dataframe, "Diagnosis", "Outliers_After", "./assets/distributions/")
        #clean.describe_data(self.dataframe, "data_info_after.txt")

        #NOVA DISTRIBUIÇÃO DE DADOS APÓS REMOÇÃO DOS OUTLIERS
        #plots.plot_distribution_grid(self.dataframe, columns, "./assets/distributions/", "data_distribution_after.png", bins=50)

        #SALVANDO NOVA BASE DE DADOS
        self.dataframe.to_csv("./data/clean_data/clean_data.csv")

        return self.dataframe