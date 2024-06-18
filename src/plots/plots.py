import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_grid(df, columns, path, file_name,  bins=10):
    """
    Cria uma grade de histogramas com Kernel Density Estimation (KDE) para visualização da distribuição de múltiplas colunas.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      columns (list): Lista de nomes das colunas a serem exibidas no histograma.
      bins (int): Número de bins (intervalos) utilizados no histograma.
         
    """
    num_rows = len(columns) // 3 + (1 if len(columns) % 3 != 0 else 0)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    #Plotagem dos histogramas
    for i, column in enumerate(columns):
        row = i // num_cols
        col = i % num_cols
        sns.histplot(df[column], kde=True, bins=bins, ax=axes[row, col], color="blue")
        axes[row, col].set_title(f'Distribuição de {column}')
    
    #Remover subplots não utilizados
    for i in range(len(columns), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    #Salvar figura
    plt.tight_layout()
    plt.savefig(path + f"{file_name}")
    plt.close()

def plot_boxplot_grid(df, columns, path, filename):
    """
    Cria uma grade de boxplots para visualização da distribuição de múltiplas colunas.

    Args:
        df (pandas.DataFrame): DataFrame contendo os dados.
        columns (list): Lista de nomes das colunas a serem exibidas no boxplot.
    """

    num_rows = len(columns) // 3 + (1 if len(columns) % 3 != 0 else 0)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    #Plotar os boxplots
    for i, column in enumerate(columns):
        row = i // num_cols
        col = i % num_cols
        sns.boxplot(
            x = df[column],
            ax = axes[row, col],
            color="blue"
        )
        axes[row, col].set_title(f'Boxplot de {column}')

    #Remover subplots não utilizados
    for i in range(len(columns), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    #Salvar figura
    plt.tight_layout()
    plt.savefig(path + f"{filename}")
    plt.close()

def correlation_matrix(df, path, filename):
    """
    Produz uma matriz de correlação entre as features do dataset a partir do cálculo dos Coeficientes de Pearson
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      path (string): Caminho para onde o arquivo será salvo
      filename (string): Nome do arquivo
    """
    #Criação e plotagem da matriz de correlação
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.savefig(path + filename)
    plt.close()

def dist_for_each_anamemia_type(df, columns, target, path, bins=10):
    """
    Plota a distribuição dos dados de cada coluna para todos os tipos de classificação de anemia.
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      columns (list): Lista de nomes das colunas a serem exibidas no histograma.
      target (string): Nome da variável alvo
      path (string): Caminho onde o arquivo será salvo
      bins (int): Número de bins (intervalos) utilizados no histograma.
         
    """
    
    colors = ['blue', 'red', 'green', 'magenta', 'yellow', 'cyan', 'white', 'black', 'gray']
    num_rows = len(columns) // 3 + (1 if len(columns) % 3 != 0 else 0)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    df_filtered = df[df['Diagnosis'] == target]
    
    #Plotagem dos histogramas
    for i, column in enumerate(columns):
        row = i // num_cols
        col = i % num_cols
        sns.histplot(df_filtered[column], kde=True, bins=bins, ax=axes[row, col], color=colors[target])
        axes[row, col].set_title(f'Distribuição de {column}')
    
    #Remover subplots não utilizados
    for i in range(len(columns), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    #Salvar figura
    plt.tight_layout()
    plt.savefig(path + f"Dist{target}.png")
    plt.close()

def plot_distribution_target(df, target, path):
    """
    Faz a plotagem do gráfico de distribuição da variável alvo, no caso, a coluna Diagnosis
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      target (string): Nome da variável alvo do dataset.
      path (string): Caminho onde o arquivo será salvo
         
    """

    #Plotagem do gráfico
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, y=target, palette='plasma', width=0.6)

    #Definição das labels e outras informações
    ax.set_title('Distribuição dos Diagnósticos', fontsize=16, fontweight='bold')
    ax.set_xlabel('Contagem', fontsize=14)
    ax.set_ylabel('Diagnóstico', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlim(0, 390)

    for p in ax.patches:
        ax.annotate(f'{p.get_width()}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, color='black', xytext=(12, 0),
                    textcoords='offset points')

    #Salvar figura
    plt.savefig(path + target + "_Distribution.png")
    plt.close()

def boxplot_to_visualize_outliers(df, target, name, path):
    """
    Faz a plotagem dos boxplots de distribuição dos dados das colunas do dataset
    
    Args:
      df (pandas.DataFrame): DataFrame contendo os dados.
      target (string): Nome da variável alvo do dataset.
      name (string): Nome do arquivo
      path (string): Caminho onde o arquivo será salvo
         
    """

    #Plotagem dos boxplots
    df.drop(target, axis = 1).plot(kind = 'box' ,figsize=(15,10), )
    plt.ylim(0, 1050)
    plt.suptitle ('Display of Box and Whisker Chart', size=16)

    #Salvar figura
    plt.savefig(path + name + ".png")
    plt.close()