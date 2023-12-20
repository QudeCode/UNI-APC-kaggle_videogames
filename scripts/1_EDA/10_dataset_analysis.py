import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import sys

# Rutas de los archivos de entrada y salida
input_file_path = 'data/Video_Games_Sales_as_at_22_Dec_2016.csv'
correlation_matrix_fig_path = 'results/1_EDA/10_first_correlation_matrix.png'
analysis_txt_path = 'results/1_EDA/10_dataset_analysis.txt'

# Cargar el conjunto de datos original
data = pd.read_csv(input_file_path)

# Columnas numéricas para la matriz de correlación, el resto después de preprocessing
numeric_columns_of_interest = ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                               'Critic_Score', 'Critic_Count', 'User_Count']

# Filtrar el conjunto de datos para incluir solo las columnas numéricas de interés
numeric_data_subset = data[numeric_columns_of_interest]

# Calcular la matriz de correlación para las columnas numéricas
correlation_matrix_numeric = numeric_data_subset.corr()

# Crear el gráfico de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_numeric, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación (Columnas Numéricas)')
plt.savefig(correlation_matrix_fig_path)
plt.show()

# Guardar el análisis de tamaño y tipología de datos en un archivo de texto
with open(analysis_txt_path, 'w') as analysis_txt:
    sys.stdout = analysis_txt
    data.info()
    sys.stdout = sys.__stdout__
