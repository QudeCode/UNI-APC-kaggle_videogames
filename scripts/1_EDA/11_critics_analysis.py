import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas de los archivos de entrada y salida
input_file_path = 'data/Video_Games_Sales_as_at_22_Dec_2016.csv'
output_fig_path = 'results/1_EDA/11_critics_balance_analysis.png'

# Cargar el conjunto de datos original
data = pd.read_csv(input_file_path)

# Seleccionar las columnas de interés para el análisis de críticos
critics_columns_of_interest = ['Critic_Score',
                               'Critic_Count', 'User_Score', 'User_Count']
critics_data_subset = data[critics_columns_of_interest]

# Visualizar el balance de cada atributo
plt.figure(figsize=(15, 8))
for i, column in enumerate(critics_data_subset.columns, 1):
    plt.subplot(2, 2, i)

    # Establecer rangos específicos para 'User_Score' para mejorar la visualización
    if column == 'User_Score':
        # Crear una columna específica para "tbd" y cambiar su valor a NaN
        critics_data_subset['User_Score_no_tbd'] = pd.to_numeric(
            data['User_Score'], errors='coerce')
        sns.histplot(
            critics_data_subset['User_Score_no_tbd'], kde=True, bins=range(1, 12))
    else:
        sns.histplot(critics_data_subset[column], kde=True)

    plt.title(f'Distribución de {column}')

plt.suptitle('Balance de Atributos de Críticos')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_fig_path)
plt.show()
