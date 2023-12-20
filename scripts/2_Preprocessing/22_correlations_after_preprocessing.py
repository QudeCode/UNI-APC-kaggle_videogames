import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas de los archivos de entrada y salida
input_filtered_path = 'data/train_tracted.csv'
correlation_matrix_fig_path = 'results/2_Preprocessing/22_correlations_after_preprocessing_full.png'

# Cargar conjunto de datos filtrado
filtered_data = pd.read_csv(input_filtered_path)

# Calcular la matriz de correlación para todas las columnas
correlation_matrix_full = filtered_data.corr()

# Crear el gráfico de la matriz de correlación
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix_full, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación Después de Preprocesamiento (Completa)')
plt.savefig(correlation_matrix_fig_path)
plt.show()
