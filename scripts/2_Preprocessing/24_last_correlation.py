import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas de los archivos de entrada y salida
input_filtered_path = 'data/train_DEF.csv'
correlation_fig_path = 'results/2_Preprocessing/24_last_correlation.png'

# Cargar conjunto de datos filtrado
filtered_data = pd.read_csv(input_filtered_path)

# Calcular la matriz de correlación
correlation_matrix = filtered_data.corr()

# Crear el gráfico de la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación (Conjunto de Datos Filtrado)')
plt.savefig(correlation_fig_path)
plt.show()
