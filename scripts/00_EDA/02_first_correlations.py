import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas de los archivos de entrada y salida
input_train_path = '../data/train_tracted.csv'
results_file_path = '../results/00_EDA/02_First_Correlations.txt'

# Cargar conjunto de datos de entrenamiento tratado
train_data = pd.read_csv(input_train_path)

# Calcular la matriz de correlación
correlation_matrix = train_data.corr()

# Graficar el mapa de calor de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor de Correlación entre Variables')
plt.savefig('../results/00_EDA/figs/01_Correlation_Heatmap.png')
plt.close()

# Crear mensaje de resultados
result_message = (
    "Primera valoracion de correlaciones entre variables:\n"
    "   - Se ha calculado la matriz de correlación entre todas las variables del conjunto de entrenamiento.\n"
    "   - Los valores nulos (-1) se han tenido en cuenta en el cálculo.\n"
    "   - Se ha generado un mapa de calor de correlación que se guarda en 'results/00_EDA/figs/01_Correlation_Heatmap.png'."
)

# Imprimir resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/00_EDA/02_First_Correlations.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
