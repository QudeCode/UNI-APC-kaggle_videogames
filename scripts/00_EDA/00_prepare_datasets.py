import pandas as pd
from sklearn.model_selection import train_test_split

# Rutas de los archivos de entrada y salida
input_file_path = '../data/Video_Games_Sales_as_at_22_Dec_2016.csv'
output_train_path = '../data/train.csv'
output_test_path = '../data/test.csv'
results_file_path = '../results/00_EDA/00_initialization.txt'

# Cargar el conjunto de datos original
data = pd.read_csv(input_file_path)

# Separar el conjunto de datos en entrenamiento y prueba (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Guardar los conjuntos de datos en archivos CSV
train_data.to_csv(output_train_path, index=False)
test_data.to_csv(output_test_path, index=False)

# Calcular porcentaje de filas en entrenamiento y prueba (debería ser el estandar de 80 - 20)
percentage_train = len(train_data) / len(data) * 100
percentage_test = len(test_data) / len(data) * 100

# Crear el mensaje de resultados
result_message = (
    f'Datos originales: {len(data)} filas\n'
    f'Datos de entrenamiento: {len(train_data)} filas ({percentage_train:.2f}%)\n'
    f'Datos de prueba: {len(test_data)} filas ({percentage_test:.2f}%)'
)

# Mostrar resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/00_initialization.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
