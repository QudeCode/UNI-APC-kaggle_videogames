import pandas as pd

# Rutas de los archivos de entrada y salida
input_train_path = '../data/train.csv'
output_train_path = '../data/train_tracted.csv'
results_file_path = '../results/00_EDA/01_NaNs_and_Encoding.txt'

# Cargar conjunto de datos de entrenamiento
train_data = pd.read_csv(input_train_path)

# Contar valores nulos por columna antes del tratamiento
null_counts_before = train_data.isnull().sum()

# Reemplazar valores nulos con "-1" en todas las columnas
train_data = train_data.fillna(-1)

# Definir mapeo de Rating a valores numéricos
rating_mapping = {'E': 0, 'T': 1, 'M': 2, 'E10+': 3, 'AO': 4, 'RP': 5}

# Convertir la columna "Rating" a tipo de datos str
train_data['Rating'] = train_data['Rating'].astype(str)

# Asignar valores numéricos según el mapeo, excluyendo los valores nulos
train_data['Rating'] = train_data['Rating'].apply(
    lambda x: rating_mapping.get(x, -1))

# Contar valores nulos por columna después del tratamiento
null_counts_after = train_data.isnull().sum()

# Guardar conjunto de datos tratado
train_data.to_csv(output_train_path, index=False)

# Crear mensaje de resultados
result_message = (
    "1. Tratamiento de valores nulos:\n"
    f"   - Se han reemplazado los nulos con '-1' en todas las columnas.\n\n"
    "2. Asignacion de valores numericos a la columna 'Rating':\n"
    "   - Se ha asignado un valor numerico a cada posible 'Rating' segun el siguiente mapeo:\n"
    f"     {rating_mapping}\n\n"
    f"3. Conteo de valores nulos por columna antes del tratamiento:\n{null_counts_before}\n\n"
    f"4. Conteo de valores nulos por columna despues del tratamiento:\n{null_counts_after}"
)

# Imprimir resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/00_EDA/01_NaNs_and_Encoding.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
