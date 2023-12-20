import os
import pandas as pd

# Rutas de los archivos de entrada y salida
input_train_path = 'data/train.csv'
output_train_path = 'data/train_tracted.csv'
results_file_path = 'results/2_Preprocessing/21_NaNs_and_Encoding.txt'
encodings_dir = 'data/encodings'

# Cargar conjunto de datos de entrenamiento
train_data = pd.read_csv(input_train_path)

# Eliminar la columna 'Name'
train_data = train_data.drop(columns=['Name'])

# Convertir "tbd" a NaN en la columna "User_Score"
train_data['User_Score'] = pd.to_numeric(
    train_data['User_Score'], errors='coerce')

# Contar valores nulos por fila antes del tratamiento
null_counts_per_row_before = train_data.isnull().sum(axis=1)

# Contar el total de filas con al menos un NaN antes del tratamiento
total_rows_with_nans_before = (null_counts_per_row_before > 0).sum()

# Obtener la muestra total (número total de filas antes del tratamiento)
total_rows_before = len(train_data)

# Eliminar filas con al menos un NaN
train_data = train_data.dropna()

# Obtener la muestra total (número total de filas después del tratamiento)
total_rows_after = len(train_data)

# Contar valores nulos por fila después del tratamiento
null_counts_per_row_after = train_data.isnull().sum(axis=1)

# Contar el total de filas con al menos un NaN después del tratamiento
total_rows_with_nans_after = (null_counts_per_row_after > 0).sum()

# Crear diccionarios para el encoding de cada columna categórica
encoding_dicts = {}

# Iterar sobre columnas categóricas
categorical_columns = ['Platform', 'Genre', 'Publisher', 'Developer']
for column in categorical_columns:
    # Filtrar valores diferentes de -1 antes del encoding
    values_for_encoding = train_data[train_data[column] != -1][column]
    # Crear un diccionario de codificación
    encoding_dict = {value: idx + 1 for idx,
                     value in enumerate(values_for_encoding.unique())}
    # Agregar el diccionario al objeto general
    encoding_dicts[column] = encoding_dict
    # Aplicar el encoding a la columna en el conjunto de datos
    train_data[column] = train_data[column].map(encoding_dict)

    # Guardar el diccionario de codificación en un archivo CSV
    encoding_df = pd.DataFrame(list(encoding_dict.items()), columns=[
                               'Original', f'id_{column}'])
    encoding_df.to_csv(os.path.join(
        encodings_dir, f'{column}_encoding.csv'), index=False)

# Definir mapeo de Rating a valores numéricos
rating_mapping = {'E': 0, 'T': 1, 'M': 2, 'E10+': 3, 'AO': 4, 'RP': 5}

# Convertir la columna "Rating" a tipo de datos str
train_data['Rating'] = train_data['Rating'].astype(str)

# Asignar valores numéricos según el mapeo, excluyendo los valores nulos
train_data['Rating'] = train_data['Rating'].apply(
    lambda x: rating_mapping.get(x, -1))

# Guardar conjunto de datos tratado
train_data.to_csv(output_train_path, index=False)

# Crear mensaje de resultados
result_message = (
    "1. Eliminacion de la columna 'Name':\n"
    "   - Se ha eliminado la columna 'Name'.\n\n"
    "2. Conversion de 'tbd' a NaN en la columna 'User_Score':\n"
    "   - Se han convertido los valores 'tbd' a NaN en la columna 'User_Score'.\n\n"
    "3. Tratamiento de valores nulos:\n"
    f"   - Se han eliminado todas las filas que contenian algun NaN:\n\n"
    f"4. Filas con al menos un NaN antes del tratamiento: {total_rows_with_nans_before}\n"
    f"    - Numero muy inferior a la muestra total ({total_rows_before})\n\n"
    f"5. Filas con al menos un NaN después del tratamiento: {total_rows_with_nans_after}\n\n"
    f"6. Filas restantes en train: {total_rows_after}\n\n"
    "7. Diccionarios de codificacion:\n"
)

# Agregar información de diccionarios de codificación al mensaje de resultados
for column, encoding_dict in encoding_dicts.items():
    result_message += f"\n{column}:\n{encoding_dict}\n"

# Imprimir resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/2_Preprocessing/21_NaNs_and_Encoding.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
