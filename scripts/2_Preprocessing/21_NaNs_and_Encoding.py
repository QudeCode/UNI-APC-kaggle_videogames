import os
import pandas as pd

# Rutas de los archivos de entrada y salida
input_train_path = 'data/train.csv'
output_train_path = 'data/train_tracted.csv'
results_file_path = 'results/2_Preprocessing/21_NaNs_and_Encoding.txt'
encodings_dir = 'data/encodings'

# Crear el directorio de encodings si no existe
os.makedirs(encodings_dir, exist_ok=True)

# Cargar conjunto de datos de entrenamiento
train_data = pd.read_csv(input_train_path)

# Eliminar la columna 'Name'
train_data = train_data.drop(columns=['Name'])

# Convertir "tbd" a NaN en la columna "User_Score"
train_data['User_Score'] = pd.to_numeric(
    train_data['User_Score'], errors='coerce')

# Contar valores nulos por columna antes del tratamiento
null_counts_before = train_data.isnull().sum()

# Reemplazar valores nulos con "-1" en todas las columnas
train_data = train_data.fillna(-1)

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
    "1. Eliminación de la columna 'Name':\n"
    "   - Se ha eliminado la columna 'Name'.\n\n"
    "2. Conversión de 'tbd' a NaN en la columna 'User_Score':\n"
    "   - Se han convertido los valores 'tbd' a NaN en la columna 'User_Score'.\n\n"
    "3. Tratamiento de valores nulos:\n"
    f"   - Se han reemplazado los nulos con '-1' en todas las columnas.\n\n"
    "4. Asignacion de valores numericos a la columna 'Rating':\n"
    "   - Se ha asignado un valor numerico a cada posible 'Rating' segun el siguiente mapeo:\n"
    f"     {rating_mapping}\n\n"
    f"5. Conteo de valores nulos por columna antes del tratamiento:\n{null_counts_before}\n\n"
    f"6. Conteo de valores nulos por columna despues del tratamiento:\n{train_data.isnull().sum()}\n\n"
    "7. Diccionarios de codificación:\n"
)

# Agregar información de diccionarios de codificación al mensaje de resultados
for column, encoding_dict in encoding_dicts.items():
    result_message += f"\n{column}:\n{encoding_dict}\n"

# Imprimir resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/2_Preprocessing/21_NaNs_and_Encoding.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
