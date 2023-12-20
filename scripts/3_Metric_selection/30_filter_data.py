import pandas as pd

# Rutas de los archivos de entrada y salida
input_train_path = 'data/train_tracted.csv'
output_filtered_path = 'data/train_DEF.csv'
results_file_path = 'results/3_Metric_selection/30_filter_data.txt'

# Cargar conjunto de datos de entrenamiento
train_data = pd.read_csv(input_train_path)

# Eliminar las columnas con correlación baja
columns_to_drop = ['Platform', 'Year_of_Release',
                   'Genre', 'Rating']
filtered_data = train_data.drop(columns=columns_to_drop)

# Guardar conjunto de datos filtrado
filtered_data.to_csv(output_filtered_path, index=False)

# Crear mensaje de resultados
result_message = (
    "1. Eliminación de columnas con correlación baja:\n"
    f"   - Se han eliminado las columnas {columns_to_drop} debido a su baja correlación con las críticas."
)

# Imprimir resultados en la consola
print(result_message)

# Guardar resultados en el archivo results/2_Preprocessing/23_filter_data.txt
with open(results_file_path, 'w') as results_file:
    results_file.write(result_message)
