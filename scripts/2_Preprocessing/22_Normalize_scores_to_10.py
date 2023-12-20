import pandas as pd

# Rutas de entrada y salida
file_path = 'data/train_tracted.csv'

# Cargar conjunto de datos
data = pd.read_csv(file_path)

# Normalizar la columna 'Critic_Score' de 0 a 10
data['Critic_Score'] = data['Critic_Score'] / 10.0

# Guardar el conjunto de datos normalizado
data.to_csv(file_path, index=False)

print("Critic_Score normalized and saved to", file_path)
