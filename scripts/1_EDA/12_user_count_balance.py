import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas de los archivos de entrada y salida
input_file_path = 'data/Video_Games_Sales_as_at_22_Dec_2016.csv'
output_fig_path_low = 'results/1_EDA/12_user_count_balance_low.png'
output_fig_path_high = 'results/1_EDA/12_user_count_balance_high.png'

# Cargar el conjunto de datos original
data = pd.read_csv(input_file_path)

# Seleccionar el atributo 'User_Count' sin valores NaN
user_count_data = data['User_Count'].dropna()

# Separar valores del 0 al 2000 y del 2000 al 10000
user_count_low = user_count_data[user_count_data <= 2000]
user_count_high = user_count_data[(
    user_count_data > 2000) & (user_count_data <= 10000)]

# Visualizar el balance de 'User_Count' para valores del 0 al 2000
plt.figure(figsize=(10, 6))
sns.histplot(user_count_low, bins=range(
    0, 2001, 100), edgecolor='black', kde=True)
plt.title('Distribución de User_Count (0-2000)')
plt.xlabel('User_Count')
plt.ylabel('Frecuencia')
plt.savefig(output_fig_path_low)
plt.show()

# Visualizar el balance de 'User_Count' para valores del 2000 al 10000
plt.figure(figsize=(10, 6))
sns.histplot(user_count_high, bins=range(
    2000, 10001, 1000), edgecolor='black', kde=True)
plt.title('Distribución de User_Count (2000-10000)')
plt.xlabel('User_Count')
plt.ylabel('Frecuencia')
plt.savefig(output_fig_path_high)
plt.show()
