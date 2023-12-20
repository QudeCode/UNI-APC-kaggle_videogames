import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
input_file_path = 'data/train_DEF.csv'
output_txt_path = 'results/4_ModelSelection_and_Crossvalidation/40_linear_regressions.txt'
output_file_path = 'results/4_ModelSelection_and_Crossvalidation/40_linear_regression_plot.png'
data = pd.read_csv(input_file_path)

# Seleccionar características y variable objetivo
exclude_columns = ['Critic_Score', 'User_Count', 'Critic_Count', 'User_Score']
X = data.drop(columns=exclude_columns)
y = data['Critic_Score']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión lineal
linear_model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
linear_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = linear_model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir resultados
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Guardar resultados en un archivo de texto
with open(output_txt_path, 'w') as f:
    f.write(f'Mean Squared Error: {mse}\n')
    f.write(f'R-squared: {r2}\n')

# Visualizar la regresión lineal simple
plt.scatter(y_test, y_pred, color='black')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle='--', color='red', linewidth=2)
plt.xlabel('Critic_Score reales')
plt.ylabel('Critic_Score predichos')
plt.title('Predicciones vs. Critic_Score reales')
plt.savefig(output_file_path)
plt.show()
