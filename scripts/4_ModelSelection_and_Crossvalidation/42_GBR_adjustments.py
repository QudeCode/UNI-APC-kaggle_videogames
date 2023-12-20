from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Cargar el conjunto de datos
input_file_path = 'data/train_DEF.csv'
output_dir = "results/4_ModelSelection_and_Crossvalidation/"
results_txt_path = output_dir + "42_GBR_results.txt"
data = pd.read_csv(input_file_path)

# Seleccionar características y variables objetivo
X = data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
          'Other_Sales', 'Global_Sales', 'User_Count']]
y_critic = data['Critic_Score']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_critic_train, y_critic_test = train_test_split(
    X, y_critic, test_size=0.2, random_state=42
)

# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Inicializar el modelo
gbr = GradientBoostingRegressor()

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(
    gbr, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_critic_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_

# Obtener el mejor modelo
best_gbr = grid_search.best_estimator_

# Realizar predicciones y evaluar el rendimiento en el conjunto de prueba
y_pred_test = best_gbr.predict(X_test)
mse_test = mean_squared_error(y_critic_test, y_pred_test)
r2_test = r2_score(y_critic_test, y_pred_test)

# Guardar los resultados en el archivo de texto
with open(results_txt_path, 'w') as f:
    f.write("Mejores hiperparámetros:\n")
    f.write(str(best_params) + "\n\n")
    f.write(f'Mean Squared Error (Test): {mse_test}\n')
    f.write(f'R-squared (Test): {r2_test}\n')
