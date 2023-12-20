import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Cargar el conjunto de datos
input_file_path = 'data/train_DEF.csv'
output_file_path = "results/4_ModelSelection_and_Crossvalidation/40_regression_critic_results.png"
results_txt_path = "results/4_ModelSelection_and_Crossvalidation/41_regression_results.txt"
data = pd.read_csv(input_file_path)

# Seleccionar características y variables objetivo
X = data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
          'Other_Sales', 'Global_Sales', 'User_Count']]
y_critic = data['Critic_Score']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_critic_train, y_critic_test = train_test_split(
    X, y_critic, test_size=0.2, random_state=42
)

# Imputar valores NaN con -1 en los conjuntos de datos de entrenamiento y prueba
imputer = SimpleImputer(strategy='constant', fill_value=-1)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Inicializar modelos
models = {
    'LR': LinearRegression(),
    'DTR': DecisionTreeRegressor(),
    'RFR': RandomForestRegressor(),
    'XGBR': XGBRegressor(),
    'GBR': GradientBoostingRegressor(),
    'ABR': AdaBoostRegressor()
}

# Almacenar los resultados en un archivo de texto
with open(results_txt_path, 'w') as f:
    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train_imputed, y_critic_train, cv=5)

        f.write(f"Model: {model_name}\n")
        f.write(f"Cross-Validation Scores : {cv_scores}\n")
        f.write(f"Average : {np.mean(cv_scores)}\n")
        f.write('---------------------------------------\n')

    # Seleccionar y ajustar el mejor modelo para Critic_Score
    # Puedes seleccionar el modelo con el mejor rendimiento
    best_model_critic = RandomForestRegressor()
    best_model_critic.fit(X_train_imputed, y_critic_train)

    # Realizar predicciones en el conjunto de prueba para Critic_Score
    y_critic_pred = best_model_critic.predict(X_test_imputed)

    # Evaluar el rendimiento del modelo para Critic_Score
    mse_critic = mean_squared_error(y_critic_test, y_critic_pred)
    r2_critic = r2_score(y_critic_test, y_critic_pred)

    f.write(f'Mean Squared Error (Critic_Score): {mse_critic}\n')
    f.write(f'R-squared (Critic_Score): {r2_critic}\n')
