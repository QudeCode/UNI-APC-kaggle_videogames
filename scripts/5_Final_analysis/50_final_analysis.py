import pandas as pd
from sklearn import impute
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

output_txt_path = 'results/5_Final_analysis/51_final_results.txt'
result_csv_path_users = 'results/5_Final_analysis/52_GBR_predictions_test_user.csv'
result_csv_path_critic = 'results/5_Final_analysis/51_GBR_predictions_test_critic.csv'

# Cargar el conjunto de datos de entrenamiento
train_file_path = 'data/train_DEF.csv'
train_data = pd.read_csv(train_file_path)

# Cargar el conjunto de datos de prueba
test_file_path = 'data/test.csv'
test_data = pd.read_csv(test_file_path)

# Aplicar encoding a las variables categóricas en el conjunto de entrenamiento
encoding_dir = 'data/encodings/'
publisher_encoding = pd.read_csv(encoding_dir + 'Publisher_encoding.csv')

test_data['Publisher'] = test_data['Publisher'].map(
    publisher_encoding.set_index('Original')['id_Publisher'])

# Seleccionar características y variables objetivo para el conjunto de entrenamiento
X_train = train_data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
                      'Other_Sales', 'Global_Sales', 'User_Count']]
y_critic_train = train_data['Critic_Score']
y_user_train = train_data['User_Score']

# Seleccionar características para el conjunto de prueba
X_test = test_data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
                    'Other_Sales', 'Global_Sales', 'User_Count']]

# Crear y entrenar el modelo con los mejores hiperparámetros para Critic_Score
best_gbr_critic = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_leaf=4,
                                            min_samples_split=2, n_estimators=100, subsample=0.8)
best_gbr_critic.fit(X_train, y_critic_train)

# Realizar predicciones en el conjunto de entrenamiento para Critic_Score
y_pred_train_critic = best_gbr_critic.predict(X_train)

# Evaluar el rendimiento del modelo en el conjunto de entrenamiento para Critic_Score
mse_train_critic = mean_squared_error(y_critic_train, y_pred_train_critic)
r2_train_critic = r2_score(y_critic_train, y_pred_train_critic)

# Imputar valores NaN con la media en el conjunto de datos de prueba
# Imputar valores NaN con la media en el conjunto de datos de prueba
imputer_test = SimpleImputer(strategy='mean')
X_test_imputed = imputer_test.fit_transform(X_test)


# Realizar predicciones en el conjunto de prueba para Critic_Score
y_pred_test_critic = best_gbr_critic.predict(X_test_imputed)

# Guardar las predicciones en el conjunto de prueba para Critic_Score en un DataFrame
predictions_test_critic_df = pd.DataFrame(
    {'Critic_Score_Predicted': y_pred_test_critic})

# Guardar el DataFrame con las predicciones en un archivo CSV
predictions_test_critic_df.to_csv(
    result_csv_path_critic, index=False)

# Crear y entrenar el modelo con los mejores hiperparámetros para User_Score
best_gbr_user = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_leaf=4,
                                          min_samples_split=2, n_estimators=100, subsample=0.8)
best_gbr_user.fit(X_train, y_user_train)

# Realizar predicciones en el conjunto de entrenamiento para User_Score
y_pred_train_user = best_gbr_user.predict(X_train)

# Evaluar el rendimiento del modelo en el conjunto de entrenamiento para User_Score
mse_train_user = mean_squared_error(y_user_train, y_pred_train_user)
r2_train_user = r2_score(y_user_train, y_pred_train_user)

# Realizar predicciones en el conjunto de prueba para User_Score
y_pred_test_user = best_gbr_user.predict(X_test_imputed)

# Guardar las predicciones en el conjunto de prueba para User_Score en un DataFrame
predictions_test_user_df = pd.DataFrame(
    {'User_Score_Predicted': y_pred_test_user})

# Guardar el DataFrame con las predicciones en un archivo CSV
predictions_test_user_df.to_csv(result_csv_path_users, index=False)

# Guardar los resultados en un archivo de texto
with open('results/5_Final_analysis/50_GBR_results_final.txt', 'w') as f:
    f.write('Results for Critic_Score (Train):\n')
    f.write(f'Mean Squared Error (Train - Critic): {mse_train_critic}\n')
    f.write(f'R-squared (Train - Critic): {r2_train_critic}\n\n')

    f.write('Results for User_Score (Train):\n')
    f.write(f'Mean Squared Error (Train - User): {mse_train_user}\n')
    f.write(f'R-squared (Train - User): {r2_train_user}\n')

print('Results saved to:', output_txt_path)
