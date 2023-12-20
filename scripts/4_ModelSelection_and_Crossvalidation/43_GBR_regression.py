import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos de entrenamiento
input_file_path = 'data/train_DEF.csv'
output_file_path_critic = 'results/4_ModelSelection_and_Crossvalidation/44_GBR_predictions_train_critic.png'
output_file_path_user = 'results/4_ModelSelection_and_Crossvalidation/45_GBR_predictions_train_user.png'

data = pd.read_csv(input_file_path)

# Seleccionar características y variables objetivo
X = data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
          'Other_Sales', 'Global_Sales', 'User_Count']]
y_critic = data['Critic_Score']
y_user = data['User_Score']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_critic_train, y_critic_test, y_user_train, y_user_test = train_test_split(
    X, y_critic, y_user, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo para las críticas de críticos con los mejores hiperparámetros
best_gbr_critic = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_leaf=4,
                                            min_samples_split=2, n_estimators=100, subsample=0.8)
best_gbr_critic.fit(X_train, y_critic_train)

# Realizar predicciones en el conjunto de entrenamiento para críticas de críticos
y_pred_train_critic = best_gbr_critic.predict(X_train)

# Evaluar el rendimiento del modelo en el conjunto de entrenamiento para críticas de críticos
mse_train_critic = mean_squared_error(y_critic_train, y_pred_train_critic)
r2_train_critic = r2_score(y_critic_train, y_pred_train_critic)

print(f'Mean Squared Error (Train - Critic): {mse_train_critic}')
print(f'R-squared (Train - Critic): {r2_train_critic}')

# Visualizar las predicciones vs. valores reales en el conjunto de entrenamiento con línea de 45 grados para críticas de críticos
predictions_train_critic_df = pd.DataFrame(
    {'Critic_Score_Real': y_critic_train, 'Critic_Score_Predicted': y_pred_train_critic})
predictions_train_critic_plot = predictions_train_critic_df.plot.scatter(
    x='Critic_Score_Real', y='Critic_Score_Predicted', title='Predicted vs. Real Critic Score (Train - Critic)')
predictions_train_critic_plot.plot([y_critic.min(), y_critic.max()], [
    y_critic.min(), y_critic.max()], '--r')
predictions_train_critic_plot.set_xlabel("Real Critic Score")
predictions_train_critic_plot.set_ylabel("Predicted Critic Score")
plt.savefig(output_file_path_critic)
plt.show()

# Crear y entrenar el modelo para las críticas de usuarios con los mejores hiperparámetros
best_gbr_user = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_leaf=4,
                                          min_samples_split=2, n_estimators=100, subsample=0.8)
best_gbr_user.fit(X_train, y_user_train)

# Realizar predicciones en el conjunto de entrenamiento para críticas de usuarios
y_pred_train_user = best_gbr_user.predict(X_train)

# Evaluar el rendimiento del modelo en el conjunto de entrenamiento para críticas de usuarios
mse_train_user = mean_squared_error(y_user_train, y_pred_train_user)
r2_train_user = r2_score(y_user_train, y_pred_train_user)

print(f'Mean Squared Error (Train - User): {mse_train_user}')
print(f'R-squared (Train - User): {r2_train_user}')

# Visualizar las predicciones vs. valores reales en el conjunto de entrenamiento con línea de 45 grados para críticas de usuarios
predictions_train_user_df = pd.DataFrame(
    {'User_Score_Real': y_user_train, 'User_Score_Predicted': y_pred_train_user})
predictions_train_user_plot = predictions_train_user_df.plot.scatter(
    x='User_Score_Real', y='User_Score_Predicted', title='Predicted vs. Real User Score (Train - User)')
predictions_train_user_plot.plot([y_user.min(), y_user.max()], [
    y_user.min(), y_user.max()], '--r')
predictions_train_user_plot.set_xlabel("Real User Score")
predictions_train_user_plot.set_ylabel("Predicted User Score")
plt.savefig(output_file_path_user)
plt.show()
