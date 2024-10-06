
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Importar el dataset
datos = pd.read_csv('datos_pandemia.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(datos.head())

# Descripción estadística del dataset
print("\nDescripción Estadística del Dataset:")
print(datos.describe())

# Visualización de los datos
plt.figure(figsize=(10,6))
sns.scatterplot(x='Movilidad', y='Casos', data=datos, color='blue')
plt.title('Relación entre Movilidad y Casos')
plt.xlabel('Movilidad')
plt.ylabel('Casos')
plt.show()

# Variables independientes y dependiente
X = datos[['Movilidad']]
y = datos['Casos']

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split, cross_val_score

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_entrenamiento_scaled = scaler.fit_transform(X_entrenamiento)
X_prueba_scaled = scaler.transform(X_prueba)

# Regresión Lineal
from sklearn.linear_model import LinearRegression

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_entrenamiento_scaled, y_entrenamiento)

# Predicciones del modelo lineal
y_pred_lineal = modelo_lineal.predict(X_prueba_scaled)

# Evaluación del modelo lineal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_lineal = mean_absolute_error(y_prueba, y_pred_lineal)
mse_lineal = mean_squared_error(y_prueba, y_pred_lineal)
r2_lineal = r2_score(y_prueba, y_pred_lineal)

print("\nEvaluación del Modelo Lineal:")
print(f"Error Absoluto Medio (MAE): {mae_lineal:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse_lineal:.2f}")
print(f"R^2 Score: {r2_lineal:.4f}")

# Visualización de resultados del modelo lineal
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_prueba['Movilidad'], y=y_prueba, color='blue', label='Datos Reales')
sns.lineplot(x=X_prueba['Movilidad'], y=y_pred_lineal, color='red', label='Predicción Lineal')
plt.title('Regresión Lineal - Predicciones vs Datos Reales')
plt.xlabel('Movilidad')
plt.ylabel('Casos')
plt.legend()
plt.show()

# Regresión Polinómica
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Definir el grado del polinomio
grado = 5

# Crear un pipeline que incluya la transformación polinómica y la regresión lineal
modelo_polinomico = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree=grado, include_bias=False)),
    ('linear_regression', LinearRegression())
])

# Entrenar el modelo polinómico
modelo_polinomico.fit(X_entrenamiento_scaled, y_entrenamiento)

# Predicciones del modelo polinómico
y_pred_polinomico = modelo_polinomico.predict(X_prueba_scaled)

# Evaluación del modelo polinómico
mae_polinomico = mean_absolute_error(y_prueba, y_pred_polinomico)
mse_polinomico = mean_squared_error(y_prueba, y_pred_polinomico)
r2_polinomico = r2_score(y_prueba, y_pred_polinomico)

print("\nEvaluación del Modelo Polinómico:")
print(f"Error Absoluto Medio (MAE): {mae_polinomico:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse_polinomico:.2f}")
print(f"R^2 Score: {r2_polinomico:.4f}")

# Visualización de resultados del modelo polinómico
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_prueba['Movilidad'], y=y_prueba, color='blue', label='Datos Reales')
sns.lineplot(x=X_prueba['Movilidad'], y=y_pred_polinomico, color='green', label='Predicción Polinómica')
plt.title('Regresión Polinómica - Predicciones vs Datos Reales')
plt.xlabel('Movilidad')
plt.ylabel('Casos')
plt.legend()
plt.show()

# Comparación de los modelos
# Crear un DataFrame para comparar las métricas
comparativa = pd.DataFrame({
    'Modelo': ['Lineal', 'Polinómico'],
    'MAE': [mae_lineal, mae_polinomico],
    'MSE': [mse_lineal, mse_polinomico],
    'R^2 Score': [r2_lineal, r2_polinomico]
})

print("\nComparativa de Métricas entre Modelo Lineal y Modelo Polinómico:")
print(comparativa)


# Gráfico comparativo de las predicciones en todo el rango de Movilidad
X_full = np.linspace(X['Movilidad'].min(), X['Movilidad'].max(), 100).reshape(-1, 1)
X_full_scaled = scaler.transform(X_full)

# Predicciones con ambos modelos
y_pred_lineal_full = modelo_lineal.predict(X_full_scaled)
y_pred_polinomico_full = modelo_polinomico.predict(X_full_scaled)

plt.figure(figsize=(12,6))
sns.scatterplot(x='Movilidad', y='Casos', data=datos, color='gray', label='Datos Reales')
sns.lineplot(x=X_full.flatten(), y=y_pred_lineal_full, color='red', label='Regresión Lineal')
sns.lineplot(x=X_full.flatten(), y=y_pred_polinomico_full, color='green', label='Regresión Polinómica')
plt.title('Comparación de Predicciones de Modelos en el Rango Completo')
plt.xlabel('Movilidad')
plt.ylabel('Casos')
plt.legend()
plt.show()
