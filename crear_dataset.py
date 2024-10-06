

import pandas as pd
import numpy as np

# Configuración de la semilla para reproducibilidad
np.random.seed(42)

# Generar datos
num_muestras = 100

# Variable independiente: Movilidad (rango entre 0.5 y 5.0)
movilidad = np.linspace(0.5, 5.0, num_muestras)

# Añadir algo de ruido a la movilidad para simular datos reales
movilidad += np.random.normal(0, 0.1, num_muestras)

# Variable dependiente: Casos
# Relación no lineal más pronunciada
a, b, c, d = 2, -3, 5, 10  
casos = a * movilidad**3 + b * movilidad**2 + c * movilidad + d

# Añadir menos ruido a los casos
casos += np.random.normal(0, 10, num_muestras)

# Asegurarse de que los casos sean números enteros positivos
casos = np.clip(casos.astype(int), 0, None)

# Crear DataFrame
datos = pd.DataFrame({
    'Movilidad': movilidad,
    'Casos': casos
})

# Guardar a CSV
datos.to_csv('datos_pandemia.csv', index=False)

print("Dataset 'datos_pandemia.csv' creado exitosamente.")
