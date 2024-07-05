import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Cargamos los datos del archivo csv
datos = pd.read_csv("titanik.csv")

# Observamos la presencia de valores vacíos en "age"
print(datos["age"].isna().sum())

# Imputamos los valores faltantes en "age" usando la media por género
media_hombres = datos[datos["gender"] == "male"]["age"].mean()
media_mujeres = datos[datos["gender"] == "female"]["age"].mean()

datos.loc[datos["age"].isna() & (datos["gender"] == "male"), "age"] = media_hombres
datos.loc[datos["age"].isna() & (datos["gender"] == "female"), "age"] = media_mujeres

# Calculamos las medidas de tendencia central y dispersión para la edad
print("Media:", datos["age"].mean())
print("Mediana:", datos["age"].median())
print("Moda:", datos["age"].mode()[0])
print("Rango:", datos["age"].max() - datos["age"].min())
print("Varianza:", datos["age"].var())
print("Desviación estándar:", datos["age"].std())

# Calculamos la tasa de supervivencia general
supervivientes = datos["survived"] == 1
tasa_supervivencia_general = supervivientes.mean()
print("Tasa de supervivencia general:", tasa_supervivencia_general)

# Calculamos la tasa de supervivencia por género
tasa_supervivencia_hombres = datos[datos["gender"] == "male"]["survived"].mean()
tasa_supervivencia_mujeres = datos[datos["gender"] == "female"]["survived"].mean()

print("Tasa de supervivencia hombres:", tasa_supervivencia_hombres)
print("Tasa de supervivencia mujeres:", tasa_supervivencia_mujeres)

# Creamos un histograma de las edades por clase
plt.figure(figsize=(12, 6))

clases = datos["p_class"].unique()
for clase in clases:
    datos_clase = datos[datos["p_class"] == clase]
    plt.hist(datos_clase["age"], bins=20, alpha=0.5, label=f'Clase {clase}')

plt.legend()
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.title("Distribución de la edad por clase")
plt.grid(True)  # Agregamos una rejilla para mejor visualización
plt.show()

# Creamos diagramas de cajas para las edades de supervivientes  y no supervivientes
plt.figure(figsize=(8, 6))

plt.boxplot([datos[datos["survived"] == 1]["age"], datos[datos["survived"] == 0]["age"]],
           labels=["Supervivientes", "No supervivientes"], vert=False)

plt.xlabel("Edad")
plt.title("Distribución de la edad por supervivencia")
plt.grid(True)  # Agregamos una rejilla para mejor visualización
plt.show()

# Nivel de confianza
nivel_confianza = 0.95

# Obtenemos el alfa
alpha = 1 - nivel_confianza

# Calculamos la media, la desviación estándar y el tamaño de la muestra
media_muestra = datos["age"].mean()
desvio_estandar_muestra = datos["age"].std()
n = datos["age"].size

# Calculamos el margen de error usando la distribución t de Student
margen_error = stats.t.ppf(1 - alpha / 2, df=n - 1) * desvio_estandar_muestra / np.sqrt(n)

# Calculamos los límites del intervalo de confianza
limite_inferior = media_muestra - margen_error
limite_superior = media_muestra + margen_error

# Mostramos el intervalo de confianza
print(f"Intervalo de confianza ({nivel_confianza * 100} %): [{limite_inferior}, {limite_superior}]")

# Parte 2: Inferencia Estadística

# Promedio de edad de las mujeres interesadas en abordar el Titanic
media_edad_mujeres = datos[datos["gender"] == "female"]["age"].mean()
t_stat_mujeres, p_valor_mujeres = stats.ttest_1samp(datos[datos["gender"] == "female"]["age"], 56)
print(f"T-statistic para mujeres: {t_stat_mujeres}, P-valor: {p_valor_mujeres}")
print("Promedio de edad de las mujeres es mayor a 56 años (95%):", p_valor_mujeres / 2 < alpha and media_edad_mujeres > 56)

# Promedio de edad de los hombres interesados en abordar el Titanic
media_edad_hombres = datos[datos["gender"] == "male"]["age"].mean()
t_stat_hombres, p_valor_hombres = stats.ttest_1samp(datos[datos["gender"] == "male"]["age"], 56)
print(f"T-statistic para hombres: {t_stat_hombres}, P-valor: {p_valor_hombres}")
print("Promedio de edad de los hombres es mayor a 56 años (95%):", p_valor_hombres / 2 < alpha and media_edad_hombres > 56)

# Diferencia significativa en la tasa de supervivencia entre hombres y mujeres
t_stat_supervivencia, p_valor_supervivencia = stats.ttest_ind(datos[datos["gender"] == "female"]["survived"],
                                                              datos[datos["gender"] == "male"]["survived"])
print(f"T-statistic para supervivencia por género: {t_stat_supervivencia}, P-valor: {p_valor_supervivencia}")
print("Diferencia significativa en la tasa de supervivencia entre hombres y mujeres (99%):", p_valor_supervivencia < 0.01)

# Diferencia significativa en la tasa de supervivencia por clase
t_stat_clase, p_valor_clase = stats.f_oneway(datos[datos["p_class"] == 1]["survived"],
                                             datos[datos["p_class"] == 2]["survived"],
                                             datos[datos["p_class"] == 3]["survived"])
print(f"F-statistic para supervivencia por clase: {t_stat_clase}, P-valor: {p_valor_clase}")
print("Diferencia significativa en la tasa de supervivencia por clase (99%):", p_valor_clase < 0.01)

# Promedio de edad de mujeres y hombres en el barco
t_stat_edad, p_valor_edad = stats.ttest_ind(datos[datos["gender"] == "female"]["age"],
                                            datos[datos["gender"] == "male"]["age"])
print(f"T-statistic para edad por género: {t_stat_edad}, P-valor: {p_valor_edad}")
print("Promedio de edad de las mujeres es menor que el de los hombres (95%):", p_valor_edad / 2 < alpha and media_edad_mujeres < media_edad_hombres)