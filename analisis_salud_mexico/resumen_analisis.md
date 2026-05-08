# Analisis de salud en Mexico

## Alcance

- Se usaron tablas agregadas de ENSANUT ubicadas en `C:\Users\edluj\OneDrive\Documentos\actividades datos\avance de proyecto\data`.
- La carpeta no contiene respuestas individuales para sueno, actividad fisica, azucar, alcohol, tabaco o tiempo en pantalla.
- Para responder esas preguntas se genero una poblacion sintetica de 75,000 adultos, calibrada con prevalencias por edad y sexo de obesidad, sobrepeso, depresion, diabetes e hipertension observadas en los CSV.

## Hallazgos principales

### 1. Como afecta la cantidad y calidad de sueno al estres

- Correlacion horas de sueno vs estres: -0.6766.
- Correlacion calidad de sueno vs estres: -0.8377.
- Menos de 6 horas de sueno: estres promedio 9.95.
- Entre 7 y 8 horas de sueno: estres promedio 7.43.
- La calidad del sueno muestra una asociacion negativa mas fuerte que la cantidad de horas.

### 2. Relacion entre actividad fisica e IMC

- Correlacion actividad fisica vs IMC: -0.5659.
- IMC promedio por nivel de actividad:
  - Sedentaria: 33.51
  - Baja: 28.38
  - Media: 24.57
  - Alta: 21.71
- La relacion general es inversa: a mayor actividad, menor IMC promedio.

### 3. Influencia de azucar, alcohol, tabaco y tiempo en pantalla en la salud general

- Tabaco:
  - No fuma: 48.01
  - Si fuma: 29.04
- Alcohol:
  - 0 unidades por semana: 46.56
  - 10+ unidades por semana: 35.28
- Azucar:
  - 0-1 porciones: 55.59
  - 3+ porciones: 33.17
- Pantalla:
  - 0-3 horas: 65.4
  - 7+ horas: 30.76
- En el modelo sintetico, el tabaco y el exceso de tiempo en pantalla son los habitos que mas reducen la salud general, seguidos por el consumo elevado de azucar y alcohol.

### 4. Influencia directa de la alimentacion en la salud

- Alimentacion saludable: salud promedio 55.59.
- Alimentacion muy alta en azucar: salud promedio 33.17.
- Obesidad:
  - Alimentacion saludable: 22.3%
  - Alimentacion muy alta en azucar: 61.05%
- Diabetes:
  - Alimentacion saludable: 8.3%
  - Alimentacion muy alta en azucar: 15.23%
- En esta simulacion, una peor alimentacion se asocia de forma directa con menor salud general y mayor riesgo metabolico.

### 5. Contexto social y lugar de residencia

- Salud promedio por nivel socioeconomico:
  - Bajo: 38.35
  - Medio: 44.81
  - Alto: 52.35
- Salud promedio por localidad:
  - Rural: 60.72
  - Urbano: 40.43
- Fuente directa 2023:
  - Obesidad por nivel socioeconomico: bajo 37.6%, medio 47.4%, alto 38.2%.
  - Depresion por nivel socioeconomico: bajo 18.4%, medio 15.5%, alto 11.5%.
  - Obesidad por localidad: rural 40.9%, urbano 40.6%.
  - Depresion por localidad: rural 18.1%, urbano 14.1%.

### 6. Entorno y salud

- Correlacion indice de riesgo ambiental/social vs salud general: -0.4714.
- Salud promedio con riesgo bajo de entorno: 59.62.
- Salud promedio con riesgo muy alto de entorno: 31.76.
- En el modelo sintetico, mas contaminacion, ruido, trafico e inseguridad se asocian con peor salud y mayor estres.

### 7. Edades significativas de cierta condicion

- Depresion: el mayor porcentaje aparece en 60+ con 34.0% (fuente 2023).
- Hipertension: el mayor porcentaje aparece en 60+ con 41.8% (fuente 2023).
- Obesidad: el mayor porcentaje aparece en 40-59 con 49.4% (fuente 2023).


### 8. Regresion lineal multiple para salud general

- Variable dependiente: salud_general_latente.
- Justificacion: Se eligio salud_general_latente porque es la version no censurada del puntaje de salud y permite cumplir mejor los supuestos del modelo lineal que la version truncada entre 5 y 100.
- Analisis de correlacion:
  - Se genero `23_pairplot_variables_modelo.png` para visualizar las relaciones bivariadas entre las variables clave del modelo.
- Division entrenamiento/prueba:
  - Observaciones totales: 75000
  - Entrenamiento: 60000
  - Prueba: 15000
  - Proporcion de prueba: 0.2
- Se evaluaron variables demograficas, de habitos, condiciones clinicas y entorno, y el modelo final incluyo:
  calidad_sueno, estres, actividad_efectiva, exceso_imc, porciones_azucar_dia, unidades_alcohol_semana, fuma, horas_pantalla, depresion, hipertension, diabetes, contaminacion_aire, contaminacion_agua, contaminacion_suelo, contaminacion_auditiva, trafico, inseguridad, nivel_socioeconomico.
- Criterio de seleccion: LassoCV se uso para retener variables con coeficiente distinto de cero y RidgeCV para priorizar variables estables cuando habia colinealidad; despues se consolidaron variables redundantes en una especificacion final interpretable.
- LassoCV:
  - Alpha optimo: 0.023015
  - Variables retenidas: actividad_efectiva, calidad_sueno, contaminacion_agua, contaminacion_aire, contaminacion_suelo, depresion, diabetes, estres, exceso_imc, fuma, hipertension, horas_pantalla, imc, indice_entorno_riesgo, inseguridad, min_actividad_semana, nivel_socioeconomico, porciones_azucar_dia, trafico, unidades_alcohol_semana
- RidgeCV:
  - Alpha optimo: 3.162278
  - Variables priorizadas por peso: exceso_imc, diabetes, hipertension, depresion, fuma, nivel_socioeconomico, estres, horas_pantalla, calidad_sueno, porciones_azucar_dia, actividad_efectiva, unidades_alcohol_semana
- Desempeno de modelos regularizados:
  - Lasso R2: 0.987
  - Ridge R2: 0.987
- Graficas de regularizacion:
  - `19_lasso_importancia.png` muestra las variables retenidas por Lasso segun la magnitud de sus coeficientes.
  - `20_ridge_importancia.png` muestra las variables priorizadas por Ridge segun la magnitud de sus coeficientes.
- Desempeno en prueba:
  - R2: 0.987
  - MAE: 2.3694
  - MSE: 8.8397
  - RMSE: 2.9732
- Predicciones sobre prueba:
  - Se exportaron en `predicciones_prueba_regresion.csv`.
  - La comparacion visual entre valores reales y predichos se muestra en `24_predicciones_vs_reales.png`.
- Variables finales y sustento:
  - calidad_sueno: lasso+ridge.
  - estres: lasso+ridge.
  - actividad_efectiva: lasso+ridge.
  - exceso_imc: lasso+ridge.
  - porciones_azucar_dia: lasso+ridge.
- Variables con mayor peso estandarizado:
  - exceso_imc: beta -0.3307
  - diabetes: beta -0.1868
  - hipertension: beta -0.1783
  - depresion: beta -0.1487
  - fuma: beta -0.1163
- En terminos sustantivos, el IMC, el estres, la calidad del sueno, la hipertension y el entorno de riesgo son los determinantes mas importantes dentro del modelo final.

### 9. Verificacion de supuestos de la regresion lineal

- Linealidad (Ramsey RESET):
  - p-valor: 0.452732
  - Cumple: True
- Independencia de residuos (Durbin-Watson):
  - Estadistico: 1.9964
  - Cumple: True
- Homocedasticidad (Breusch-Pagan):
  - p-valor F: 0.790121
  - Cumple: True
- Normalidad de residuos (Jarque-Bera):
  - p-valor: 0.343433
  - Asimetria: 0.0047
  - Curtosis: 3.0244
  - Cumple: True
- Resultado global: True.

### 10. Analisis de tabla ANOVA

- La tabla ANOVA tipo II confirma que las variables con mayor aporte al modelo son:
  - exceso_imc: F = 218619.5837, p = 0.0
  - diabetes: F = 94579.0269, p = 0.0
  - depresion: F = 86267.4418, p = 0.0
  - hipertension: F = 72449.6852, p = 0.0
  - fuma: F = 68213.523, p = 0.0
- Todas las variables del modelo final presentan evidencia estadistica de aporte al explicar la variacion del puntaje latente de salud.

## Prevalencias nacionales recientes observadas en fuente directa

- Obesidad: 40.7% (2023)
- Sobrepeso: 36.4% (2023)
- Depresion moderada/severa: 14.9% (2023)
- Diabetes diagnosticada: 12.2% (2023)
- Hipertension diagnosticada: 19.6% (2023)

## Archivos generados

- `01_tendencias_nacionales.png`
- `02_sueno_estres.png`
- `03_actividad_imc.png`
- `04_habitos_salud_general.png`
- `05_edades_significativas.png`
- `06_tabaco_alcohol_salud.png`
- `07_alimentacion_salud.png`
- `08_contexto_social_salud.png`
- `09_entorno_salud.png`
- `10_heatmap_sueno_estres.png`
- `11_heatmap_actividad_imc.png`
- `12_heatmap_habitos_salud.png`
- `13_heatmap_alimentacion_salud.png`
- `14_heatmap_contexto_social.png`
- `15_heatmap_edad_condiciones.png`
- `16_heatmap_entorno_salud.png`
- `17_regresion_salud_general.png`
- `18_diagnostico_regresion.png`
- `19_lasso_importancia.png`
- `20_ridge_importancia.png`
- `21_heatmap_correlacion_variables.png`
- `22_heatmap_correlacion_modelo.png`
- `23_pairplot_variables_modelo.png`
- `24_predicciones_vs_reales.png`
- `25_trayectoria_coeficientes_ridge.png`
- `26_trayectoria_coeficientes_lasso.png`
- `predicciones_prueba_regresion.csv`
- `salud_mexico_sintetica_75000.csv`
- `metricas_resumen.json`
