# Proyecto alternativo: regresion con indices compuestos

## Justificacion
- Se creo un proyecto nuevo, separado del original, para reducir la cantidad de variables explicativas.
- En lugar de modelar muchas variables individuales, se agruparon por categoria conceptual.
- Se mantuvo un quinto indice de carga clinica para no perder la señal fuerte de diabetes e hipertension.
- En esta version, los indices no usan pesos heredados del puntaje de salud: se forman con variables estandarizadas dentro de cada categoria.

## Variables compuestas
- indice_sueno_salud_mental: horas de sueno, calidad de sueno, estres y depresion estandarizados.
- indice_actividad_corporal: actividad efectiva, minutos de actividad, horas de pantalla y exceso de IMC estandarizados.
- indice_habitos_saludables: azucar, alcohol y tabaquismo estandarizados.
- indice_contexto_socioambiental: nivel socioeconomico y condiciones ambientales estandarizados.
- indice_carga_clinica: diabetes e hipertension estandarizadas.

## Metodo de construccion
- Cada variable se transformo a escala z dentro de su categoria.
- Luego se promediaron con signo positivo para factores favorables y signo negativo para factores desfavorables.
- Esto reduce la circularidad respecto al proyecto compuesto anterior.

## Entrenamiento y prueba
- Entrenamiento: 60000
- Prueba: 15000

## Metricas en prueba
- R2: 0.9587
- MAE: 4.2456
- MSE: 28.1821
- RMSE: 5.3087

## Coeficientes del modelo
- indice_carga_clinica: 11.8805
- indice_actividad_corporal: 11.2447
- indice_sueno_salud_mental: 10.7139
- indice_habitos_saludables: 7.6934
- indice_contexto_socioambiental: 6.9510

## Supuestos del modelo
- Linealidad: False (p=0.0)
- Independencia: True (DW=2.0075)
- Homocedasticidad: False (p=0.0)
- Normalidad: False (p=0.0)
- Cumplen los 4 supuestos: False

## ANOVA
- indice_carga_clinica: F=247272.5042, p=0.0
- indice_sueno_salud_mental: F=118759.9444, p=0.0
- indice_actividad_corporal: F=114799.9455, p=0.0
- indice_habitos_saludables: F=50667.5251, p=0.0
- indice_contexto_socioambiental: F=25378.2864, p=0.0

## Archivos generados
- 01_pairplot_indices_compuestos.png
- 02_heatmap_indices_compuestos.png
- 03_coeficientes_indices_compuestos.png
- 04_predicciones_indices_compuestos.png
- 05_diagnostico_regresion_indices_compuestos.png
- predicciones_indices_compuestos.csv
- tabla_anova_indices_compuestos.csv
- dataset_indices_compuestos.csv
- metricas_indices_compuestos.json