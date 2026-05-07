# Modelo compuesto - Interfaz grafica

- Dataset: dataset_indices_compuestos.csv
- Estrategia de nulos: imputar
- Registros antes: 75000
- Registros despues: 75000
- R2: 0.9871
- MAE: 2.3686
- MSE: 8.8346
- RMSE: 2.9723

## Variables del modelo
- Dependiente: salud_general_latente
- Independientes: indice_sueno_salud_mental, indice_actividad_corporal, indice_habitos_alimentacion, indice_contexto_socioambiental, indice_carga_clinica

## Ecuacion
`99.9393 + 0.9989*indice_sueno_salud_mental + 1.0013*indice_actividad_corporal + 1.0068*indice_habitos_alimentacion + 0.9937*indice_contexto_socioambiental + 0.9975*indice_carga_clinica`

## Limpieza
- Se imputaron nulos con la mediana de cada indice compuesto.