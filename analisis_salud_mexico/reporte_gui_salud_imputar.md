# Modelo lineal multiple - Interfaz grafica

- Dataset: salud_mexico_sintetica_75000.csv
- Estrategia de nulos: imputar
- Registros antes: 75000
- Registros despues: 75000
- R2: 0.9870
- MAE: 2.3694
- MSE: 8.8397
- RMSE: 2.9732

## Variables del modelo
- Dependiente: salud_general_latente
- Independientes: calidad_sueno, estres, actividad_efectiva, exceso_imc, porciones_azucar_dia, unidades_alcohol_semana, fuma, horas_pantalla, depresion, hipertension, diabetes, contaminacion_aire, contaminacion_agua, contaminacion_suelo, contaminacion_auditiva, trafico, inseguridad, nivel_socioeconomico

## Ecuacion
`101.3261 + 3.0138*num__calidad_sueno - 2.1204*num__estres + 0.0354*num__actividad_efectiva - 1.4973*num__exceso_imc - 2.2148*num__porciones_azucar_dia - 0.3530*num__unidades_alcohol_semana - 8.0316*num__fuma - 1.6989*num__horas_pantalla - 10.8804*num__depresion - 11.9249*num__hipertension - 15.0701*num__diabetes - 0.8220*num__contaminacion_aire - 0.5328*num__contaminacion_agua - 0.3639*num__contaminacion_suelo - 0.5707*num__contaminacion_auditiva - 0.3418*num__trafico - 0.7334*num__inseguridad - 3.0397*cat__nivel_socioeconomico_bajo - 1.4096*cat__nivel_socioeconomico_medio`

## Limpieza
- Se imputaron nulos con la mediana en numericas y la moda en categoricas.