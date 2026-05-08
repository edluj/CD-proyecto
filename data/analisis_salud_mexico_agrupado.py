import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. MANTENEMOS LAS MISMAS RUTAS DEL PROYECTO ORIGINAL
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "analisis_salud_mexico"

# Archivo de entrada (generado por tu código original)
ARCHIVO_ENTRADA = OUTPUT_DIR / "salud_mexico_sintetica_75000.csv"
# Archivo de salida (con las 5 nuevas variables)
ARCHIVO_SALIDA = OUTPUT_DIR / "salud_mexico_sintetica_agrupada.csv"

def agrupar_dataset():
    print(f"Buscando dataset en: {ARCHIVO_ENTRADA}")
    
    if not ARCHIVO_ENTRADA.exists():
        print("Error: No se encontró el dataset sintético.")
        print("Asegúrate de haber corrido 'analisis_salud_mexico.py' primero para generarlo.")
        return

    # Leer los datos
    df = pd.read_csv(ARCHIVO_ENTRADA)
    print(f"✅ Dataset cargado correctamente. Filas: {len(df)}")
    
    # 2. ESCALADO DE DATOS (MinMaxScaler para estandarizar entre 0 y 1)
    scaler = MinMaxScaler()
    cols_a_escalar = [
        "min_actividad_semana", "actividad_efectiva", "horas_pantalla",
        "porciones_azucar_dia", "unidades_alcohol_semana", "fuma",
        "contaminacion_aire", "contaminacion_agua", "contaminacion_suelo", 
        "contaminacion_auditiva", "trafico", "inseguridad", "indice_entorno_riesgo",
        "horas_sueno", "calidad_sueno", "estres",
        "imc", "exceso_imc", "depresion", "hipertension", "diabetes"
    ]
    
    # Filtrar solo las columnas que realmente existen en el CSV
    existentes = [c for c in cols_a_escalar if c in df.columns]
    df_scaled = pd.DataFrame(scaler.fit_transform(df[existentes]), columns=existentes, index=df.index)

    print("⏳ Generando las 5 variables principales...")
    
    # --- 1. Actividad General ---
    act_vars = [df_scaled.get("min_actividad_semana", 0), df_scaled.get("actividad_efectiva", 0)]
    # Restamos las horas pantalla (invertimos el valor)
    if "horas_pantalla" in df_scaled.columns:
        act_vars.append(1 - df_scaled["horas_pantalla"])
    df["actividad_general"] = sum(act_vars) / len(act_vars)

    # --- 2. Hábitos de Nutrición y Consumo ---
    nutricion_vars = ["porciones_azucar_dia", "unidades_alcohol_semana", "fuma"]
    cols_nutricion = [c for c in nutricion_vars if c in df_scaled.columns]
    df["habitos_de_nutricion"] = df_scaled[cols_nutricion].mean(axis=1) if cols_nutricion else 0

    # --- 3. Contexto Social, Económico y Ambiental ---
    ambientales_vars = ["contaminacion_aire", "contaminacion_agua", "contaminacion_suelo", 
                        "contaminacion_auditiva", "trafico", "inseguridad", "indice_entorno_riesgo"]
    cols_amb = [c for c in ambientales_vars if c in df_scaled.columns]
    df["contexto_social_economico_y_ambiental"] = df_scaled[cols_amb].mean(axis=1) if cols_amb else 0

    # --- 4. Salud Mental y Descanso ---
    descanso_vars = [df_scaled.get("horas_sueno", 0), df_scaled.get("calidad_sueno", 0)]
    # Invertimos el estrés para que sume como bienestar
    if "estres" in df_scaled.columns:
        descanso_vars.append(1 - df_scaled["estres"])
    df["salud_mental_y_descanso"] = sum(descanso_vars) / len(descanso_vars)

    # --- 5. Clínica y Condiciones ---
    clinicas_vars = ["imc", "exceso_imc", "depresion", "hipertension", "diabetes"]
    cols_clin = [c for c in clinicas_vars if c in df_scaled.columns]
    df["clinica"] = df_scaled[cols_clin].mean(axis=1) if cols_clin else 0

    # 3. GUARDAR EL NUEVO DATASET
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"¡Proceso finalizado! Nuevo dataset exportado en:")
    print(f"   {ARCHIVO_SALIDA}")
    
    # Muestra un pequeño resumen de las nuevas variables
    print("\nResumen de las nuevas variables (primeras 5 filas):")
    print(df[["actividad_general", "habitos_de_nutricion", 
              "contexto_social_economico_y_ambiental", 
              "salud_mental_y_descanso", "clinica"]].head())

if __name__ == "__main__":
    agrupar_dataset()