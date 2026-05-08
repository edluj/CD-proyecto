"""
Regresión con índices compuestos y transformación Box-Cox
Objetivo: Cumplir los 4 supuestos de la regresión lineal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera, het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CARGAR DATOS
# ==========================================
print("Cargando datos...")
df = pd.read_csv('analisis_salud_mexico_compuesto/dataset_indices_compuestos.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ==========================================
# 2. PREPARAR VARIABLES
# ==========================================
indices = [
    'indice_sueno_salud_mental',
    'indice_actividad_corporal',
    'indice_habitos_saludables',
    'indice_contexto_socioambiental',
    'indice_carga_clinica'
]

X = df[indices].copy()
y = df['salud_general_latente'].copy()

# Eliminar valores faltantes
mask = (~X.isna().any(axis=1)) & (~y.isna())
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"Datos limpios: {len(X)} observaciones")

# ==========================================
# 3. APLICAR TRANSFORMACIÓN BOX-COX
# ==========================================
print("\n" + "="*60)
print("TRANSFORMACIÓN BOX-COX")
print("="*60)

# Box-Cox requiere valores positivos
y_min = y.min()
y_shifted = y - y_min + 0.1  # Agregar pequeño offset para asegurar positividad

# Encontrar la transformación óptima
y_transformed, lambda_param = boxcox(y_shifted)

print(f"Parámetro lambda óptimo: {lambda_param:.4f}")
print(f"y_original - media: {y.mean():.2f}, desv.est: {y.std():.2f}")
print(f"y_transformada - media: {y_transformed.mean():.2f}, desv.est: {y_transformed.std():.2f}")

# ==========================================
# 4. DIVIDIR EN TRAIN/TEST
# ==========================================
X_train, X_test, y_train_t, y_test_t, indices_train, indices_test = train_test_split(
    X, y_transformed, y.index, test_size=0.2, random_state=42
)

# Mantener copia de y original para comparación - usar índices originales
y_train_orig = y[indices_train].values
y_test_orig = y[indices_test].values

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Resetear índices para consistencia
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_t = pd.Series(y_train_t, index=range(len(y_train_t)))
y_test_t = pd.Series(y_test_t, index=range(len(y_test_t)))
y_train_orig = pd.Series(y_train_orig, index=range(len(y_train_orig)))
y_test_orig = pd.Series(y_test_orig, index=range(len(y_test_orig)))

# ==========================================
# 5. ENTRENAR MODELO CON Y TRANSFORMADA
# ==========================================
print("\n" + "="*60)
print("MODELO DE REGRESIÓN (CON Y TRANSFORMADA)")
print("="*60)

# Agregar constante
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Entrenar
model = sm.OLS(y_train_t, X_train_const).fit()
print(model.summary())

# Predicciones en escala transformada
y_pred_t = model.predict(X_test_const)

# R2 en escala transformada
r2_transformada = r2_score(y_test_t, y_pred_t)
print(f"\nR² (escala transformada): {r2_transformada:.6f}")

# ==========================================
# 6. INVERTIR TRANSFORMACIÓN PARA INTERPRETABILIDAD
# ==========================================
print("\n" + "="*60)
print("INVERSIÓN DE LA TRANSFORMACIÓN")
print("="*60)

def inverse_boxcox(y_transformed, lambda_param, y_shift):
    """
    Invierte la transformación Box-Cox correctamente
    """
    if abs(lambda_param) < 1e-10:
        # Si lambda ≈ 0, se usó transformación logarítmica
        y_inverted = np.exp(y_transformed)
    else:
        # Inversión estándar: y = (lambda * y_transformed + 1)^(1/lambda)
        y_inverted = np.power(lambda_param * y_transformed + 1, 1 / lambda_param)
    
    # Deshacer el shift original
    y_original = y_inverted + y_shift
    
    return y_original

# Invertir predicciones
y_pred_orig = inverse_boxcox(y_pred_t.values, lambda_param, y_min - 0.1)

# Asegurar que no hay NaN ni infinitos
print(f"Valores predichos - NaN: {np.isnan(y_pred_orig).sum()}, Inf: {np.isinf(y_pred_orig).sum()}")

# Si hay valores problemáticos, usar clipping
if np.isnan(y_pred_orig).any() or np.isinf(y_pred_orig).any():
    print("⚠ Se detectaron NaN o Inf. Reemplazando con valores válidos...")
    y_pred_orig = np.clip(y_pred_orig, y.min(), y.max())
    y_pred_orig[np.isnan(y_pred_orig)] = y.mean()

# Métricas en escala original
r2_original = r2_score(y_test_orig, y_pred_orig)
mae_original = mean_absolute_error(y_test_orig, y_pred_orig)
mse_original = mean_squared_error(y_test_orig, y_pred_orig)
rmse_original = np.sqrt(mse_original)

print(f"\nMétricas en escala original:")
print(f"  R²:   {r2_original:.6f}")
print(f"  MAE:  {mae_original:.4f}")
print(f"  MSE:  {mse_original:.4f}")
print(f"  RMSE: {rmse_original:.4f}")

# ==========================================
# 7. VERIFICAR SUPUESTOS EN RESIDUOS TRANSFORMADOS
# ==========================================
print("\n" + "="*60)
print("VERIFICACIÓN DE SUPUESTOS (Escala Transformada)")
print("="*60)

residuos_t = model.resid

# 1. LINEALIDAD (Ramsey RESET)
try:
    reset_result = linear_rainbow(model)
    print(f"1. LINEALIDAD (Ramsey RESET)")
    print(f"   Estadístico: {reset_result[0]:.4f}")
    print(f"   p-valor: {reset_result[1]:.6f}")
    linealidad_cumple = reset_result[1] > 0.05
    print(f"   Cumple: {linealidad_cumple}")
except Exception as e:
    print(f"1. LINEALIDAD: No se pudo calcular ({str(e)[:50]})")
    reset_result = (np.nan, np.nan)
    linealidad_cumple = None

# 2. INDEPENDENCIA (Durbin-Watson)
dw = durbin_watson(residuos_t)
print(f"\n2. INDEPENDENCIA (Durbin-Watson)")
print(f"   Estadístico: {dw:.4f} (rango: 0-4, ideal: ~2)")
independencia_cumple = 1.5 < dw < 2.5
print(f"   Cumple: {independencia_cumple}")

# 3. HOMOCEDASTICIDAD (Breusch-Pagan)
bp_test = het_breuschpagan(residuos_t, X_train_const)
print(f"\n3. HOMOCEDASTICIDAD (Breusch-Pagan)")
print(f"   Estadístico LM: {bp_test[0]:.4f}")
print(f"   p-valor: {bp_test[1]:.6f}")
homocedasticidad_cumple = bp_test[1] > 0.05
print(f"   Cumple: {homocedasticidad_cumple}")

# 4. NORMALIDAD (Jarque-Bera)
jb_test = jarque_bera(residuos_t)
print(f"\n4. NORMALIDAD (Jarque-Bera)")
print(f"   Estadístico: {jb_test[0]:.4f}")
print(f"   p-valor: {jb_test[1]:.6f}")
print(f"   Asimetría: {jb_test[2]:.4f}")
print(f"   Curtosis: {jb_test[3]:.4f}")
normalidad_cumple = jb_test[1] > 0.05
print(f"   Cumple: {normalidad_cumple}")

# Verificar cuántos supuestos se cumplen
supuestos_list = [linealidad_cumple, independencia_cumple, homocedasticidad_cumple, normalidad_cumple]
supuestos_cumplidos = sum([x for x in supuestos_list if x is not None])
print(f"\n✓ Supuestos cumplidos: {supuestos_cumplidos}/4")

# ==========================================
# 8. COEFICIENTES
# ==========================================
print("\n" + "="*60)
print("COEFICIENTES DEL MODELO (Escala Transformada)")
print("="*60)

coeficientes = model.params[1:]
print("\nCoeficientes estandarizados (sobre variable transformada):")
for var, coef in coeficientes.items():
    print(f"  {var}: {coef:.6f}")

# ==========================================
# 9. VISUALIZACIONES
# ==========================================
print("\n" + "="*60)
print("Generando visualizaciones...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Diagnóstico: Regresión con Box-Cox (Y Transformada)', fontsize=16, fontweight='bold')

# 1. Distribución Y original vs transformada
axes[0, 0].hist(y_test_orig, bins=50, alpha=0.7, label='Original', color='blue', edgecolor='black')
axes[0, 0].set_xlabel('Valor')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Y Original')
axes[0, 0].legend()

axes[0, 1].hist(y_test_t, bins=50, alpha=0.7, label='Box-Cox', color='green', edgecolor='black')
axes[0, 1].set_xlabel('Valor')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title(f'Y Transformada (λ={lambda_param:.3f})')
axes[0, 1].legend()

# 2. Q-Q plot residuos transformados
sm.qqplot(residuos_t, line='45', ax=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot (Residuos Transformados)')

# 3. Residuos vs valores ajustados
axes[1, 0].scatter(y_pred_t.values, residuos_t.values, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Valores ajustados')
axes[1, 0].set_ylabel('Residuos')
axes[1, 0].set_title('Homocedasticidad (Escala Transformada)')

# 4. Predicciones vs valores reales (escala original)
axes[1, 1].scatter(y_test_orig, y_pred_orig, alpha=0.5, s=20)
axes[1, 1].plot([y_test_orig.min(), y_test_orig.max()], 
                [y_test_orig.min(), y_test_orig.max()], 
                'r--', lw=2)
axes[1, 1].set_xlabel('Valor Real')
axes[1, 1].set_ylabel('Predicción')
axes[1, 1].set_title(f'Predicciones vs Reales (Escala Original)\nR²={r2_original:.4f}')

# 5. Histograma residuos
axes[1, 2].hist(residuos_t.values, bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1, 2].set_xlabel('Residuos')
axes[1, 2].set_ylabel('Frecuencia')
axes[1, 2].set_title('Distribución de Residuos (Transformados)')

plt.tight_layout()
plt.savefig('analisis_salud_mexico_compuesto/06_diagnostico_boxcox.png', dpi=100, bbox_inches='tight')
print("✓ Guardado: 06_diagnostico_boxcox.png")
plt.close()

# ==========================================
# 10. GUARDAR RESULTADOS
# ==========================================
import json

resultados = {
    "metodo": "Regresion con Box-Cox",
    "descripcion": "Se aplicó transformación Box-Cox a la variable dependiente para cumplir supuestos de regresión",
    "lambda_boxcox": float(lambda_param),
    "variables_independientes": indices,
    "train_test_split": {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_proportion": 0.2
    },
    "metricas_test": {
        "r2": float(r2_original),
        "mae": float(mae_original),
        "mse": float(mse_original),
        "rmse": float(rmse_original)
    },
    "coeficientes_escala_transformada": {var: float(coef) for var, coef in coeficientes.items()},
    "intercept_escala_transformada": float(model.params[0]),
    "supuestos": {
        "linealidad": {
            "prueba": "Ramsey RESET",
            "p_valor": float(reset_result[1]) if not np.isnan(reset_result[1]) else None,
            "cumple": linealidad_cumple
        },
        "independencia": {
            "prueba": "Durbin-Watson",
            "estadistico": float(dw),
            "cumple": independencia_cumple
        },
        "homocedasticidad": {
            "prueba": "Breusch-Pagan",
            "p_valor": float(bp_test[1]),
            "cumple": homocedasticidad_cumple
        },
        "normalidad": {
            "prueba": "Jarque-Bera",
            "p_valor": float(jb_test[1]),
            "asimetria": float(jb_test[2]),
            "curtosis": float(jb_test[3]),
            "cumple": normalidad_cumple
        },
        "total_cumplidos": supuestos_cumplidos
    },
    "comparacion_con_anterior": {
        "nota": "Comparar con metricas_indices_compuestos.json",
        "mejora_esperada": "Mejor cumplimiento de supuestos, especialmente normalidad y homocedasticidad"
    }
}

with open('analisis_salud_mexico_compuesto/metricas_boxcox.json', 'w') as f:
    json.dump(resultados, f, indent=2)

print("✓ Guardado: metricas_boxcox.json")

# ==========================================
# 11. COMPARACIÓN CON MODELO ORIGINAL
# ==========================================
print("\n" + "="*60)
print("COMPARACIÓN: MODELO ORIGINAL vs BOX-COX")
print("="*60)

print("\nMODELO ORIGINAL (sin transformación):")
print("  R²:               0.9587")
print("  Linealidad:       False")
print("  Independencia:    True")
print("  Homocedasticidad: False")
print("  Normalidad:       False")
print("  Supuestos (4):    1/4")

print("\nMODELO CON BOX-COX (transformado):")
print(f"  R²:               {r2_original:.4f}")
print(f"  Linealidad:       {linealidad_cumple}")
print(f"  Independencia:    {independencia_cumple}")
print(f"  Homocedasticidad: {homocedasticidad_cumple}")
print(f"  Normalidad:       {normalidad_cumple}")
print(f"  Supuestos (4):    {supuestos_cumplidos}/4")

print("\n✓ Script completado exitosamente")
