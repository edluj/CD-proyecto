from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Directorios específicos para la versión simplificada
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analisis_simplificado" / "resultados"
MPL_DIR = OUTPUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

TARGET_N = 75_000
SEED = 42

FILES = {
    "obesidad": "Ideas_ENSANUT_poblacion_adulta_con_obesidad.csv",
    "sobrepeso": "Ideas_ENSANUT_poblacion_adulta_con_sobrepeso.csv",
    "depresion": "Ideas_ENSANUT_depresion_poblacion_adulta_con_sintomas_de_depresion.csv",
    "diabetes": "Ideas_ENSANUT_diabetes_poblacion_adulta_con_diabetes_diagnosticada.csv",
    "hipertension": "Ideas_ENSANUT_hipertension_poblacion_adulta_con_hipertension_diagnosticada.csv",
}

AGE_RANGES = {"20-39": (20, 39), "40-59": (40, 59), "60+": (60, 85)}
PREFERRED_YEARS = [2023, 2022, 2021, 2018, 2016, 2012, 2006, 2000]

REGRESSION_TARGET = "salud_general_latente"

# Nuevas variables fusionadas
REGRESSION_FINAL_NUMERIC = [
    "indice_sueno_salud_mental",
    "indice_actividad_fisica",
    "indice_alimentacion",
    "indice_socioeconomico_ambiental",
    "hipertension",
    "diabetes"
]
REGRESSION_FINAL_CATEGORICAL = [] # Simplificado a solo numéricas compuestas

REGRESSION_LABELS = {
    "indice_sueno_salud_mental": "Sueño y Salud Mental",
    "indice_actividad_fisica": "Actividad Física y Sedentarismo",
    "indice_alimentacion": "Hábitos Alimenticios",
    "indice_socioeconomico_ambiental": "Contexto Socio-Ambiental",
    "hipertension": "Hipertensión",
    "diabetes": "Diabetes"
}

GUI_ESTRATEGIA_ELIMINAR = "eliminar"
GUI_ESTRATEGIA_IMPUTAR = "imputar"
GUI_ESTRATEGIAS_VALIDAS = {GUI_ESTRATEGIA_ELIMINAR, GUI_ESTRATEGIA_IMPUTAR}

# Reutilizamos funciones de carga y procesamiento de ENSANUT del original
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].fillna("-").astype(str).str.strip().str.lower()
    return df

def percentage_column(df: pd.DataFrame) -> str:
    return next(col for col in df.columns if col.startswith("porcentaje_"))

def population_column(df: pd.DataFrame) -> str | None:
    cols = [col for col in df.columns if col.startswith("poblacion_representada")]
    return cols[0] if cols else None

def parse_population(value: object) -> float:
    if pd.isna(value): return np.nan
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    return float(digits) if digits else np.nan

def filter_exact(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    subset = df.copy()
    for col, value in filters.items():
        if value is None: continue
        if col in subset.columns:
            subset = subset[subset[col] == value]
    return subset

def latest_value(df: pd.DataFrame, filters: dict[str, object], preferred_years: list[int] | None = None) -> tuple[float, int | None]:
    pct_col = percentage_column(df)
    years = preferred_years or PREFERRED_YEARS
    for year in years:
        subset = filter_exact(df, {"año": year, **filters})
        subset = subset[subset[pct_col].notna()]
        if not subset.empty:
            return float(subset.iloc[0][pct_col]), year
    return np.nan, None

def latest_value_with_fallbacks(df: pd.DataFrame, sex: str, age_group: str) -> tuple[float, int | None]:
    base = {
        "sexo": sex,
        "grupo_de_edad": age_group,
        "tipo_de_localidad_rural_o_urbano": "-",
        "region_geografica_de_Mexico": "-",
        "nivel_educativo": "-",
        "nivel_socioeconomico": "-",
        "presencia_de_discapacidad": "-",
    }
    if "nivel_de_agregacion" in df.columns:
        base["nivel_de_agregacion"] = "estratificado"
    attempts = [base, {**base, "sexo": "-", "grupo_de_edad": age_group}, {**base, "sexo": sex, "grupo_de_edad": "-"}, {**base, "sexo": "-", "grupo_de_edad": "-"}]
    for filters in attempts:
        value, year = latest_value(df, filters)
        if not np.isnan(value): return value, year
    return np.nan, None

def build_national_trend(df: pd.DataFrame, label: str) -> pd.DataFrame:
    pct_col = percentage_column(df)
    rows = []
    years = sorted(int(year) for year in df["año"].dropna().unique())
    for year in years:
        filters = {"año": year, "sexo": "-", "grupo_de_edad": "-", "tipo_de_localidad_rural_o_urbano": "-", "region_geografica_de_Mexico": "-", "nivel_educativo": "-", "nivel_socioeconomico": "-", "presencia_de_discapacidad": "-"}
        if "nivel_de_agregacion" in df.columns: filters["nivel_de_agregacion"] = "nacional"
        subset = filter_exact(df, filters)
        subset = subset[subset[pct_col].notna()]
        if subset.empty and "nivel_de_agregacion" in df.columns:
            filters["nivel_de_agregacion"] = "nacional "
            subset = filter_exact(df, filters)
        if not subset.empty:
            rows.append({"anio": year, "indicador": label, "porcentaje": float(subset.iloc[0][pct_col])})
    return pd.DataFrame(rows)

def build_cell_population(obesity_df: pd.DataFrame) -> pd.DataFrame:
    pct_col = percentage_column(obesity_df)
    pop_col = population_column(obesity_df)
    filters = {"año": 2023, "tipo_de_localidad_rural_o_urbano": "-", "region_geografica_de_Mexico": "-", "nivel_educativo": "-", "nivel_socioeconomico": "-", "presencia_de_discapacidad": "-"}
    if "nivel_de_agregacion" in obesity_df.columns: filters["nivel_de_agregacion"] = "estratificado"
    subset = filter_exact(obesity_df, filters)
    subset = subset[subset["sexo"].isin(["hombre", "mujer"]) & subset["grupo_de_edad"].isin(["20-39", "40-59", "60+"]) & subset[pct_col].notna()].copy()
    subset["poblacion_celda"] = subset[pop_col].map(parse_population) / (subset[pct_col] / 100.0)
    subset["peso_muestral"] = subset["poblacion_celda"] / subset["poblacion_celda"].sum()
    return subset[["sexo", "grupo_de_edad", "poblacion_celda", "peso_muestral"]].reset_index(drop=True)

def build_prevalence_map(datasets: dict[str, pd.DataFrame], metric: str) -> dict[tuple[str, str], float]:
    df = datasets[metric]
    result: dict[tuple[str, str], float] = {}
    for sex in ["hombre", "mujer"]:
        for age_group in ["20-39", "40-59", "60+"]:
            value, _ = latest_value_with_fallbacks(df, sex, age_group)
            result[(sex, age_group)] = float(value)
    return result

# --- Generación de datos sintéticos (adaptado) ---

def sample_population(cell_pop: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    chosen = rng.choice(cell_pop.index.to_numpy(), size=TARGET_N, p=cell_pop["peso_muestral"].to_numpy())
    base = cell_pop.loc[chosen, ["sexo", "grupo_de_edad"]].reset_index(drop=True)
    base["id_persona"] = np.arange(1, TARGET_N + 1)
    ages = np.zeros(TARGET_N, dtype=int)
    for age_group, (low, high) in AGE_RANGES.items():
        mask = base["grupo_de_edad"] == age_group
        count = int(mask.sum())
        if not count: continue
        if age_group != "60+":
            ages[mask] = rng.integers(low, high + 1, size=count)
        else:
            ages[mask] = np.clip(np.round(60 + rng.gamma(shape=2.2, scale=6.0, size=count)), low, high).astype(int)
    base["edad"] = ages
    return base

def assign_social_context(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rural_p = df["grupo_de_edad"].map({"20-39": 0.18, "40-59": 0.21, "60+": 0.27}).to_numpy(dtype=float)
    df["tipo_localidad"] = np.where(rng.random(len(df)) < rural_p, "rural", "urbano")
    
    ses_choices = []
    for _, row in df.iterrows():
        probs = np.array([0.53, 0.32, 0.15]) if row["tipo_localidad"] == "rural" else np.array([0.29, 0.41, 0.30])
        ses_choices.append(rng.choice(["bajo", "medio", "alto"], p=probs))
    df["nivel_socioeconomico"] = ses_choices

    urban = df["tipo_localidad"].eq("urbano").astype(float).to_numpy()
    low_ses = df["nivel_socioeconomico"].eq("bajo").astype(float).to_numpy()
    df["contaminacion_aire"] = np.clip(2.8 + urban * 2.8 + low_ses * 1.0 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["contaminacion_agua"] = np.clip(2.6 + (1 - urban) * 1.2 + low_ses * 1.1 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["contaminacion_suelo"] = np.clip(2.4 + (1 - urban) * 1.0 + low_ses * 0.9 + rng.normal(0, 0.95, len(df)), 1, 10)
    df["contaminacion_auditiva"] = np.clip(1.8 + urban * 3.2 + low_ses * 0.3 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["trafico"] = np.clip(1.7 + urban * 3.5 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["inseguridad"] = np.clip(2.5 + urban * 1.8 + low_ses * 1.4 + rng.normal(0, 1.1, len(df)), 1, 10)
    df["indice_entorno_riesgo"] = (df["contaminacion_aire"] * 0.22 + df["contaminacion_agua"] * 0.14 + df["contaminacion_suelo"] * 0.10 + df["contaminacion_auditiva"] * 0.16 + df["trafico"] * 0.16 + df["inseguridad"] * 0.22)
    return df

def generate_habits(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Lógica de generación simplificada para mantener la base
    df["horas_pantalla"] = np.clip(rng.normal(5.0, 2.0, len(df)), 0, 12)
    df["porciones_azucar_dia"] = np.clip(rng.normal(2.0, 1.0, len(df)), 0, 7)
    df["min_actividad_semana"] = np.clip(rng.normal(150, 60, len(df)), 0, 420)
    df["horas_sueno"] = np.clip(rng.normal(7.0, 1.0, len(df)), 4, 10)
    df["calidad_sueno"] = np.clip(rng.normal(3.5, 0.8, len(df)), 1, 5)
    df["estres"] = np.clip(rng.normal(5.0, 2.0, len(df)), 1, 10)
    df["fuma"] = (rng.random(len(df)) < 0.15).astype(int)
    df["unidades_alcohol_semana"] = np.clip(rng.gamma(2, 2, len(df)), 0, 25)
    df["actividad_efectiva"] = np.minimum(df["min_actividad_semana"], 300)
    return df

def assign_bmi(df: pd.DataFrame, overweight_map: dict, obesity_map: dict, rng: np.random.Generator) -> pd.DataFrame:
    # Lógica de IMC base para calibrar prevalencias
    df["riesgo_bmi"] = 24.0 + rng.normal(0, 3, len(df))
    df["categoria_imc"] = "normal"
    df["imc"] = np.nan
    for (sex, age_group), idx in df.groupby(["sexo", "grupo_de_edad"]).groups.items():
        sub = df.loc[list(idx)].sort_values("riesgo_bmi")
        n = len(sub)
        p_over = overweight_map[(sex, age_group)] / 100.0
        p_ob = obesity_map[(sex, age_group)] / 100.0
        k_ob = int(round(n * p_ob))
        k_over = int(round(n * p_over))
        df.loc[sub.index[-k_ob:], "categoria_imc"] = "obesidad"
        df.loc[sub.index[-(k_ob+k_over):-k_ob], "categoria_imc"] = "sobrepeso"
        df.loc[df.index.isin(idx) & (df["categoria_imc"] == "obesidad"), "imc"] = rng.uniform(30, 40, k_ob)
        df.loc[df.index.isin(idx) & (df["categoria_imc"] == "sobrepeso"), "imc"] = rng.uniform(25, 29.9, k_over)
        df.loc[df.index.isin(idx) & (df["categoria_imc"] == "normal"), "imc"] = rng.uniform(18.5, 24.9, n - k_ob - k_over)
    return df

def assign_binary_condition(df: pd.DataFrame, score: np.ndarray, prevalence_map: dict, column_name: str) -> pd.DataFrame:
    df[column_name] = 0
    for (sex, age_group), idx in df.groupby(["sexo", "grupo_de_edad"]).groups.items():
        target = prevalence_map[(sex, age_group)] / 100.0
        k = int(round(len(idx) * target))
        if k > 0:
            chosen = df.loc[idx].sample(k).index
            df.loc[chosen, column_name] = 1
    return df

def add_conditions(df: pd.DataFrame, prevalence_maps: dict, rng: np.random.Generator) -> pd.DataFrame:
    df = assign_binary_condition(df, None, prevalence_maps["depresion"], "depresion")
    df = assign_binary_condition(df, None, prevalence_maps["hipertension"], "hipertension")
    df = assign_binary_condition(df, None, prevalence_maps["diabetes"], "diabetes")
    return df

def add_health_score(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Score de salud latente basado en los nuevos índices
    score = 100 - (10 - df["indice_sueno_salud_mental"]) * 2.0 \
                - (10 - df["indice_actividad_fisica"]) * 1.5 \
                - (10 - df["indice_alimentacion"]) * 1.8 \
                - (10 - df["indice_socioeconomico_ambiental"]) * 1.2 \
                - df["diabetes"] * 15 - df["hipertension"] * 12 + rng.normal(0, 2, len(df))
    df["salud_general_latente"] = score
    df["salud_general"] = np.clip(score, 5, 100)
    return df

# --- NUEVA LÓGICA DE FUSIÓN DE VARIABLES ---

def fuse_variables(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Sueño y Salud Mental (Calidad sueño, Estrés, Depresión)
    # Normalizamos a 0-10 (10 es mejor salud)
    s_calidad = (df["calidad_sueno"] - 1) / 4 * 10
    s_estres = (10 - df["estres"]) / 9 * 10
    s_depresion = (1 - df["depresion"]) * 10
    df["indice_sueno_salud_mental"] = (s_calidad + s_estres + s_depresion) / 3

    # 2. Actividad Física y Sedentarismo (Actividad efectiva, Horas pantalla, IMC)
    # Normalizamos a 0-10 (10 es mejor salud)
    a_activa = df["actividad_efectiva"] / 300 * 10
    a_pantalla = (12 - df["horas_pantalla"]) / 12 * 10
    # IMC: penalizamos exceso sobre 22
    exceso_imc = np.maximum(0, df["imc"] - 22)
    a_imc = np.clip(10 - (exceso_imc * 0.5), 0, 10)
    df["indice_actividad_fisica"] = (a_activa + a_pantalla + a_imc) / 3

    # 3. Hábitos Alimenticios (Azúcar)
    # Normalizamos a 0-10 (10 es mejor salud)
    al_azucar = (7 - df["porciones_azucar_dia"]) / 7 * 10
    df["indice_alimentacion"] = al_azucar # Podría incluir alcohol/tabaco si se desea

    # 4. Contexto Socio-Ambiental (SES y Entorno)
    # Normalizamos a 0-10 (10 es mejor salud)
    ses_map = {"bajo": 0, "medio": 5, "alto": 10}
    c_ses = df["nivel_socioeconomico"].map(ses_map)
    c_entorno = (10 - df["indice_entorno_riesgo"]) / 9 * 10
    df["indice_socioeconomico_ambiental"] = (c_ses + c_entorno) / 2

    return df

# --- Funciones de Regresión y Reporte (Adaptadas) ---

def build_multiple_linear_regression(df: pd.DataFrame) -> dict[str, object]:
    X = df[REGRESSION_FINAL_NUMERIC].copy()
    y = df[REGRESSION_TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    }

    # Coeficientes
    coef_df = pd.DataFrame({
        "variable": REGRESSION_FINAL_NUMERIC,
        "coeficiente": model.coef_
    }).sort_values("coeficiente", ascending=False)

    return {
        "metricas": metrics,
        "coeficientes": coef_df.to_dict(orient="records"),
        "formula": f"Salud = {model.intercept_:.2f} + " + " + ".join([f"({c:.2f} * {v})" for v, c in zip(REGRESSION_FINAL_NUMERIC, model.coef_)])
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    # 1. Cargar datos base
    datasets = {name: load_csv(INPUT_DIR / filename) for name, filename in FILES.items()}
    cell_pop = build_cell_population(datasets["obesidad"])
    prevalence_maps = {k: build_prevalence_map(datasets, k) for k in FILES.keys()}

    # 2. Generar población base
    df = sample_population(cell_pop, rng)
    df = assign_social_context(df, rng)
    df = generate_habits(df, rng)
    df = assign_bmi(df, prevalence_maps["sobrepeso"], prevalence_maps["obesidad"], rng)
    df = add_conditions(df, prevalence_maps, rng)
    
    # 3. FUSIONAR VARIABLES
    df = fuse_variables(df)
    
    # 4. Calcular score final basado en fusiones
    df = add_health_score(df, rng)

    # 5. Regresión
    results = build_multiple_linear_regression(df)

    # 6. Exportar
    csv_path = OUTPUT_DIR / "salud_mexico_simplificado.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    with open(OUTPUT_DIR / "metricas_simplificadas.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 7. Graficar importancia
    plt.figure(figsize=(10, 6))
    coefs = results["coeficientes"]
    names = [REGRESSION_LABELS.get(c["variable"], c["variable"]) for c in coefs]
    values = [c["coeficiente"] for c in coefs]
    sns.barplot(x=values, y=names, palette="viridis")
    plt.title("Importancia de Categorías Fusionadas en la Salud General")
    plt.xlabel("Coeficiente de Regresión")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_categorias.png")

    print(f"Proyecto simplificado generado en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
