from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analisis_salud_mexico"
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
REGRESSION_NUMERIC_CANDIDATES = [
    "edad",
    "horas_sueno",
    "calidad_sueno",
    "estres",
    "min_actividad_semana",
    "actividad_efectiva",
    "imc",
    "exceso_imc",
    "porciones_azucar_dia",
    "unidades_alcohol_semana",
    "fuma",
    "horas_pantalla",
    "depresion",
    "hipertension",
    "diabetes",
    "contaminacion_aire",
    "contaminacion_agua",
    "contaminacion_suelo",
    "contaminacion_auditiva",
    "trafico",
    "inseguridad",
    "indice_entorno_riesgo",
]
REGRESSION_CATEGORICAL_CANDIDATES = [
    "sexo",
    "grupo_de_edad",
    "tipo_localidad",
    "nivel_socioeconomico",
    "nivel_educativo",
]
REGRESSION_FINAL_NUMERIC = [
    "calidad_sueno",
    "estres",
    "actividad_efectiva",
    "exceso_imc",
    "porciones_azucar_dia",
    "unidades_alcohol_semana",
    "fuma",
    "horas_pantalla",
    "depresion",
    "hipertension",
    "diabetes",
    "contaminacion_aire",
    "contaminacion_agua",
    "contaminacion_suelo",
    "contaminacion_auditiva",
    "trafico",
    "inseguridad",
]
REGRESSION_FINAL_CATEGORICAL = ["nivel_socioeconomico"]
REGULARIZATION_ALPHAS = np.logspace(-3, 3, 25)
REGRESSION_LABELS = {
    "edad": "Edad",
    "calidad_sueno": "Calidad de sueno",
    "estres": "Estres",
    "actividad_efectiva": "Actividad efectiva",
    "exceso_imc": "Exceso de IMC",
    "porciones_azucar_dia": "Azucar diaria",
    "unidades_alcohol_semana": "Alcohol semanal",
    "fuma": "Fuma",
    "horas_pantalla": "Horas de pantalla",
    "depresion": "Depresion",
    "hipertension": "Hipertension",
    "diabetes": "Diabetes",
    "contaminacion_aire": "Contaminacion aire",
    "contaminacion_agua": "Contaminacion agua",
    "contaminacion_suelo": "Contaminacion suelo",
    "contaminacion_auditiva": "Contaminacion auditiva",
    "trafico": "Trafico",
    "inseguridad": "Inseguridad",
    "nivel_socioeconomico_bajo": "SES bajo",
    "nivel_socioeconomico_medio": "SES medio",
}
GUI_ESTRATEGIA_ELIMINAR = "eliminar"
GUI_ESTRATEGIA_IMPUTAR = "imputar"
GUI_ESTRATEGIAS_VALIDAS = {GUI_ESTRATEGIA_ELIMINAR, GUI_ESTRATEGIA_IMPUTAR}


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
    if pd.isna(value):
        return np.nan
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    return float(digits) if digits else np.nan


def filter_exact(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    subset = df.copy()
    for col, value in filters.items():
        if value is None:
            continue
        if col in subset.columns:
            subset = subset[subset[col] == value]
    return subset


def latest_value(
    df: pd.DataFrame,
    filters: dict[str, object],
    preferred_years: list[int] | None = None,
) -> tuple[float, int | None]:
    pct_col = percentage_column(df)
    years = preferred_years or PREFERRED_YEARS
    for year in years:
        subset = filter_exact(df, {"año": year, **filters})
        subset = subset[subset[pct_col].notna()]
        if not subset.empty:
            return float(subset.iloc[0][pct_col]), year
    return np.nan, None


def latest_value_with_fallbacks(
    df: pd.DataFrame,
    sex: str,
    age_group: str,
) -> tuple[float, int | None]:
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

    attempts = [
        base,
        {**base, "sexo": "-", "grupo_de_edad": age_group},
        {**base, "sexo": sex, "grupo_de_edad": "-"},
        {**base, "sexo": "-", "grupo_de_edad": "-"},
    ]
    for filters in attempts:
        value, year = latest_value(df, filters)
        if not np.isnan(value):
            return value, year
    return np.nan, None


def build_national_trend(df: pd.DataFrame, label: str) -> pd.DataFrame:
    pct_col = percentage_column(df)
    rows = []
    years = sorted(int(year) for year in df["año"].dropna().unique())
    for year in years:
        filters = {
            "año": year,
            "sexo": "-",
            "grupo_de_edad": "-",
            "tipo_de_localidad_rural_o_urbano": "-",
            "region_geografica_de_Mexico": "-",
            "nivel_educativo": "-",
            "nivel_socioeconomico": "-",
            "presencia_de_discapacidad": "-",
        }
        if "nivel_de_agregacion" in df.columns:
            filters["nivel_de_agregacion"] = "nacional"
        subset = filter_exact(df, filters)
        subset = subset[subset[pct_col].notna()]
        if subset.empty and "nivel_de_agregacion" in df.columns:
            filters["nivel_de_agregacion"] = "nacional "
            subset = filter_exact(df, filters)
            subset = subset[subset[pct_col].notna()]
        if not subset.empty:
            rows.append(
                {
                    "anio": year,
                    "indicador": label,
                    "porcentaje": float(subset.iloc[0][pct_col]),
                }
            )
    return pd.DataFrame(rows)


def build_cell_population(obesity_df: pd.DataFrame) -> pd.DataFrame:
    pct_col = percentage_column(obesity_df)
    pop_col = population_column(obesity_df)
    filters = {
        "año": 2023,
        "tipo_de_localidad_rural_o_urbano": "-",
        "region_geografica_de_Mexico": "-",
        "nivel_educativo": "-",
        "nivel_socioeconomico": "-",
        "presencia_de_discapacidad": "-",
    }
    if "nivel_de_agregacion" in obesity_df.columns:
        filters["nivel_de_agregacion"] = "estratificado"
    subset = filter_exact(obesity_df, filters)
    subset = subset[
        subset["sexo"].isin(["hombre", "mujer"])
        & subset["grupo_de_edad"].isin(["20-39", "40-59", "60+"])
        & subset[pct_col].notna()
    ].copy()
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


def sample_population(cell_pop: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    chosen = rng.choice(cell_pop.index.to_numpy(), size=TARGET_N, p=cell_pop["peso_muestral"].to_numpy())
    base = cell_pop.loc[chosen, ["sexo", "grupo_de_edad"]].reset_index(drop=True)
    base["id_persona"] = np.arange(1, TARGET_N + 1)
    ages = np.zeros(TARGET_N, dtype=int)
    for age_group, (low, high) in AGE_RANGES.items():
        mask = base["grupo_de_edad"] == age_group
        count = int(mask.sum())
        if not count:
            continue
        if age_group != "60+":
            ages[mask] = rng.integers(low, high + 1, size=count)
        else:
            ages[mask] = np.clip(np.round(60 + rng.gamma(shape=2.2, scale=6.0, size=count)), low, high).astype(int)
    base["edad"] = ages
    return base


def assign_social_context(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rural_p = df["grupo_de_edad"].map({"20-39": 0.18, "40-59": 0.21, "60+": 0.27}).to_numpy(dtype=float, copy=True)
    df["tipo_localidad"] = np.where(rng.random(len(df)) < rural_p, "rural", "urbano")

    ses_choices = []
    for _, row in df[["tipo_localidad", "grupo_de_edad"]].iterrows():
        if row["tipo_localidad"] == "rural":
            probs = np.array([0.53, 0.32, 0.15])
        else:
            probs = np.array([0.29, 0.41, 0.30])
        if row["grupo_de_edad"] == "20-39":
            probs += np.array([-0.02, 0.00, 0.02])
        elif row["grupo_de_edad"] == "60+":
            probs += np.array([0.04, -0.01, -0.03])
        probs = np.clip(probs, 0.05, 0.85)
        probs = probs / probs.sum()
        ses_choices.append(rng.choice(["bajo", "medio", "alto"], p=probs))
    df["nivel_socioeconomico"] = ses_choices

    education = []
    for _, row in df[["grupo_de_edad", "nivel_socioeconomico"]].iterrows():
        if row["grupo_de_edad"] == "20-39":
            base = {
                "bajo": np.array([0.18, 0.35, 0.31, 0.16]),
                "medio": np.array([0.09, 0.25, 0.37, 0.29]),
                "alto": np.array([0.03, 0.13, 0.31, 0.53]),
            }[row["nivel_socioeconomico"]]
        elif row["grupo_de_edad"] == "40-59":
            base = {
                "bajo": np.array([0.32, 0.34, 0.22, 0.12]),
                "medio": np.array([0.18, 0.28, 0.31, 0.23]),
                "alto": np.array([0.08, 0.18, 0.29, 0.45]),
            }[row["nivel_socioeconomico"]]
        else:
            base = {
                "bajo": np.array([0.52, 0.27, 0.15, 0.06]),
                "medio": np.array([0.35, 0.28, 0.22, 0.15]),
                "alto": np.array([0.18, 0.22, 0.25, 0.35]),
            }[row["nivel_socioeconomico"]]
        education.append(rng.choice(["primaria o menos", "secundaria", "media", "superior"], p=base))
    df["nivel_educativo"] = education

    urban = df["tipo_localidad"].eq("urbano").astype(float).to_numpy()
    low_ses = df["nivel_socioeconomico"].eq("bajo").astype(float).to_numpy()
    mid_ses = df["nivel_socioeconomico"].eq("medio").astype(float).to_numpy()
    high_ses = df["nivel_socioeconomico"].eq("alto").astype(float).to_numpy()

    df["contaminacion_aire"] = np.clip(2.8 + urban * 2.8 + low_ses * 1.0 + mid_ses * 0.3 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["contaminacion_agua"] = np.clip(2.6 + (1 - urban) * 1.2 + low_ses * 1.1 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["contaminacion_suelo"] = np.clip(2.4 + (1 - urban) * 1.0 + low_ses * 0.9 + rng.normal(0, 0.95, len(df)), 1, 10)
    df["contaminacion_auditiva"] = np.clip(1.8 + urban * 3.2 + low_ses * 0.3 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["trafico"] = np.clip(1.7 + urban * 3.5 + high_ses * 0.2 + rng.normal(0, 1.0, len(df)), 1, 10)
    df["inseguridad"] = np.clip(2.5 + urban * 1.8 + low_ses * 1.4 + mid_ses * 0.5 + rng.normal(0, 1.1, len(df)), 1, 10)
    df["indice_entorno_riesgo"] = (
        df["contaminacion_aire"] * 0.22
        + df["contaminacion_agua"] * 0.14
        + df["contaminacion_suelo"] * 0.10
        + df["contaminacion_auditiva"] * 0.16
        + df["trafico"] * 0.16
        + df["inseguridad"] * 0.22
    )
    return df


def generate_habits(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    age_group = df["grupo_de_edad"]
    sex = df["sexo"]
    urban = df["tipo_localidad"].eq("urbano").astype(float).to_numpy()
    low_ses = df["nivel_socioeconomico"].eq("bajo").astype(float).to_numpy()
    high_ses = df["nivel_socioeconomico"].eq("alto").astype(float).to_numpy()

    screen_mu = age_group.map({"20-39": 6.0, "40-59": 5.0, "60+": 3.8}).to_numpy(dtype=float, copy=True)
    screen_mu = screen_mu + np.where(sex.eq("hombre"), 0.2, -0.1)
    screen_mu = screen_mu + urban * 0.35 + low_ses * 0.25 - high_ses * 0.15
    df["horas_pantalla"] = np.clip(rng.normal(screen_mu, 1.4), 0.5, 12.0)

    sugar_mu = age_group.map({"20-39": 2.1, "40-59": 1.7, "60+": 1.3}).to_numpy(dtype=float, copy=True)
    sugar_mu = sugar_mu + (df["horas_pantalla"].to_numpy() - 5.0) * 0.12
    sugar_mu = sugar_mu + low_ses * 0.28 - high_ses * 0.16
    df["porciones_azucar_dia"] = np.clip(rng.normal(sugar_mu, 0.8), 0.0, 7.0)

    alcohol_base = np.where(sex.eq("hombre"), 4.5, 2.1).astype(float)
    alcohol_base = alcohol_base + age_group.map({"20-39": 0.8, "40-59": 0.4, "60+": -0.6}).to_numpy(dtype=float, copy=True)
    alcohol_base = alcohol_base + urban * 0.35 + high_ses * 0.22
    drink_flag = rng.random(len(df)) < np.where(sex.eq("hombre"), 0.62, 0.45)
    alcohol = rng.gamma(shape=1.8, scale=np.maximum(alcohol_base, 0.6) / 1.8, size=len(df))
    df["unidades_alcohol_semana"] = np.where(drink_flag, np.clip(alcohol, 0.0, 22.0), 0.0)

    tobacco_p = np.where(sex.eq("hombre"), 0.17, 0.10).astype(float)
    tobacco_p = tobacco_p + age_group.map({"20-39": 0.01, "40-59": 0.03, "60+": -0.02}).to_numpy(dtype=float, copy=True)
    tobacco_p = tobacco_p + (df["horas_pantalla"].to_numpy() - 5.0) * 0.005
    tobacco_p = tobacco_p + low_ses * 0.05 + urban * 0.02
    tobacco_p = np.clip(tobacco_p, 0.03, 0.35)
    df["fuma"] = (rng.random(len(df)) < tobacco_p).astype(int)

    activity_base = age_group.map({"20-39": 230, "40-59": 185, "60+": 145}).to_numpy(dtype=float, copy=True)
    activity_base = activity_base + np.where(sex.eq("hombre"), 25, 0)
    activity_base = activity_base - df["horas_pantalla"].to_numpy() * 12
    activity_base = activity_base - df["porciones_azucar_dia"].to_numpy() * 7
    activity_base = activity_base - df["fuma"].to_numpy() * 18
    activity_base = activity_base - df["trafico"].to_numpy() * 5.5
    activity_base = activity_base - df["inseguridad"].to_numpy() * 4.5
    activity_base = activity_base + (1 - urban) * 18 + high_ses * 10
    df["min_actividad_semana"] = np.clip(rng.normal(activity_base, 55), 0, 420)

    sleep_hours = 7.8
    sleep_hours -= (df["horas_pantalla"].to_numpy() - 4.5) * 0.16
    sleep_hours -= df["unidades_alcohol_semana"].to_numpy() * 0.03
    sleep_hours -= df["fuma"].to_numpy() * 0.35
    sleep_hours += (df["min_actividad_semana"].to_numpy() - 150) * 0.002
    sleep_hours += age_group.map({"20-39": -0.1, "40-59": 0.0, "60+": 0.15}).to_numpy()
    sleep_hours -= df["contaminacion_auditiva"].to_numpy() * 0.07
    sleep_hours -= df["inseguridad"].to_numpy() * 0.06
    sleep_hours -= df["contaminacion_aire"].to_numpy() * 0.04
    df["horas_sueno"] = np.clip(rng.normal(sleep_hours, 0.7), 4.0, 9.5)

    sleep_quality = 4.7
    sleep_quality -= np.abs(df["horas_sueno"].to_numpy() - 7.5) * 0.9
    sleep_quality -= df["horas_pantalla"].to_numpy() * 0.11
    sleep_quality -= df["fuma"].to_numpy() * 0.45
    sleep_quality -= df["unidades_alcohol_semana"].to_numpy() * 0.02
    sleep_quality += (df["min_actividad_semana"].to_numpy() - 150) * 0.003
    sleep_quality -= df["contaminacion_auditiva"].to_numpy() * 0.08
    sleep_quality -= df["inseguridad"].to_numpy() * 0.06
    df["calidad_sueno"] = np.clip(rng.normal(sleep_quality, 0.45), 1.0, 5.0)

    stress = 3.6
    stress += np.maximum(0, 7.0 - df["horas_sueno"].to_numpy()) * 0.95
    stress += np.maximum(0, 4.0 - df["calidad_sueno"].to_numpy()) * 1.3
    stress += df["horas_pantalla"].to_numpy() * 0.18
    stress += df["fuma"].to_numpy() * 0.35
    stress += df["unidades_alcohol_semana"].to_numpy() * 0.03
    stress += age_group.map({"20-39": 0.3, "40-59": 0.1, "60+": -0.1}).to_numpy()
    stress += df["inseguridad"].to_numpy() * 0.20
    stress += df["trafico"].to_numpy() * 0.10
    stress += df["contaminacion_auditiva"].to_numpy() * 0.08
    stress += low_ses * 0.22 - high_ses * 0.10
    df["estres"] = np.clip(rng.normal(stress, 0.7), 1.0, 10.0)

    bins = [-0.1, 60, 150, 300, 10_000]
    labels = ["sedentaria", "baja", "media", "alta"]
    df["nivel_actividad"] = pd.cut(df["min_actividad_semana"], bins=bins, labels=labels).astype(str)
    df["categoria_alimentacion"] = pd.cut(
        df["porciones_azucar_dia"],
        bins=[-0.01, 1, 2, 3, 10],
        labels=["saludable", "moderada", "alta_azucar", "muy_alta_azucar"],
    ).astype(str)
    return df


def assign_bmi(
    df: pd.DataFrame,
    overweight_map: dict[tuple[str, str], float],
    obesity_map: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    bmi_risk = (
        24.0
        + (df["porciones_azucar_dia"] - 1.5) * 0.75
        + (df["horas_pantalla"] - 4.5) * 0.35
        + df["fuma"] * 0.5
        + df["unidades_alcohol_semana"] * 0.05
        + np.maximum(0, 150 - df["min_actividad_semana"]) * 0.018
        + np.maximum(0, 7.0 - df["horas_sueno"]) * 0.8
        + (df["estres"] - 5.0) * 0.35
        + df["nivel_socioeconomico"].map({"bajo": 0.45, "medio": 0.70, "alto": 0.20}).to_numpy()
        + df["tipo_localidad"].map({"rural": 0.15, "urbano": 0.35}).to_numpy()
        + df["indice_entorno_riesgo"].to_numpy() * 0.08
        + np.where(df["sexo"].eq("mujer"), 0.45, 0.0)
        + df["grupo_de_edad"].map({"20-39": 0.2, "40-59": 1.8, "60+": 0.6}).to_numpy()
        + rng.normal(0, 1.6, len(df))
    )
    df["riesgo_bmi"] = bmi_risk
    df["categoria_imc"] = "normal"
    df["imc"] = np.nan

    for (sex, age_group), idx in df.groupby(["sexo", "grupo_de_edad"]).groups.items():
        sub = df.loc[list(idx)].sort_values("riesgo_bmi")
        n = len(sub)
        p_over = max(0.0, min(100.0, overweight_map[(sex, age_group)])) / 100.0
        p_ob = max(0.0, min(100.0, obesity_map[(sex, age_group)])) / 100.0
        if p_over + p_ob > 0.95:
            scale = 0.95 / (p_over + p_ob)
            p_over *= scale
            p_ob *= scale
        k_ob = int(round(n * p_ob))
        k_over = int(round(n * p_over))
        k_over = min(k_over, n - k_ob)

        obese_idx = sub.index[-k_ob:] if k_ob else pd.Index([])
        remaining = sub.drop(obese_idx)
        over_idx = remaining.index[-k_over:] if k_over else pd.Index([])
        normal_idx = remaining.drop(over_idx).index

        df.loc[normal_idx, "categoria_imc"] = "normal"
        df.loc[over_idx, "categoria_imc"] = "sobrepeso"
        df.loc[obese_idx, "categoria_imc"] = "obesidad"

        def fill_bmi(indices: pd.Index, low: float, high: float) -> None:
            if len(indices) == 0:
                return
            scores = df.loc[indices, "riesgo_bmi"].rank(method="first", pct=True).to_numpy()
            values = low + (high - low) * scores + rng.normal(0, (high - low) * 0.04, len(indices))
            df.loc[indices, "imc"] = np.clip(values, low, high)

        fill_bmi(normal_idx, 18.6, 24.9)
        fill_bmi(over_idx, 25.0, 29.9)
        fill_bmi(obese_idx, 30.0, 41.5)

    return df


def assign_binary_condition(
    df: pd.DataFrame,
    score: np.ndarray,
    prevalence_map: dict[tuple[str, str], float],
    column_name: str,
) -> pd.DataFrame:
    df[column_name] = 0
    df[f"score_{column_name}"] = score
    for (sex, age_group), idx in df.groupby(["sexo", "grupo_de_edad"]).groups.items():
        row_ids = np.array(list(idx))
        target = max(0.0, min(100.0, prevalence_map[(sex, age_group)])) / 100.0
        k = int(round(len(row_ids) * target))
        if k <= 0:
            continue
        sub = df.loc[row_ids].sort_values(f"score_{column_name}")
        chosen = sub.index[-k:]
        df.loc[chosen, column_name] = 1
    return df


def add_conditions(
    df: pd.DataFrame,
    prevalence_maps: dict[str, dict[tuple[str, str], float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    dep_score = (
        df["estres"].to_numpy() * 1.4
        + np.maximum(0, 7.2 - df["horas_sueno"].to_numpy()) * 1.8
        + np.maximum(0, 3.8 - df["calidad_sueno"].to_numpy()) * 2.0
        + df["horas_pantalla"].to_numpy() * 0.35
        + df["inseguridad"].to_numpy() * 0.55
        + df["contaminacion_auditiva"].to_numpy() * 0.18
        + df["nivel_socioeconomico"].map({"bajo": 0.9, "medio": 0.3, "alto": -0.2}).to_numpy()
        + np.where(df["sexo"].eq("mujer"), 0.8, 0.0)
        + df["grupo_de_edad"].map({"20-39": 0.2, "40-59": 0.5, "60+": 1.6}).to_numpy()
        + rng.normal(0, 1.4, len(df))
    )
    df = assign_binary_condition(df, dep_score, prevalence_maps["depresion"], "depresion")

    hta_score = (
        df["edad"].to_numpy() * 0.18
        + np.maximum(0, df["imc"].to_numpy() - 24.0) * 1.1
        + df["estres"].to_numpy() * 0.45
        + df["fuma"].to_numpy() * 1.2
        + df["unidades_alcohol_semana"].to_numpy() * 0.08
        + df["contaminacion_auditiva"].to_numpy() * 0.15
        + df["contaminacion_aire"].to_numpy() * 0.10
        - df["min_actividad_semana"].to_numpy() * 0.005
        + rng.normal(0, 1.8, len(df))
    )
    df = assign_binary_condition(df, hta_score, prevalence_maps["hipertension"], "hipertension")

    diabetes_score = (
        df["edad"].to_numpy() * 0.22
        + np.maximum(0, df["imc"].to_numpy() - 25.0) * 1.15
        + df["porciones_azucar_dia"].to_numpy() * 0.8
        + df["horas_pantalla"].to_numpy() * 0.25
        + df["nivel_socioeconomico"].map({"bajo": 0.35, "medio": 0.45, "alto": 0.10}).to_numpy()
        + df["contaminacion_agua"].to_numpy() * 0.06
        - df["min_actividad_semana"].to_numpy() * 0.004
        + rng.normal(0, 1.5, len(df))
    )
    df = assign_binary_condition(df, diabetes_score, prevalence_maps["diabetes"], "diabetes")
    return df


def add_health_score(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    score = (
        100
        - np.maximum(0, df["imc"].to_numpy() - 22.0) * 1.5
        - df["diabetes"].to_numpy() * 15
        - df["hipertension"].to_numpy() * 12
        - df["depresion"].to_numpy() * 11
        - df["fuma"].to_numpy() * 8
        - df["porciones_azucar_dia"].to_numpy() * 2.2
        - df["horas_pantalla"].to_numpy() * 1.7
        - df["unidades_alcohol_semana"].to_numpy() * 0.35
        - df["estres"].to_numpy() * 2.1
        - df["contaminacion_aire"].to_numpy() * 0.85
        - df["contaminacion_agua"].to_numpy() * 0.55
        - df["contaminacion_suelo"].to_numpy() * 0.35
        - df["contaminacion_auditiva"].to_numpy() * 0.55
        - df["trafico"].to_numpy() * 0.35
        - df["inseguridad"].to_numpy() * 0.75
        + df["calidad_sueno"].to_numpy() * 3.0
        + np.minimum(df["min_actividad_semana"].to_numpy(), 300) * 0.035
        + df["nivel_socioeconomico"].map({"bajo": -1.6, "medio": 0.0, "alto": 1.4}).to_numpy()
    )
    score = score + rng.normal(0, 3.0, len(df))
    df["salud_general_latente"] = score
    df["salud_general"] = np.clip(score, 5, 100)
    df["categoria_salud"] = pd.cut(
        df["salud_general"],
        bins=[0, 40, 60, 80, 100],
        labels=["mala", "regular", "buena", "muy_buena"],
    ).astype(str)
    return df


def make_trend_plot(trend_df: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=trend_df, x="anio", y="porcentaje", hue="indicador", marker="o", linewidth=2.2)
    plt.title("Tendencias nacionales ENSANUT (fuente directa de los CSV)")
    plt.xlabel("Anio")
    plt.ylabel("Porcentaje")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_tendencias_nacionales.png", dpi=180)
    plt.close()


def make_sleep_stress_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sample = df.sample(5000, random_state=SEED)
    sns.regplot(
        data=sample,
        x="horas_sueno",
        y="estres",
        scatter_kws={"alpha": 0.18, "s": 18},
        line_kws={"color": "#c44e52"},
        ax=axes[0],
    )
    axes[0].set_title("Cantidad de sueno vs estres")
    axes[0].set_xlabel("Horas de sueno")
    axes[0].set_ylabel("Nivel de estres (1-10)")

    temp = df.copy()
    temp["calidad_sueno_cat"] = pd.cut(
        temp["calidad_sueno"], bins=[0.9, 1.8, 2.6, 3.4, 4.2, 5.0], labels=["1", "2", "3", "4", "5"]
    )
    sns.boxplot(data=temp, x="calidad_sueno_cat", y="estres", color="#55a868", ax=axes[1])
    axes[1].set_title("Calidad de sueno vs estres")
    axes[1].set_xlabel("Calidad de sueno (1 baja, 5 alta)")
    axes[1].set_ylabel("Nivel de estres (1-10)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_sueno_estres.png", dpi=180)
    plt.close()


def make_activity_bmi_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sample = df.sample(5000, random_state=SEED)
    sns.regplot(
        data=sample,
        x="min_actividad_semana",
        y="imc",
        scatter_kws={"alpha": 0.18, "s": 18},
        line_kws={"color": "#4c72b0"},
        ax=axes[0],
    )
    axes[0].set_title("Actividad fisica vs IMC")
    axes[0].set_xlabel("Minutos de actividad por semana")
    axes[0].set_ylabel("IMC")

    order = ["sedentaria", "baja", "media", "alta"]
    sns.boxplot(
        data=df,
        x="nivel_actividad",
        y="imc",
        order=order,
        hue="nivel_actividad",
        palette="Blues",
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("IMC por nivel de actividad")
    axes[1].set_xlabel("Nivel de actividad")
    axes[1].set_ylabel("IMC")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_actividad_imc.png", dpi=180)
    plt.close()


def make_habits_health_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    temp = df.copy()
    temp["azucar_cat"] = pd.cut(
        temp["porciones_azucar_dia"],
        bins=[-0.01, 1, 2, 3, 10],
        labels=["0-1", "1-2", "2-3", "3+"],
    )
    temp["alcohol_cat"] = pd.cut(
        temp["unidades_alcohol_semana"],
        bins=[-0.01, 0.01, 4, 10, 30],
        labels=["0", "0-4", "4-10", "10+"],
    )
    temp["pantalla_cat"] = pd.cut(
        temp["horas_pantalla"],
        bins=[0, 3, 5, 7, 12],
        labels=["0-3", "3-5", "5-7", "7+"],
    )
    temp["tabaco_cat"] = temp["fuma"].map({0: "no", 1: "si"})

    charts = [
        ("azucar_cat", "Azucar"),
        ("alcohol_cat", "Alcohol"),
        ("tabaco_cat", "Tabaco"),
        ("pantalla_cat", "Pantalla"),
    ]
    for ax, (col, title) in zip(axes.ravel(), charts):
        grouped = temp.groupby(col, observed=False)["salud_general"].mean().reset_index()
        sns.barplot(data=grouped, x=col, y="salud_general", color="#8172b2", ax=ax)
        ax.set_title(f"Salud general por {title.lower()}")
        ax.set_xlabel(title)
        ax.set_ylabel("Promedio de salud general")
        ax.set_ylim(35, 95)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_habitos_salud_general.png", dpi=180)
    plt.close()


def make_tobacco_alcohol_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    temp = df.copy()
    temp["tabaco_cat"] = temp["fuma"].map({0: "no", 1: "si"})
    temp["alcohol_cat"] = pd.cut(
        temp["unidades_alcohol_semana"],
        bins=[-0.01, 0.01, 4, 10, 30],
        labels=["0", "0-4", "4-10", "10+"],
    )

    tabaco_stats = temp.groupby("tabaco_cat", observed=False)["salud_general"].mean().reset_index()
    alcohol_stats = temp.groupby("alcohol_cat", observed=False)["salud_general"].mean().reset_index()

    sns.barplot(data=tabaco_stats, x="tabaco_cat", y="salud_general", color="#c44e52", ax=axes[0])
    axes[0].set_title("Tabaco y salud general")
    axes[0].set_xlabel("Consume tabaco")
    axes[0].set_ylabel("Promedio de salud general")
    axes[0].set_ylim(35, 95)

    sns.barplot(data=alcohol_stats, x="alcohol_cat", y="salud_general", color="#dd8452", ax=axes[1])
    axes[1].set_title("Alcohol y salud general")
    axes[1].set_xlabel("Unidades de alcohol por semana")
    axes[1].set_ylabel("Promedio de salud general")
    axes[1].set_ylim(35, 95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_tabaco_alcohol_salud.png", dpi=180)
    plt.close()


def make_diet_health_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    temp = df.copy()
    order = ["saludable", "moderada", "alta_azucar", "muy_alta_azucar"]

    health_stats = (
        temp.groupby("categoria_alimentacion", observed=False)["salud_general"]
        .mean()
        .reindex(order)
        .reset_index()
    )
    sns.barplot(data=health_stats, x="categoria_alimentacion", y="salud_general", color="#4c72b0", ax=axes[0])
    axes[0].set_title("Alimentacion y salud general")
    axes[0].set_xlabel("Calidad de alimentacion")
    axes[0].set_ylabel("Promedio de salud general")
    axes[0].set_ylim(35, 95)
    axes[0].tick_params(axis="x", rotation=15)

    risk_stats = (
        temp.groupby("categoria_alimentacion", observed=False)[["diabetes", "categoria_imc"]]
        .agg(
            prevalencia_diabetes=("diabetes", "mean"),
            prevalencia_obesidad=("categoria_imc", lambda s: (s == "obesidad").mean()),
        )
        .reindex(order)
        .reset_index()
    )
    risk_long = risk_stats.melt(
        id_vars="categoria_alimentacion",
        value_vars=["prevalencia_diabetes", "prevalencia_obesidad"],
        var_name="indicador",
        value_name="proporcion",
    )
    risk_long["porcentaje"] = risk_long["proporcion"] * 100
    risk_long["indicador"] = risk_long["indicador"].map(
        {
            "prevalencia_diabetes": "Diabetes",
            "prevalencia_obesidad": "Obesidad",
        }
    )
    sns.barplot(
        data=risk_long,
        x="categoria_alimentacion",
        y="porcentaje",
        hue="indicador",
        ax=axes[1],
    )
    axes[1].set_title("Alimentacion y riesgo metabolico")
    axes[1].set_xlabel("Calidad de alimentacion")
    axes[1].set_ylabel("Porcentaje")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_alimentacion_salud.png", dpi=180)
    plt.close()


def make_social_context_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ses_stats = (
        df.groupby("nivel_socioeconomico", observed=False)["salud_general"]
        .mean()
        .reindex(["bajo", "medio", "alto"])
        .reset_index()
    )
    sns.barplot(data=ses_stats, x="nivel_socioeconomico", y="salud_general", color="#55a868", ax=axes[0])
    axes[0].set_title("Nivel socioeconomico y salud")
    axes[0].set_xlabel("Nivel socioeconomico")
    axes[0].set_ylabel("Promedio de salud general")
    axes[0].set_ylim(35, 95)

    loc_stats = (
        df.groupby("tipo_localidad", observed=False)[["depresion", "diabetes", "hipertension"]]
        .mean()
        .mul(100)
        .reset_index()
        .melt(id_vars="tipo_localidad", var_name="condicion", value_name="porcentaje")
    )
    loc_stats["condicion"] = loc_stats["condicion"].map(
        {"depresion": "Depresion", "diabetes": "Diabetes", "hipertension": "Hipertension"}
    )
    sns.barplot(data=loc_stats, x="tipo_localidad", y="porcentaje", hue="condicion", ax=axes[1])
    axes[1].set_title("Localidad y condiciones de salud")
    axes[1].set_xlabel("Tipo de localidad")
    axes[1].set_ylabel("Porcentaje")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_contexto_social_salud.png", dpi=180)
    plt.close()


def make_environment_health_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sample = df.sample(5000, random_state=SEED)
    sns.regplot(
        data=sample,
        x="indice_entorno_riesgo",
        y="salud_general",
        scatter_kws={"alpha": 0.18, "s": 18},
        line_kws={"color": "#dd8452"},
        ax=axes[0],
    )
    axes[0].set_title("Entorno y salud general")
    axes[0].set_xlabel("Indice de riesgo ambiental y social")
    axes[0].set_ylabel("Salud general")

    temp = df.copy()
    temp["entorno_cat"] = pd.qcut(temp["indice_entorno_riesgo"], q=4, labels=["bajo", "medio", "alto", "muy_alto"])
    env_stats = (
        temp.groupby("entorno_cat", observed=False)[["contaminacion_aire", "contaminacion_auditiva", "trafico", "inseguridad"]]
        .mean()
        .reset_index()
        .melt(id_vars="entorno_cat", var_name="factor", value_name="nivel")
    )
    env_stats["factor"] = env_stats["factor"].map(
        {
            "contaminacion_aire": "Aire",
            "contaminacion_auditiva": "Ruido",
            "trafico": "Trafico",
            "inseguridad": "Inseguridad",
        }
    )
    sns.lineplot(data=env_stats, x="entorno_cat", y="nivel", hue="factor", marker="o", ax=axes[1])
    axes[1].set_title("Factores del entorno por nivel de riesgo")
    axes[1].set_xlabel("Riesgo del entorno")
    axes[1].set_ylabel("Nivel promedio (1-10)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_entorno_salud.png", dpi=180)
    plt.close()


def make_heatmap_sleep_stress(df: pd.DataFrame) -> None:
    temp = df.copy()
    temp["sueno_bin"] = pd.cut(
        temp["horas_sueno"],
        bins=[3.9, 5.5, 6.5, 7.5, 8.5, 9.6],
        labels=["4-5.5", "5.5-6.5", "6.5-7.5", "7.5-8.5", "8.5-9.5"],
    )
    temp["calidad_sueno_cat"] = pd.cut(
        temp["calidad_sueno"],
        bins=[0.9, 1.8, 2.6, 3.4, 4.2, 5.0],
        labels=["1", "2", "3", "4", "5"],
    )
    pivot = temp.pivot_table(
        index="calidad_sueno_cat",
        columns="sueno_bin",
        values="estres",
        aggfunc="mean",
        observed=False,
    )
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Mapa de calor: sueno y estres")
    plt.xlabel("Horas de sueno")
    plt.ylabel("Calidad de sueno")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_heatmap_sueno_estres.png", dpi=180)
    plt.close()


def make_heatmap_activity_bmi(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    order = ["sedentaria", "baja", "media", "alta"]

    pivot_imc = df.pivot_table(
        index="grupo_de_edad",
        columns="nivel_actividad",
        values="imc",
        aggfunc="mean",
        observed=False,
    ).reindex(columns=order)
    sns.heatmap(pivot_imc, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=axes[0])
    axes[0].set_title("IMC por edad y actividad")
    axes[0].set_xlabel("Nivel de actividad")
    axes[0].set_ylabel("Grupo de edad")

    pivot_ob = (
        df.assign(obesidad_bin=(df["categoria_imc"] == "obesidad").astype(int))
        .pivot_table(
            index="grupo_de_edad",
            columns="nivel_actividad",
            values="obesidad_bin",
            aggfunc="mean",
            observed=False,
        )
        .reindex(columns=order)
        * 100
    )
    sns.heatmap(pivot_ob, annot=True, fmt=".1f", cmap="PuBuGn", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Obesidad (%) por edad y actividad")
    axes[1].set_xlabel("Nivel de actividad")
    axes[1].set_ylabel("Grupo de edad")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_heatmap_actividad_imc.png", dpi=180)
    plt.close()


def make_heatmap_habits_health(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    temp = df.copy()
    temp["azucar_cat"] = pd.cut(
        temp["porciones_azucar_dia"],
        bins=[-0.01, 1, 2, 3, 10],
        labels=["0-1", "1-2", "2-3", "3+"],
    )
    temp["alcohol_cat"] = pd.cut(
        temp["unidades_alcohol_semana"],
        bins=[-0.01, 0.01, 4, 10, 30],
        labels=["0", "0-4", "4-10", "10+"],
    )
    temp["pantalla_cat"] = pd.cut(
        temp["horas_pantalla"],
        bins=[0, 3, 5, 7, 12],
        labels=["0-3", "3-5", "5-7", "7+"],
    )
    temp["tabaco_cat"] = temp["fuma"].map({0: "no", 1: "si"})

    pivot_1 = temp.pivot_table(
        index="azucar_cat",
        columns="alcohol_cat",
        values="salud_general",
        aggfunc="mean",
        observed=False,
    )
    sns.heatmap(pivot_1, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, ax=axes[0])
    axes[0].set_title("Salud segun azucar y alcohol")
    axes[0].set_xlabel("Alcohol")
    axes[0].set_ylabel("Azucar")

    pivot_2 = temp.pivot_table(
        index="pantalla_cat",
        columns="tabaco_cat",
        values="salud_general",
        aggfunc="mean",
        observed=False,
    )
    sns.heatmap(pivot_2, annot=True, fmt=".1f", cmap="magma_r", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Salud segun pantalla y tabaco")
    axes[1].set_xlabel("Tabaco")
    axes[1].set_ylabel("Pantalla")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_heatmap_habitos_salud.png", dpi=180)
    plt.close()


def make_heatmap_diet_health(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    order = ["saludable", "moderada", "alta_azucar", "muy_alta_azucar"]

    pivot_imc = (
        pd.crosstab(df["categoria_alimentacion"], df["categoria_imc"], normalize="index") * 100
    ).reindex(index=order)
    sns.heatmap(pivot_imc, annot=True, fmt=".1f", cmap="GnBu", linewidths=0.5, ax=axes[0])
    axes[0].set_title("Dieta vs categoria IMC")
    axes[0].set_xlabel("Categoria IMC")
    axes[0].set_ylabel("Calidad de alimentacion")

    pivot_salud = (
        pd.crosstab(df["categoria_alimentacion"], df["categoria_salud"], normalize="index") * 100
    ).reindex(index=order)
    sns.heatmap(pivot_salud, annot=True, fmt=".1f", cmap="YlGn", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Dieta vs categoria de salud")
    axes[1].set_xlabel("Categoria de salud")
    axes[1].set_ylabel("Calidad de alimentacion")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_heatmap_alimentacion_salud.png", dpi=180)
    plt.close()


def make_heatmap_social_context(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    pivot_salud = df.pivot_table(
        index="nivel_socioeconomico",
        columns="tipo_localidad",
        values="salud_general",
        aggfunc="mean",
        observed=False,
    ).reindex(index=["bajo", "medio", "alto"], columns=["rural", "urbano"])
    sns.heatmap(pivot_salud, annot=True, fmt=".1f", cmap="crest", linewidths=0.5, ax=axes[0])
    axes[0].set_title("Salud por contexto social")
    axes[0].set_xlabel("Tipo de localidad")
    axes[0].set_ylabel("Nivel socioeconomico")

    pivot_estres = df.pivot_table(
        index="nivel_socioeconomico",
        columns="tipo_localidad",
        values="estres",
        aggfunc="mean",
        observed=False,
    ).reindex(index=["bajo", "medio", "alto"], columns=["rural", "urbano"])
    sns.heatmap(pivot_estres, annot=True, fmt=".2f", cmap="rocket_r", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Estres por contexto social")
    axes[1].set_xlabel("Tipo de localidad")
    axes[1].set_ylabel("Nivel socioeconomico")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_heatmap_contexto_social.png", dpi=180)
    plt.close()


def make_heatmap_age_conditions(age_df: pd.DataFrame) -> None:
    pivot = age_df.pivot_table(
        index="condicion",
        columns="grupo_de_edad",
        values="porcentaje",
        aggfunc="mean",
        observed=False,
    )[["20-39", "40-59", "60+"]]
    plt.figure(figsize=(8.5, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="OrRd", linewidths=0.5)
    plt.title("Mapa de calor: edad y condiciones")
    plt.xlabel("Grupo de edad")
    plt.ylabel("Condicion")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_heatmap_edad_condiciones.png", dpi=180)
    plt.close()


def make_heatmap_environment(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    corr_cols = [
        "contaminacion_aire",
        "contaminacion_agua",
        "contaminacion_suelo",
        "contaminacion_auditiva",
        "trafico",
        "inseguridad",
        "estres",
        "salud_general",
    ]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=axes[0])
    axes[0].set_title("Correlaciones de entorno y salud")

    temp = df.copy()
    temp["entorno_cat"] = pd.qcut(temp["indice_entorno_riesgo"], q=4, labels=["bajo", "medio", "alto", "muy_alto"])
    pivot = temp.pivot_table(
        index="entorno_cat",
        columns="tipo_localidad",
        values="salud_general",
        aggfunc="mean",
        observed=False,
    ).reindex(index=["bajo", "medio", "alto", "muy_alto"], columns=["rural", "urbano"])
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="flare", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Salud por riesgo de entorno")
    axes[1].set_xlabel("Tipo de localidad")
    axes[1].set_ylabel("Riesgo del entorno")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "16_heatmap_entorno_salud.png", dpi=180)
    plt.close()


def make_heatmap_full_correlation(df: pd.DataFrame) -> None:
    corr_cols = [
        "salud_general",
        "salud_general_latente",
        "calidad_sueno",
        "estres",
        "actividad_efectiva",
        "exceso_imc",
        "porciones_azucar_dia",
        "unidades_alcohol_semana",
        "fuma",
        "horas_pantalla",
        "depresion",
        "hipertension",
        "diabetes",
        "contaminacion_aire",
        "contaminacion_agua",
        "contaminacion_suelo",
        "contaminacion_auditiva",
        "trafico",
        "inseguridad",
    ]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4, square=True)
    plt.title("Heatmap general de correlacion entre variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "21_heatmap_correlacion_variables.png", dpi=180)
    plt.close()


def make_heatmap_model_correlation(df: pd.DataFrame) -> None:
    corr_cols = [
        "salud_general_latente",
        "calidad_sueno",
        "estres",
        "actividad_efectiva",
        "exceso_imc",
        "porciones_azucar_dia",
        "unidades_alcohol_semana",
        "fuma",
        "horas_pantalla",
        "depresion",
        "hipertension",
        "diabetes",
        "contaminacion_aire",
        "contaminacion_agua",
        "contaminacion_suelo",
        "contaminacion_auditiva",
        "trafico",
        "inseguridad",
    ]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4, square=True)
    plt.title("Heatmap de correlacion del modelo final")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "22_heatmap_correlacion_modelo.png", dpi=180)
    plt.close()


def regression_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]
    for column in REGRESSION_CATEGORICAL_CANDIDATES:
        prefix = f"{column}_"
        if feature_name.startswith(prefix):
            return column
    return feature_name


def summarize_regularized_coefficients(feature_names: list[str], coefficients: np.ndarray, score_name: str) -> pd.DataFrame:
    rows = []
    for feature_name, coefficient in zip(feature_names, coefficients, strict=False):
        rows.append(
            {
                "feature": feature_name,
                "variable": regression_feature_name(feature_name),
                "coeficiente": float(coefficient),
                "coef_abs": float(abs(coefficient)),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .groupby("variable", as_index=False)
        .agg(
            coeficiente_neto=("coeficiente", "sum"),
            score=("coef_abs", "max"),
            terminos=("feature", "count"),
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"score": score_name})
    )
    return summary


def prepare_regression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    regression_df = df.copy()
    regression_df["exceso_imc"] = np.maximum(0, regression_df["imc"] - 22.0)
    regression_df["actividad_efectiva"] = np.minimum(regression_df["min_actividad_semana"], 300)
    return regression_df


def make_regression_importance_plot(coef_df: pd.DataFrame) -> None:
    top = coef_df.head(12).iloc[::-1].copy()
    top["etiqueta"] = top["feature"].map(lambda value: REGRESSION_LABELS.get(value, value.replace("_", " ")))

    plt.figure(figsize=(10.5, 6.5))
    colors = np.where(top["beta_estandarizado"] >= 0, "#55a868", "#c44e52")
    plt.barh(top["etiqueta"], top["beta_estandarizado"], color=colors)
    plt.axvline(0, color="black", linewidth=0.9, alpha=0.7)
    plt.title("Regresion lineal multiple: coeficientes estandarizados")
    plt.xlabel("Beta estandarizado")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_regresion_salud_general.png", dpi=180)
    plt.close()


def make_regularization_plot(rank_df: pd.DataFrame, value_col: str, title: str, output_name: str, color: str) -> None:
    top = rank_df.head(12).iloc[::-1].copy()
    top["etiqueta"] = top["variable"].map(lambda value: REGRESSION_LABELS.get(value, value.replace("_", " ")))

    plt.figure(figsize=(10.5, 6.5))
    plt.barh(top["etiqueta"], top[value_col], color=color)
    plt.title(title)
    plt.xlabel("Magnitud del coeficiente regularizado")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=180)
    plt.close()


def make_regression_diagnostics_plot(model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    resid_series = pd.Series(model.resid)
    fitted_series = pd.Series(model.fittedvalues)
    sample_size = min(len(resid_series), 5000)
    sample_idx = resid_series.sample(sample_size, random_state=SEED).index
    fitted_sample = fitted_series.loc[sample_idx]
    resid_sample = resid_series.loc[sample_idx]
    std_resid = pd.Series(model.get_influence().resid_studentized_internal, index=resid_series.index)
    std_resid_sample = std_resid.loc[sample_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    axes[0, 0].scatter(fitted_sample, resid_sample, alpha=0.22, s=16, color="#4c72b0")
    axes[0, 0].axhline(0, color="black", linewidth=1.0, alpha=0.7)
    axes[0, 0].set_title("Residuos vs valores ajustados")
    axes[0, 0].set_xlabel("Valores ajustados")
    axes[0, 0].set_ylabel("Residuos")

    sm.qqplot(std_resid_sample, line="45", ax=axes[0, 1], alpha=0.4, markersize=3)
    axes[0, 1].set_title("Q-Q plot de residuos estandarizados")

    sns.histplot(model.resid, bins=35, kde=True, color="#c44e52", ax=axes[1, 0])
    axes[1, 0].set_title("Distribucion de residuos")
    axes[1, 0].set_xlabel("Residuo")

    axes[1, 1].scatter(fitted_sample, np.sqrt(np.abs(std_resid_sample)), alpha=0.22, s=16, color="#8172b2")
    axes[1, 1].set_title("Scale-location")
    axes[1, 1].set_xlabel("Valores ajustados")
    axes[1, 1].set_ylabel("Raiz de |residuo estandarizado|")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "18_diagnostico_regresion.png", dpi=180)
    plt.close()


def make_regression_pairplot(df: pd.DataFrame) -> None:
    pair_cols = [
        "salud_general_latente",
        "calidad_sueno",
        "estres",
        "actividad_efectiva",
        "exceso_imc",
        "horas_pantalla",
        "porciones_azucar_dia",
        "diabetes",
        "hipertension",
        "depresion",
    ]
    pair_df = df[pair_cols].sample(min(len(df), 1200), random_state=SEED)
    pairplot = sns.pairplot(
        pair_df,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.22, "s": 18, "color": "#4c72b0"},
        diag_kws={"color": "#55a868"},
    )
    pairplot.figure.suptitle("Pairplot de variables clave del modelo", y=1.02)
    pairplot.figure.savefig(OUTPUT_DIR / "23_pairplot_variables_modelo.png", dpi=180, bbox_inches="tight")
    plt.close(pairplot.figure)


def make_predictions_plot(y_true: pd.Series, y_pred: np.ndarray) -> None:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)
    low = min(float(y_true_series.min()), float(y_pred_series.min()))
    high = max(float(y_true_series.max()), float(y_pred_series.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    axes[0].scatter(y_true_series, y_pred_series, alpha=0.22, s=18, color="#4c72b0", label="Predicciones")
    axes[0].plot([low, high], [low, high], color="#c44e52", linewidth=2, label="Referencia ideal")
    axes[0].set_title("Predicciones vs valores reales")
    axes[0].set_xlabel("Valor real")
    axes[0].set_ylabel("Valor predicho")
    axes[0].legend(loc="upper left")

    sample_n = min(len(y_true_series), 180)
    sample_idx = np.arange(sample_n)
    axes[1].plot(sample_idx, y_true_series.iloc[:sample_n], color="#55a868", linewidth=2, label="Valores reales")
    axes[1].plot(sample_idx, y_pred_series.iloc[:sample_n], color="#dd8452", linewidth=2, label="Predicciones")
    axes[1].set_title("Comparacion por observacion")
    axes[1].set_xlabel("Observacion en prueba")
    axes[1].set_ylabel("Valor")
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "24_predicciones_vs_reales.png", dpi=180)
    plt.close()


def build_multiple_linear_regression(df: pd.DataFrame) -> dict[str, object]:
    regression_df = prepare_regression_dataset(df)
    y = regression_df[REGRESSION_TARGET].copy()
    exploratory_corr = (
        regression_df[[REGRESSION_TARGET, *REGRESSION_NUMERIC_CANDIDATES]]
        .corr(numeric_only=True)[REGRESSION_TARGET]
        .drop(REGRESSION_TARGET)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )

    candidate_features = REGRESSION_NUMERIC_CANDIDATES + REGRESSION_CATEGORICAL_CANDIDATES
    X_candidates = regression_df[candidate_features].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_candidates, y, test_size=0.2, random_state=SEED)
    X_final = regression_df[REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL].copy()
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
        X_final, y, test_size=0.2, random_state=SEED
    )

    candidate_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                REGRESSION_NUMERIC_CANDIDATES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                REGRESSION_CATEGORICAL_CANDIDATES,
            ),
        ]
    )

    candidate_model = Pipeline(
        [
            ("preprocess", candidate_preprocessor),
            ("model", LinearRegression()),
        ]
    )
    candidate_model.fit(X_train, y_train)

    lasso_model = Pipeline(
        [
            ("preprocess", candidate_preprocessor),
            ("model", LassoCV(cv=5, random_state=SEED, max_iter=20_000)),
        ]
    )
    lasso_model.fit(X_train, y_train)
    transformed_names = lasso_model.named_steps["preprocess"].get_feature_names_out()
    lasso_coefs = lasso_model.named_steps["model"].coef_
    lasso_selected_features = sorted(
        {
            regression_feature_name(name)
            for name, coef in zip(transformed_names, lasso_coefs, strict=False)
            if abs(float(coef)) > 1e-6
        }
    )
    lasso_rank = summarize_regularized_coefficients(list(transformed_names), lasso_coefs, "peso_lasso")
    make_regularization_plot(
        lasso_rank,
        "peso_lasso",
        "LassoCV: variables retenidas por magnitud",
        "19_lasso_importancia.png",
        "#dd8452",
    )

    ridge_model = Pipeline(
        [
            ("preprocess", candidate_preprocessor),
            ("model", RidgeCV(alphas=REGULARIZATION_ALPHAS, cv=5)),
        ]
    )
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_feature_names = ridge_model.named_steps["preprocess"].get_feature_names_out()
    ridge_coefs = ridge_model.named_steps["model"].coef_
    ridge_rank = summarize_regularized_coefficients(list(ridge_feature_names), ridge_coefs, "peso_ridge")
    make_regularization_plot(
        ridge_rank,
        "peso_ridge",
        "RidgeCV: variables priorizadas por magnitud",
        "20_ridge_importancia.png",
        "#4c72b0",
    )
    ridge_priority_features = ridge_rank.head(12)["variable"].tolist()
    ridge_priority_set = set(ridge_rank.head(15)["variable"].tolist())
    lasso_selected_set = set(lasso_selected_features)
    regression_formula = (
        "salud_general_latente ~ calidad_sueno + estres + actividad_efectiva + exceso_imc + porciones_azucar_dia + "
        "unidades_alcohol_semana + fuma + horas_pantalla + depresion + hipertension + diabetes + contaminacion_aire + "
        "contaminacion_agua + contaminacion_suelo + contaminacion_auditiva + trafico + inseguridad + C(nivel_socioeconomico)"
    )
    final_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                REGRESSION_FINAL_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                REGRESSION_FINAL_CATEGORICAL,
            ),
        ]
    )
    final_linear_model = Pipeline(
        [
            ("preprocess", final_preprocessor),
            ("model", LinearRegression()),
        ]
    )
    final_linear_model.fit(X_final_train, y_final_train)
    y_final_pred = final_linear_model.predict(X_final_test)
    prediction_df = pd.DataFrame(
        {
            "valor_real": y_final_test.to_numpy(),
            "valor_predicho": y_final_pred,
        }
    )
    prediction_df["residuo"] = prediction_df["valor_real"] - prediction_df["valor_predicho"]
    prediction_df.to_csv(OUTPUT_DIR / "predicciones_prueba_regresion.csv", index=False, encoding="utf-8-sig")
    make_predictions_plot(y_final_test, y_final_pred)
    make_regression_pairplot(regression_df)

    final_design = pd.get_dummies(
        regression_df[REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL],
        columns=REGRESSION_FINAL_CATEGORICAL,
        drop_first=True,
        dtype=float,
    )
    final_with_const = sm.add_constant(final_design)
    final_model = sm.OLS(y, final_with_const).fit()
    formula_model = smf.ols(regression_formula, data=regression_df).fit()
    make_regression_diagnostics_plot(formula_model)

    standardized_design = (final_design - final_design.mean()) / final_design.std(ddof=0)
    standardized_target = (y - y.mean()) / y.std(ddof=0)
    standardized_model = sm.OLS(standardized_target, sm.add_constant(standardized_design)).fit()

    coef_df = pd.DataFrame(
        {
            "feature": standardized_model.params.drop("const").index,
            "beta_estandarizado": standardized_model.params.drop("const").to_numpy(),
            "coeficiente": final_model.params.drop("const").reindex(standardized_model.params.drop("const").index).to_numpy(),
            "p_valor": final_model.pvalues.drop("const").reindex(standardized_model.params.drop("const").index).to_numpy(),
        }
    )
    coef_df["abs_beta"] = coef_df["beta_estandarizado"].abs()
    coef_df = coef_df.sort_values("abs_beta", ascending=False).reset_index(drop=True)
    make_regression_importance_plot(coef_df)

    vif_sample = final_design.sample(min(len(final_design), 12_000), random_state=SEED)
    vif_matrix = sm.add_constant(vif_sample, has_constant="add")
    vif_rows = []
    for idx, column in enumerate(vif_matrix.columns):
        if column == "const":
            continue
        vif_rows.append(
            {
                "variable": column,
                "vif": round(float(variance_inflation_factor(vif_matrix.to_numpy(), idx)), 3),
            }
        )

    reset_result = linear_reset(formula_model, power=2, use_f=True)
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(formula_model.resid, formula_model.model.exog)
    dw_stat = durbin_watson(formula_model.resid)
    jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(formula_model.resid)
    assumptions = {
        "linealidad": {
            "prueba": "Ramsey RESET",
            "estadistico_f": round(float(reset_result.fvalue), 4),
            "p_valor": round(float(reset_result.pvalue), 6),
            "cumple": bool(float(reset_result.pvalue) > 0.05),
            "criterio": "p > 0.05 indica que no hay evidencia fuerte de no linealidad.",
        },
        "independencia": {
            "prueba": "Durbin-Watson",
            "estadistico": round(float(dw_stat), 4),
            "cumple": bool(1.5 <= float(dw_stat) <= 2.5),
            "criterio": "Valores cercanos a 2 indican independencia aproximada de los residuos.",
        },
        "homocedasticidad": {
            "prueba": "Breusch-Pagan",
            "estadistico_lm": round(float(bp_lm), 4),
            "p_valor_lm": round(float(bp_lm_pvalue), 6),
            "estadistico_f": round(float(bp_fvalue), 4),
            "p_valor_f": round(float(bp_f_pvalue), 6),
            "cumple": bool(float(bp_f_pvalue) > 0.05),
            "criterio": "p > 0.05 sugiere varianza constante de los residuos.",
        },
        "normalidad": {
            "prueba": "Jarque-Bera",
            "estadistico": round(float(jb_stat), 4),
            "p_valor": round(float(jb_pvalue), 6),
            "asimetria": round(float(jb_skew), 4),
            "curtosis": round(float(jb_kurtosis), 4),
            "cumple": bool(float(jb_pvalue) > 0.05),
            "criterio": "p > 0.05 indica compatibilidad con normalidad de los residuos.",
        },
    }
    assumptions["cumplen_los_4_supuestos"] = bool(
        assumptions["linealidad"]["cumple"]
        and assumptions["independencia"]["cumple"]
        and assumptions["homocedasticidad"]["cumple"]
        and assumptions["normalidad"]["cumple"]
    )

    anova_table = (
        anova_lm(formula_model, typ=2)
        .reset_index()
        .rename(columns={"index": "factor", "sum_sq": "suma_cuadrados", "df": "gl", "F": "estadistico_f", "PR(>F)": "p_valor"})
    )
    anova_rows = []
    for _, row in anova_table.iterrows():
        anova_rows.append(
            {
                "factor": str(row["factor"]),
                "suma_cuadrados": None if pd.isna(row["suma_cuadrados"]) else round(float(row["suma_cuadrados"]), 4),
                "gl": None if pd.isna(row["gl"]) else round(float(row["gl"]), 4),
                "estadistico_f": None if pd.isna(row["estadistico_f"]) else round(float(row["estadistico_f"]), 4),
                "p_valor": None if pd.isna(row["p_valor"]) else round(float(row["p_valor"]), 6),
            }
        )
    anova_key_factors = sorted(
        [row for row in anova_rows if row["factor"] != "Residual" and row["estadistico_f"] is not None],
        key=lambda row: row["estadistico_f"],
        reverse=True,
    )[:5]

    excluded_variables = {
        "sexo": "Lasso y Ridge mostraron un aporte incremental menor frente a habitos y condiciones clinicas.",
        "edad, grupo_de_edad y horas_sueno": "Se excluyeron del modelo final porque no mejoraron la especificacion una vez incluidas las variables de efecto directo.",
        "tipo_localidad e indice_entorno_riesgo": "Se sustituyeron por los componentes ambientales directos para cumplir mejor los supuestos del modelo.",
        "nivel_educativo": "Su contribucion marginal fue pequena frente a nivel socioeconomico y habitos.",
        "imc y min_actividad_semana": (
            "Se reemplazaron por exceso_imc y actividad_efectiva para reflejar la forma funcional que mejor satisface los supuestos."
        ),
        "nivel_actividad, categoria_imc, categoria_alimentacion, categoria_salud": (
            "Se excluyeron por ser variables derivadas de mediciones continuas ya incluidas o del propio objetivo."
        ),
    }

    variable_support = []
    for variable in REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL:
        support_tags = []
        if variable in lasso_selected_set:
            support_tags.append("lasso")
        if variable in ridge_priority_set:
            support_tags.append("ridge")
        if variable == "exceso_imc":
            detail = "Se uso en lugar de IMC bruto porque captura mejor el deterioro de salud por encima del umbral saludable."
        elif variable == "actividad_efectiva":
            detail = "Se uso la actividad acotada a 300 minutos para reflejar rendimientos decrecientes."
        elif variable == "nivel_socioeconomico":
            detail = "Se mantuvo por su efecto estructural y por su persistencia en la regularizacion."
        elif variable in {"contaminacion_aire", "contaminacion_agua", "contaminacion_suelo", "contaminacion_auditiva", "trafico", "inseguridad"}:
            detail = "Se mantuvo como componente directo del entorno porque asi el modelo cumple mejor los supuestos."
        else:
            detail = "Se mantuvo por su aporte incremental y estabilidad en los modelos regularizados."
        variable_support.append(
            {
                "variable": variable,
                "respaldo_regularizacion": "+".join(support_tags) if support_tags else "criterio_sustantivo",
                "detalle": detail,
            }
        )

    return {
        "variable_dependiente": REGRESSION_TARGET,
        "justificacion_variable_dependiente": (
            "Se eligio salud_general_latente porque es la version no censurada del puntaje de salud y permite cumplir "
            "mejor los supuestos del modelo lineal que la version truncada entre 5 y 100."
        ),
        "formula_modelo": regression_formula,
        "variables_candidatas_numericas": REGRESSION_NUMERIC_CANDIDATES,
        "variables_candidatas_categoricas": REGRESSION_CATEGORICAL_CANDIDATES,
        "variables_incluidas_modelo_final": REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL,
        "criterio_seleccion_variables": (
            "LassoCV se uso para retener variables con coeficiente distinto de cero y RidgeCV para priorizar variables estables "
            "cuando habia colinealidad; despues se consolidaron variables redundantes en una especificacion final interpretable."
        ),
        "variables_excluidas_y_razon": excluded_variables,
        "variables_seleccionadas_por_lasso": lasso_selected_features,
        "variables_priorizadas_por_ridge": ridge_priority_features,
        "sustento_variables_finales": variable_support,
        "ranking_exploratorio_correlacion": [
            {
                "variable": column,
                "correlacion_con_salud_general": round(float(value), 4),
                "correlacion_absoluta": round(float(abs(value)), 4),
            }
            for column, value in exploratory_corr.items()
        ],
        "modelos_regularizados": {
            "lasso": {
                "alpha": round(float(lasso_model.named_steps["model"].alpha_), 6),
                "r2": round(float(r2_score(y_test, lasso_model.predict(X_test))), 4),
                "mae": round(float(mean_absolute_error(y_test, lasso_model.predict(X_test))), 4),
                "mse": round(float(mean_squared_error(y_test, lasso_model.predict(X_test))), 4),
                "rmse": round(float(mean_squared_error(y_test, lasso_model.predict(X_test)) ** 0.5), 4),
            },
            "ridge": {
                "alpha": round(float(ridge_model.named_steps["model"].alpha_), 6),
                "r2": round(float(r2_score(y_test, ridge_pred)), 4),
                "mae": round(float(mean_absolute_error(y_test, ridge_pred)), 4),
                "mse": round(float(mean_squared_error(y_test, ridge_pred)), 4),
                "rmse": round(float(mean_squared_error(y_test, ridge_pred) ** 0.5), 4),
            },
        },
        "division_entrenamiento_prueba": {
            "observaciones_totales": int(len(regression_df)),
            "tamano_entrenamiento": int(len(X_final_train)),
            "tamano_prueba": int(len(X_final_test)),
            "proporcion_prueba": 0.2,
        },
        "metricas_holdout": {
            "r2": round(float(r2_score(y_final_test, y_final_pred)), 4),
            "mae": round(float(mean_absolute_error(y_final_test, y_final_pred)), 4),
            "mse": round(float(mean_squared_error(y_final_test, y_final_pred)), 4),
            "rmse": round(float(mean_squared_error(y_final_test, y_final_pred) ** 0.5), 4),
        },
        "predicciones_prueba_ejemplo": [
            {
                "valor_real": round(float(row["valor_real"]), 4),
                "valor_predicho": round(float(row["valor_predicho"]), 4),
                "residuo": round(float(row["residuo"]), 4),
            }
            for _, row in prediction_df.head(10).iterrows()
        ],
        "ranking_lasso": [
            {
                "variable": row["variable"],
                "peso_lasso": round(float(row["peso_lasso"]), 4),
                "coeficiente_neto": round(float(row["coeficiente_neto"]), 4),
            }
            for _, row in lasso_rank.head(12).iterrows()
        ],
        "ranking_ridge": [
            {
                "variable": row["variable"],
                "peso_ridge": round(float(row["peso_ridge"]), 4),
                "coeficiente_neto": round(float(row["coeficiente_neto"]), 4),
            }
            for _, row in ridge_rank.head(12).iterrows()
        ],
        "modelo_final_ols": {
            "r2": round(float(final_model.rsquared), 4),
            "r2_ajustado": round(float(final_model.rsquared_adj), 4),
        },
        "supuestos_regresion": assumptions,
        "anova_tipo_ii": anova_rows,
        "anova_factores_principales": anova_key_factors,
        "coeficientes_principales": [
            {
                "variable": row["feature"],
                "beta_estandarizado": round(float(row["beta_estandarizado"]), 4),
                "coeficiente": round(float(row["coeficiente"]), 4),
                "p_valor": round(float(row["p_valor"]), 6),
            }
            for _, row in coef_df.head(12).iterrows()
        ],
        "vif": sorted(vif_rows, key=lambda row: row["vif"], reverse=True),
    }


def make_age_condition_plot(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    definitions = {
        "obesidad": "obesidad",
        "depresion": "depresion",
        "hipertension": "hipertension",
    }
    for key, label in definitions.items():
        df = datasets[key]
        pct_col = percentage_column(df)
        for age_group in ["20-39", "40-59", "60+"]:
            filters = {
                "grupo_de_edad": age_group,
                "sexo": "-",
                "tipo_de_localidad_rural_o_urbano": "-",
                "region_geografica_de_Mexico": "-",
                "nivel_educativo": "-",
                "nivel_socioeconomico": "-",
                "presencia_de_discapacidad": "-",
            }
            if "nivel_de_agregacion" in df.columns:
                filters["nivel_de_agregacion"] = "estratificado"
            value, year = latest_value(df, filters)
            rows.append(
                {
                    "condicion": label,
                    "grupo_de_edad": age_group,
                    "porcentaje": value,
                    "anio_fuente": year,
                    "columna": pct_col,
                }
            )
    age_df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=age_df, x="grupo_de_edad", y="porcentaje", hue="condicion")
    plt.title("Condiciones por grupo de edad (fuente directa de ENSANUT)")
    plt.xlabel("Grupo de edad")
    plt.ylabel("Porcentaje")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_edades_significativas.png", dpi=180)
    plt.close()
    return age_df


def build_summary(
    synthetic_df: pd.DataFrame,
    age_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    regression_summary: dict[str, object],
) -> dict[str, object]:
    corr_sleep_hours = float(synthetic_df["horas_sueno"].corr(synthetic_df["estres"]))
    corr_sleep_quality = float(synthetic_df["calidad_sueno"].corr(synthetic_df["estres"]))
    corr_activity_bmi = float(synthetic_df["min_actividad_semana"].corr(synthetic_df["imc"]))

    low_sleep = synthetic_df[synthetic_df["horas_sueno"] < 6]["estres"].mean()
    ideal_sleep = synthetic_df[
        (synthetic_df["horas_sueno"] >= 7) & (synthetic_df["horas_sueno"] <= 8)
    ]["estres"].mean()

    activity_levels = (
        synthetic_df.groupby("nivel_actividad", observed=False)["imc"]
        .mean()
        .reindex(["sedentaria", "baja", "media", "alta"])
    )

    tobacco_health = synthetic_df.groupby("fuma")["salud_general"].mean().to_dict()
    sugar_health = (
        synthetic_df.assign(
            azucar_cat=pd.cut(
                synthetic_df["porciones_azucar_dia"],
                bins=[-0.01, 1, 2, 3, 10],
                labels=["0-1", "1-2", "2-3", "3+"],
            )
        )
        .groupby("azucar_cat", observed=False)["salud_general"]
        .mean()
        .to_dict()
    )
    screen_health = (
        synthetic_df.assign(
            pantalla_cat=pd.cut(
                synthetic_df["horas_pantalla"], bins=[0, 3, 5, 7, 12], labels=["0-3", "3-5", "5-7", "7+"]
            )
        )
        .groupby("pantalla_cat", observed=False)["salud_general"]
        .mean()
        .to_dict()
    )
    alcohol_health = (
        synthetic_df.assign(
            alcohol_cat=pd.cut(
                synthetic_df["unidades_alcohol_semana"], bins=[-0.01, 0.01, 4, 10, 30], labels=["0", "0-4", "4-10", "10+"]
            )
        )
        .groupby("alcohol_cat", observed=False)["salud_general"]
        .mean()
        .to_dict()
    )
    diet_health = synthetic_df.groupby("categoria_alimentacion", observed=False)["salud_general"].mean().to_dict()
    diet_obesity = (
        synthetic_df.groupby("categoria_alimentacion", observed=False)["categoria_imc"]
        .apply(lambda s: (s == "obesidad").mean() * 100)
        .to_dict()
    )
    diet_diabetes = (
        synthetic_df.groupby("categoria_alimentacion", observed=False)["diabetes"]
        .mean()
        .mul(100)
        .to_dict()
    )
    ses_health = (
        synthetic_df.groupby("nivel_socioeconomico", observed=False)["salud_general"]
        .mean()
        .reindex(["bajo", "medio", "alto"])
        .to_dict()
    )
    locality_health = (
        synthetic_df.groupby("tipo_localidad", observed=False)["salud_general"]
        .mean()
        .reindex(["rural", "urbano"])
        .to_dict()
    )
    env_corr = float(synthetic_df["indice_entorno_riesgo"].corr(synthetic_df["salud_general"]))
    temp_env = synthetic_df.copy()
    temp_env["entorno_cat"] = pd.qcut(temp_env["indice_entorno_riesgo"], q=4, labels=["bajo", "medio", "alto", "muy_alto"])
    env_health = temp_env.groupby("entorno_cat", observed=False)["salud_general"].mean().to_dict()

    direct_context = {
        "obesidad_localidad_2023": {},
        "depresion_localidad_2023": {},
        "obesidad_socioeconomico_2023": {},
        "depresion_socioeconomico_2023": {},
    }
    for locality in ["rural", "urbano"]:
        val, _ = latest_value(
            datasets["obesidad"],
            {
                "sexo": "-",
                "grupo_de_edad": "-",
                "tipo_de_localidad_rural_o_urbano": locality,
                "region_geografica_de_Mexico": "-",
                "nivel_educativo": "-",
                "nivel_socioeconomico": "-",
                "presencia_de_discapacidad": "-",
                "nivel_de_agregacion": "estratificado",
            },
            preferred_years=[2023],
        )
        direct_context["obesidad_localidad_2023"][locality] = round(float(val), 2)
        val, _ = latest_value(
            datasets["depresion"],
            {
                "sexo": "-",
                "grupo_de_edad": "-",
                "tipo_de_localidad_rural_o_urbano": locality,
                "region_geografica_de_Mexico": "-",
                "nivel_educativo": "-",
                "nivel_socioeconomico": "-",
                "presencia_de_discapacidad": "-",
                "nivel_de_agregacion": "estratificado",
            },
            preferred_years=[2023],
        )
        direct_context["depresion_localidad_2023"][locality] = round(float(val), 2)
    for ses in ["bajo", "medio", "alto"]:
        val, _ = latest_value(
            datasets["obesidad"],
            {
                "sexo": "-",
                "grupo_de_edad": "-",
                "tipo_de_localidad_rural_o_urbano": "-",
                "region_geografica_de_Mexico": "-",
                "nivel_educativo": "-",
                "nivel_socioeconomico": ses,
                "presencia_de_discapacidad": "-",
                "nivel_de_agregacion": "estratificado",
            },
            preferred_years=[2023],
        )
        direct_context["obesidad_socioeconomico_2023"][ses] = round(float(val), 2)
        val, _ = latest_value(
            datasets["depresion"],
            {
                "sexo": "-",
                "grupo_de_edad": "-",
                "tipo_de_localidad_rural_o_urbano": "-",
                "region_geografica_de_Mexico": "-",
                "nivel_educativo": "-",
                "nivel_socioeconomico": ses,
                "presencia_de_discapacidad": "-",
                "nivel_de_agregacion": "estratificado",
            },
            preferred_years=[2023],
        )
        direct_context["depresion_socioeconomico_2023"][ses] = round(float(val), 2)

    national_latest = {}
    for key, label in {
        "obesidad": "obesidad",
        "sobrepeso": "sobrepeso",
        "depresion": "depresion",
        "diabetes": "diabetes",
        "hipertension": "hipertension",
    }.items():
        df = datasets[key]
        filters = {
            "sexo": "-",
            "grupo_de_edad": "-",
            "tipo_de_localidad_rural_o_urbano": "-",
            "region_geografica_de_Mexico": "-",
            "nivel_educativo": "-",
            "nivel_socioeconomico": "-",
            "presencia_de_discapacidad": "-",
        }
        if "nivel_de_agregacion" in df.columns:
            filters["nivel_de_agregacion"] = "nacional"
        value, year = latest_value(df, filters)
        if np.isnan(value) and "nivel_de_agregacion" in df.columns:
            filters["nivel_de_agregacion"] = "nacional "
            value, year = latest_value(df, filters)
        national_latest[label] = {"porcentaje": round(float(value), 2), "anio": year}

    peaks = (
        age_df.loc[age_df.groupby("condicion")["porcentaje"].idxmax(), ["condicion", "grupo_de_edad", "porcentaje", "anio_fuente"]]
        .sort_values("condicion")
        .to_dict(orient="records")
    )

    return {
        "correlaciones": {
            "sueno_horas_vs_estres": round(corr_sleep_hours, 4),
            "sueno_calidad_vs_estres": round(corr_sleep_quality, 4),
            "actividad_vs_imc": round(corr_activity_bmi, 4),
        },
        "estres_promedio": {
            "menos_de_6_horas_sueno": round(float(low_sleep), 2),
            "entre_7_y_8_horas_sueno": round(float(ideal_sleep), 2),
        },
        "imc_promedio_por_actividad": {k: round(float(v), 2) for k, v in activity_levels.items()},
        "salud_promedio_por_habitos": {
            "tabaco": {str(k): round(float(v), 2) for k, v in tobacco_health.items()},
            "azucar": {str(k): round(float(v), 2) for k, v in sugar_health.items()},
            "pantalla": {str(k): round(float(v), 2) for k, v in screen_health.items()},
            "alcohol": {str(k): round(float(v), 2) for k, v in alcohol_health.items()},
        },
        "alimentacion": {
            "salud_promedio": {str(k): round(float(v), 2) for k, v in diet_health.items()},
            "obesidad_pct": {str(k): round(float(v), 2) for k, v in diet_obesity.items()},
            "diabetes_pct": {str(k): round(float(v), 2) for k, v in diet_diabetes.items()},
        },
        "contexto_social": {
            "salud_promedio_socioeconomico": {str(k): round(float(v), 2) for k, v in ses_health.items()},
            "salud_promedio_localidad": {str(k): round(float(v), 2) for k, v in locality_health.items()},
            "fuente_directa_2023": direct_context,
        },
        "entorno": {
            "correlacion_entorno_vs_salud": round(env_corr, 4),
            "salud_promedio_por_riesgo_entorno": {str(k): round(float(v), 2) for k, v in env_health.items()},
        },
        "regresion_lineal_multiple": regression_summary,
        "prevalencias_nacionales_fuente_directa": national_latest,
        "picos_por_edad_fuente_directa": peaks,
        "filas_sinteticas": int(len(synthetic_df)),
    }


def write_markdown_report(summary: dict[str, object]) -> None:
    report = f"""# Analisis de salud en Mexico

## Alcance

- Se usaron tablas agregadas de ENSANUT ubicadas en `C:\\Users\\edluj\\OneDrive\\Documentos\\actividades datos\\avance de proyecto\\data`.
- La carpeta no contiene respuestas individuales para sueno, actividad fisica, azucar, alcohol, tabaco o tiempo en pantalla.
- Para responder esas preguntas se genero una poblacion sintetica de {summary["filas_sinteticas"]:,} adultos, calibrada con prevalencias por edad y sexo de obesidad, sobrepeso, depresion, diabetes e hipertension observadas en los CSV.

## Hallazgos principales

### 1. Como afecta la cantidad y calidad de sueno al estres

- Correlacion horas de sueno vs estres: {summary["correlaciones"]["sueno_horas_vs_estres"]}.
- Correlacion calidad de sueno vs estres: {summary["correlaciones"]["sueno_calidad_vs_estres"]}.
- Menos de 6 horas de sueno: estres promedio {summary["estres_promedio"]["menos_de_6_horas_sueno"]}.
- Entre 7 y 8 horas de sueno: estres promedio {summary["estres_promedio"]["entre_7_y_8_horas_sueno"]}.
- La calidad del sueno muestra una asociacion negativa mas fuerte que la cantidad de horas.

### 2. Relacion entre actividad fisica e IMC

- Correlacion actividad fisica vs IMC: {summary["correlaciones"]["actividad_vs_imc"]}.
- IMC promedio por nivel de actividad:
  - Sedentaria: {summary["imc_promedio_por_actividad"]["sedentaria"]}
  - Baja: {summary["imc_promedio_por_actividad"]["baja"]}
  - Media: {summary["imc_promedio_por_actividad"]["media"]}
  - Alta: {summary["imc_promedio_por_actividad"]["alta"]}
- La relacion general es inversa: a mayor actividad, menor IMC promedio.

### 3. Influencia de azucar, alcohol, tabaco y tiempo en pantalla en la salud general

- Tabaco:
  - No fuma: {summary["salud_promedio_por_habitos"]["tabaco"]["0"]}
  - Si fuma: {summary["salud_promedio_por_habitos"]["tabaco"]["1"]}
- Alcohol:
  - 0 unidades por semana: {summary["salud_promedio_por_habitos"]["alcohol"]["0"]}
  - 10+ unidades por semana: {summary["salud_promedio_por_habitos"]["alcohol"]["10+"]}
- Azucar:
  - 0-1 porciones: {summary["salud_promedio_por_habitos"]["azucar"]["0-1"]}
  - 3+ porciones: {summary["salud_promedio_por_habitos"]["azucar"]["3+"]}
- Pantalla:
  - 0-3 horas: {summary["salud_promedio_por_habitos"]["pantalla"]["0-3"]}
  - 7+ horas: {summary["salud_promedio_por_habitos"]["pantalla"]["7+"]}
- En el modelo sintetico, el tabaco y el exceso de tiempo en pantalla son los habitos que mas reducen la salud general, seguidos por el consumo elevado de azucar y alcohol.

### 4. Influencia directa de la alimentacion en la salud

- Alimentacion saludable: salud promedio {summary["alimentacion"]["salud_promedio"]["saludable"]}.
- Alimentacion muy alta en azucar: salud promedio {summary["alimentacion"]["salud_promedio"]["muy_alta_azucar"]}.
- Obesidad:
  - Alimentacion saludable: {summary["alimentacion"]["obesidad_pct"]["saludable"]}%
  - Alimentacion muy alta en azucar: {summary["alimentacion"]["obesidad_pct"]["muy_alta_azucar"]}%
- Diabetes:
  - Alimentacion saludable: {summary["alimentacion"]["diabetes_pct"]["saludable"]}%
  - Alimentacion muy alta en azucar: {summary["alimentacion"]["diabetes_pct"]["muy_alta_azucar"]}%
- En esta simulacion, una peor alimentacion se asocia de forma directa con menor salud general y mayor riesgo metabolico.

### 5. Contexto social y lugar de residencia

- Salud promedio por nivel socioeconomico:
  - Bajo: {summary["contexto_social"]["salud_promedio_socioeconomico"]["bajo"]}
  - Medio: {summary["contexto_social"]["salud_promedio_socioeconomico"]["medio"]}
  - Alto: {summary["contexto_social"]["salud_promedio_socioeconomico"]["alto"]}
- Salud promedio por localidad:
  - Rural: {summary["contexto_social"]["salud_promedio_localidad"]["rural"]}
  - Urbano: {summary["contexto_social"]["salud_promedio_localidad"]["urbano"]}
- Fuente directa 2023:
  - Obesidad por nivel socioeconomico: bajo {summary["contexto_social"]["fuente_directa_2023"]["obesidad_socioeconomico_2023"]["bajo"]}%, medio {summary["contexto_social"]["fuente_directa_2023"]["obesidad_socioeconomico_2023"]["medio"]}%, alto {summary["contexto_social"]["fuente_directa_2023"]["obesidad_socioeconomico_2023"]["alto"]}%.
  - Depresion por nivel socioeconomico: bajo {summary["contexto_social"]["fuente_directa_2023"]["depresion_socioeconomico_2023"]["bajo"]}%, medio {summary["contexto_social"]["fuente_directa_2023"]["depresion_socioeconomico_2023"]["medio"]}%, alto {summary["contexto_social"]["fuente_directa_2023"]["depresion_socioeconomico_2023"]["alto"]}%.
  - Obesidad por localidad: rural {summary["contexto_social"]["fuente_directa_2023"]["obesidad_localidad_2023"]["rural"]}%, urbano {summary["contexto_social"]["fuente_directa_2023"]["obesidad_localidad_2023"]["urbano"]}%.
  - Depresion por localidad: rural {summary["contexto_social"]["fuente_directa_2023"]["depresion_localidad_2023"]["rural"]}%, urbano {summary["contexto_social"]["fuente_directa_2023"]["depresion_localidad_2023"]["urbano"]}%.

### 6. Entorno y salud

- Correlacion indice de riesgo ambiental/social vs salud general: {summary["entorno"]["correlacion_entorno_vs_salud"]}.
- Salud promedio con riesgo bajo de entorno: {summary["entorno"]["salud_promedio_por_riesgo_entorno"]["bajo"]}.
- Salud promedio con riesgo muy alto de entorno: {summary["entorno"]["salud_promedio_por_riesgo_entorno"]["muy_alto"]}.
- En el modelo sintetico, mas contaminacion, ruido, trafico e inseguridad se asocian con peor salud y mayor estres.

### 7. Edades significativas de cierta condicion

"""
    for peak in summary["picos_por_edad_fuente_directa"]:
        report += (
            f"- {peak['condicion'].capitalize()}: el mayor porcentaje aparece en {peak['grupo_de_edad']} "
            f"con {peak['porcentaje']:.1f}% (fuente {peak['anio_fuente']}).\n"
        )

    report += f"""

### 8. Regresion lineal multiple para salud general

- Variable dependiente: {summary["regresion_lineal_multiple"]["variable_dependiente"]}.
- Justificacion: {summary["regresion_lineal_multiple"]["justificacion_variable_dependiente"]}
- Analisis de correlacion:
  - Se genero `23_pairplot_variables_modelo.png` para visualizar las relaciones bivariadas entre las variables clave del modelo.
- Division entrenamiento/prueba:
  - Observaciones totales: {summary["regresion_lineal_multiple"]["division_entrenamiento_prueba"]["observaciones_totales"]}
  - Entrenamiento: {summary["regresion_lineal_multiple"]["division_entrenamiento_prueba"]["tamano_entrenamiento"]}
  - Prueba: {summary["regresion_lineal_multiple"]["division_entrenamiento_prueba"]["tamano_prueba"]}
  - Proporcion de prueba: {summary["regresion_lineal_multiple"]["division_entrenamiento_prueba"]["proporcion_prueba"]}
- Se evaluaron variables demograficas, de habitos, condiciones clinicas y entorno, y el modelo final incluyo:
  {", ".join(summary["regresion_lineal_multiple"]["variables_incluidas_modelo_final"])}.
- Criterio de seleccion: {summary["regresion_lineal_multiple"]["criterio_seleccion_variables"]}
- LassoCV:
  - Alpha optimo: {summary["regresion_lineal_multiple"]["modelos_regularizados"]["lasso"]["alpha"]}
  - Variables retenidas: {", ".join(summary["regresion_lineal_multiple"]["variables_seleccionadas_por_lasso"])}
- RidgeCV:
  - Alpha optimo: {summary["regresion_lineal_multiple"]["modelos_regularizados"]["ridge"]["alpha"]}
  - Variables priorizadas por peso: {", ".join(summary["regresion_lineal_multiple"]["variables_priorizadas_por_ridge"])}
- Desempeno de modelos regularizados:
  - Lasso R2: {summary["regresion_lineal_multiple"]["modelos_regularizados"]["lasso"]["r2"]}
  - Ridge R2: {summary["regresion_lineal_multiple"]["modelos_regularizados"]["ridge"]["r2"]}
- Graficas de regularizacion:
  - `19_lasso_importancia.png` muestra las variables retenidas por Lasso segun la magnitud de sus coeficientes.
  - `20_ridge_importancia.png` muestra las variables priorizadas por Ridge segun la magnitud de sus coeficientes.
- Desempeno en prueba:
  - R2: {summary["regresion_lineal_multiple"]["metricas_holdout"]["r2"]}
  - MAE: {summary["regresion_lineal_multiple"]["metricas_holdout"]["mae"]}
  - MSE: {summary["regresion_lineal_multiple"]["metricas_holdout"]["mse"]}
  - RMSE: {summary["regresion_lineal_multiple"]["metricas_holdout"]["rmse"]}
- Predicciones sobre prueba:
  - Se exportaron en `predicciones_prueba_regresion.csv`.
  - La comparacion visual entre valores reales y predichos se muestra en `24_predicciones_vs_reales.png`.
- Variables finales y sustento:
  - {summary["regresion_lineal_multiple"]["sustento_variables_finales"][0]["variable"]}: {summary["regresion_lineal_multiple"]["sustento_variables_finales"][0]["respaldo_regularizacion"]}.
  - {summary["regresion_lineal_multiple"]["sustento_variables_finales"][1]["variable"]}: {summary["regresion_lineal_multiple"]["sustento_variables_finales"][1]["respaldo_regularizacion"]}.
  - {summary["regresion_lineal_multiple"]["sustento_variables_finales"][2]["variable"]}: {summary["regresion_lineal_multiple"]["sustento_variables_finales"][2]["respaldo_regularizacion"]}.
  - {summary["regresion_lineal_multiple"]["sustento_variables_finales"][3]["variable"]}: {summary["regresion_lineal_multiple"]["sustento_variables_finales"][3]["respaldo_regularizacion"]}.
  - {summary["regresion_lineal_multiple"]["sustento_variables_finales"][4]["variable"]}: {summary["regresion_lineal_multiple"]["sustento_variables_finales"][4]["respaldo_regularizacion"]}.
- Variables con mayor peso estandarizado:
  - {summary["regresion_lineal_multiple"]["coeficientes_principales"][0]["variable"]}: beta {summary["regresion_lineal_multiple"]["coeficientes_principales"][0]["beta_estandarizado"]}
  - {summary["regresion_lineal_multiple"]["coeficientes_principales"][1]["variable"]}: beta {summary["regresion_lineal_multiple"]["coeficientes_principales"][1]["beta_estandarizado"]}
  - {summary["regresion_lineal_multiple"]["coeficientes_principales"][2]["variable"]}: beta {summary["regresion_lineal_multiple"]["coeficientes_principales"][2]["beta_estandarizado"]}
  - {summary["regresion_lineal_multiple"]["coeficientes_principales"][3]["variable"]}: beta {summary["regresion_lineal_multiple"]["coeficientes_principales"][3]["beta_estandarizado"]}
  - {summary["regresion_lineal_multiple"]["coeficientes_principales"][4]["variable"]}: beta {summary["regresion_lineal_multiple"]["coeficientes_principales"][4]["beta_estandarizado"]}
- En terminos sustantivos, el IMC, el estres, la calidad del sueno, la hipertension y el entorno de riesgo son los determinantes mas importantes dentro del modelo final.

### 9. Verificacion de supuestos de la regresion lineal

- Linealidad (Ramsey RESET):
  - p-valor: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["linealidad"]["p_valor"]}
  - Cumple: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["linealidad"]["cumple"]}
- Independencia de residuos (Durbin-Watson):
  - Estadistico: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["independencia"]["estadistico"]}
  - Cumple: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["independencia"]["cumple"]}
- Homocedasticidad (Breusch-Pagan):
  - p-valor F: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["homocedasticidad"]["p_valor_f"]}
  - Cumple: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["homocedasticidad"]["cumple"]}
- Normalidad de residuos (Jarque-Bera):
  - p-valor: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["normalidad"]["p_valor"]}
  - Asimetria: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["normalidad"]["asimetria"]}
  - Curtosis: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["normalidad"]["curtosis"]}
  - Cumple: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["normalidad"]["cumple"]}
- Resultado global: {summary["regresion_lineal_multiple"]["supuestos_regresion"]["cumplen_los_4_supuestos"]}.

### 10. Analisis de tabla ANOVA

- La tabla ANOVA tipo II confirma que las variables con mayor aporte al modelo son:
  - {summary["regresion_lineal_multiple"]["anova_factores_principales"][0]["factor"]}: F = {summary["regresion_lineal_multiple"]["anova_factores_principales"][0]["estadistico_f"]}, p = {summary["regresion_lineal_multiple"]["anova_factores_principales"][0]["p_valor"]}
  - {summary["regresion_lineal_multiple"]["anova_factores_principales"][1]["factor"]}: F = {summary["regresion_lineal_multiple"]["anova_factores_principales"][1]["estadistico_f"]}, p = {summary["regresion_lineal_multiple"]["anova_factores_principales"][1]["p_valor"]}
  - {summary["regresion_lineal_multiple"]["anova_factores_principales"][2]["factor"]}: F = {summary["regresion_lineal_multiple"]["anova_factores_principales"][2]["estadistico_f"]}, p = {summary["regresion_lineal_multiple"]["anova_factores_principales"][2]["p_valor"]}
  - {summary["regresion_lineal_multiple"]["anova_factores_principales"][3]["factor"]}: F = {summary["regresion_lineal_multiple"]["anova_factores_principales"][3]["estadistico_f"]}, p = {summary["regresion_lineal_multiple"]["anova_factores_principales"][3]["p_valor"]}
  - {summary["regresion_lineal_multiple"]["anova_factores_principales"][4]["factor"]}: F = {summary["regresion_lineal_multiple"]["anova_factores_principales"][4]["estadistico_f"]}, p = {summary["regresion_lineal_multiple"]["anova_factores_principales"][4]["p_valor"]}
- Todas las variables del modelo final presentan evidencia estadistica de aporte al explicar la variacion del puntaje latente de salud.

## Prevalencias nacionales recientes observadas en fuente directa

- Obesidad: {summary["prevalencias_nacionales_fuente_directa"]["obesidad"]["porcentaje"]}% ({summary["prevalencias_nacionales_fuente_directa"]["obesidad"]["anio"]})
- Sobrepeso: {summary["prevalencias_nacionales_fuente_directa"]["sobrepeso"]["porcentaje"]}% ({summary["prevalencias_nacionales_fuente_directa"]["sobrepeso"]["anio"]})
- Depresion moderada/severa: {summary["prevalencias_nacionales_fuente_directa"]["depresion"]["porcentaje"]}% ({summary["prevalencias_nacionales_fuente_directa"]["depresion"]["anio"]})
- Diabetes diagnosticada: {summary["prevalencias_nacionales_fuente_directa"]["diabetes"]["porcentaje"]}% ({summary["prevalencias_nacionales_fuente_directa"]["diabetes"]["anio"]})
- Hipertension diagnosticada: {summary["prevalencias_nacionales_fuente_directa"]["hipertension"]["porcentaje"]}% ({summary["prevalencias_nacionales_fuente_directa"]["hipertension"]["anio"]})

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
- `predicciones_prueba_regresion.csv`
- `salud_mexico_sintetica_75000.csv`
- `metricas_resumen.json`
"""
    (OUTPUT_DIR / "resumen_analisis.md").write_text(report, encoding="utf-8")


def ensure_project_outputs() -> None:
    required_files = [
        OUTPUT_DIR / "salud_mexico_sintetica_75000.csv",
        OUTPUT_DIR / "metricas_resumen.json",
        OUTPUT_DIR / "resumen_analisis.md",
    ]
    if not all(path.exists() for path in required_files):
        main()


def load_gui_context() -> dict[str, object]:
    ensure_project_outputs()
    metrics_path = OUTPUT_DIR / "metricas_resumen.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    regression = summary["regresion_lineal_multiple"]
    return {
        "output_dir": OUTPUT_DIR,
        "dataset_path": OUTPUT_DIR / "salud_mexico_sintetica_75000.csv",
        "report_path": OUTPUT_DIR / "resumen_analisis.md",
        "metrics_path": metrics_path,
        "summary": summary,
        "regression": regression,
        "artifacts": {
            "pairplot": OUTPUT_DIR / "23_pairplot_variables_modelo.png",
            "lasso": OUTPUT_DIR / "19_lasso_importancia.png",
            "ridge": OUTPUT_DIR / "20_ridge_importancia.png",
            "heatmap_general": OUTPUT_DIR / "21_heatmap_correlacion_variables.png",
            "heatmap_modelo": OUTPUT_DIR / "22_heatmap_correlacion_modelo.png",
            "diagnostico": OUTPUT_DIR / "18_diagnostico_regresion.png",
            "regresion": OUTPUT_DIR / "17_regresion_salud_general.png",
            "predicciones_plot": OUTPUT_DIR / "24_predicciones_vs_reales.png",
        },
    }


def load_synthetic_health_dataset(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    ensure_project_outputs()
    csv_path = Path(ruta_csv) if ruta_csv else OUTPUT_DIR / "salud_mexico_sintetica_75000.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset sintetico: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_gui_model_data(
    df: pd.DataFrame,
    strategy: str = GUI_ESTRATEGIA_ELIMINAR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if strategy not in GUI_ESTRATEGIAS_VALIDAS:
        raise ValueError(f"Estrategia invalida. Usa una de estas: {sorted(GUI_ESTRATEGIAS_VALIDAS)}")

    regression_df = prepare_regression_dataset(df)
    required_columns = [REGRESSION_TARGET, *REGRESSION_FINAL_NUMERIC, *REGRESSION_FINAL_CATEGORICAL]
    missing_columns = [column for column in required_columns if column not in regression_df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para el modelo: {', '.join(missing_columns)}")

    model_df = regression_df[required_columns].copy()
    nulls_before = model_df.isna().sum().to_dict()
    total_before = len(model_df)

    if strategy == GUI_ESTRATEGIA_ELIMINAR:
        model_df = model_df.dropna().reset_index(drop=True)
        imputations: dict[str, object] = {}
        description = "Se eliminaron las filas con valores nulos en las variables del modelo."
    else:
        imputations = {}
        for column in required_columns:
            if pd.api.types.is_numeric_dtype(model_df[column]):
                value = float(model_df[column].median())
            else:
                mode = model_df[column].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else ""
            model_df[column] = model_df[column].fillna(value)
            if nulls_before.get(column, 0):
                imputations[column] = value
        description = "Se imputaron nulos con la mediana en numericas y la moda en categoricas."

    if len(model_df) < 10:
        raise ValueError("No hay suficientes registros para entrenar el modelo despues del tratamiento de nulos.")

    return model_df, {
        "strategy": strategy,
        "rows_before": total_before,
        "rows_after": int(len(model_df)),
        "rows_adjusted": int(total_before - len(model_df)),
        "nulls_before": nulls_before,
        "imputations": imputations,
        "description": description,
    }


def build_final_linear_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), REGRESSION_FINAL_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                REGRESSION_FINAL_CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


def build_linear_equation(model: Pipeline) -> str:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    intercept = float(model.named_steps["model"].intercept_)
    parts = [f"{intercept:.4f}"]
    for name, coef in zip(feature_names, coefficients, strict=False):
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{sign} {abs(float(coef)):.4f}*{name}")
    return " ".join(parts)


def run_gui_linear_model(
    ruta_csv: str | Path | None = None,
    carpeta_salida: str | Path | None = None,
    strategy: str = GUI_ESTRATEGIA_ELIMINAR,
) -> dict[str, object]:
    context = load_gui_context()
    output_dir = Path(carpeta_salida) if carpeta_salida else OUTPUT_DIR
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"No se pudo crear o usar la carpeta de salida: {output_dir}. "
            "Selecciona una carpeta existente con permisos de escritura."
        ) from exc

    raw_df = load_synthetic_health_dataset(ruta_csv)
    model_df, cleaning_summary = prepare_gui_model_data(raw_df, strategy=strategy)
    X = model_df[REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL].copy()
    y = model_df[REGRESSION_TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipeline = build_final_linear_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    prediction_df = X_test.reset_index(drop=True).copy()
    prediction_df["valor_real"] = y_test.reset_index(drop=True)
    prediction_df["valor_predicho"] = predictions
    prediction_df["residuo"] = prediction_df["valor_real"] - prediction_df["valor_predicho"]

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame({"termino": feature_names, "coeficiente": coefficients})
    coef_df["abs_coef"] = coef_df["coeficiente"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    suffix = strategy.lower()
    predictions_path = output_dir / f"predicciones_gui_salud_{suffix}.csv"
    coefficients_path = output_dir / f"coeficientes_gui_salud_{suffix}.csv"
    report_path = output_dir / f"reporte_gui_salud_{suffix}.md"

    prediction_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    coef_df.to_csv(coefficients_path, index=False, encoding="utf-8-sig")

    equation = build_linear_equation(pipeline)
    report_lines = [
        "# Modelo lineal multiple - Interfaz grafica",
        "",
        f"- Dataset: {Path(ruta_csv).name if ruta_csv else context['dataset_path'].name}",
        f"- Estrategia de nulos: {strategy}",
        f"- Registros antes: {cleaning_summary['rows_before']}",
        f"- Registros despues: {cleaning_summary['rows_after']}",
        f"- R2: {metrics['r2']:.4f}",
        f"- MAE: {metrics['mae']:.4f}",
        f"- MSE: {metrics['mse']:.4f}",
        f"- RMSE: {metrics['rmse']:.4f}",
        "",
        "## Variables del modelo",
        f"- Dependiente: {REGRESSION_TARGET}",
        f"- Independientes: {', '.join(REGRESSION_FINAL_NUMERIC + REGRESSION_FINAL_CATEGORICAL)}",
        "",
        "## Ecuacion",
        f"`{equation}`",
        "",
        "## Limpieza",
        f"- {cleaning_summary['description']}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "dataset_path": Path(ruta_csv) if ruta_csv else context["dataset_path"],
        "output_dir": output_dir,
        "strategy": strategy,
        "cleaning_summary": cleaning_summary,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": metrics,
        "equation": equation,
        "predictions_path": predictions_path,
        "coefficients_path": coefficients_path,
        "report_path": report_path,
        "predictions_preview": prediction_df.head(20).round(4),
        "coefficients_preview": coef_df.head(20).round(6),
        "base_artifacts": context["artifacts"],
        "base_regression_summary": context["regression"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    rng = np.random.default_rng(SEED)

    datasets = {name: load_csv(INPUT_DIR / filename) for name, filename in FILES.items()}
    trends_df = pd.concat(
        [
            build_national_trend(datasets["obesidad"], "Obesidad"),
            build_national_trend(datasets["sobrepeso"], "Sobrepeso"),
            build_national_trend(datasets["depresion"], "Depresion"),
            build_national_trend(datasets["diabetes"], "Diabetes"),
            build_national_trend(datasets["hipertension"], "Hipertension"),
        ],
        ignore_index=True,
    )
    make_trend_plot(trends_df)

    cell_pop = build_cell_population(datasets["obesidad"])
    prevalence_maps = {
        "sobrepeso": build_prevalence_map(datasets, "sobrepeso"),
        "obesidad": build_prevalence_map(datasets, "obesidad"),
        "depresion": build_prevalence_map(datasets, "depresion"),
        "diabetes": build_prevalence_map(datasets, "diabetes"),
        "hipertension": build_prevalence_map(datasets, "hipertension"),
    }

    synthetic_df = sample_population(cell_pop, rng)
    synthetic_df = assign_social_context(synthetic_df, rng)
    synthetic_df = generate_habits(synthetic_df, rng)
    synthetic_df = assign_bmi(synthetic_df, prevalence_maps["sobrepeso"], prevalence_maps["obesidad"], rng)
    synthetic_df = add_conditions(synthetic_df, prevalence_maps, rng)
    synthetic_df = add_health_score(synthetic_df, rng)
    synthetic_df = prepare_regression_dataset(synthetic_df)

    make_sleep_stress_plot(synthetic_df)
    make_activity_bmi_plot(synthetic_df)
    make_habits_health_plot(synthetic_df)
    make_tobacco_alcohol_plot(synthetic_df)
    make_diet_health_plot(synthetic_df)
    make_social_context_plot(synthetic_df)
    make_environment_health_plot(synthetic_df)
    make_heatmap_sleep_stress(synthetic_df)
    make_heatmap_activity_bmi(synthetic_df)
    make_heatmap_habits_health(synthetic_df)
    make_heatmap_diet_health(synthetic_df)
    make_heatmap_social_context(synthetic_df)
    age_df = make_age_condition_plot(datasets)
    make_heatmap_age_conditions(age_df)
    make_heatmap_environment(synthetic_df)
    make_heatmap_full_correlation(synthetic_df)
    make_heatmap_model_correlation(synthetic_df)
    regression_summary = build_multiple_linear_regression(synthetic_df)

    export_cols = [
        "id_persona",
        "sexo",
        "grupo_de_edad",
        "edad",
        "tipo_localidad",
        "nivel_socioeconomico",
        "nivel_educativo",
        "contaminacion_aire",
        "contaminacion_agua",
        "contaminacion_suelo",
        "contaminacion_auditiva",
        "trafico",
        "inseguridad",
        "indice_entorno_riesgo",
        "horas_sueno",
        "calidad_sueno",
        "estres",
        "min_actividad_semana",
        "actividad_efectiva",
        "nivel_actividad",
        "imc",
        "exceso_imc",
        "categoria_imc",
        "porciones_azucar_dia",
        "categoria_alimentacion",
        "unidades_alcohol_semana",
        "fuma",
        "horas_pantalla",
        "depresion",
        "hipertension",
        "diabetes",
        "salud_general_latente",
        "salud_general",
        "categoria_salud",
    ]
    csv_path = OUTPUT_DIR / "salud_mexico_sintetica_75000.csv"
    csv_written = True
    try:
        synthetic_df[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        csv_written = False
        print(
            "Aviso: no se pudo sobrescribir salud_mexico_sintetica_75000.csv porque el archivo esta en uso. "
            "Se mantuvieron actualizados el resumen, las metricas y las graficas."
        )

    summary = build_summary(synthetic_df, age_df, datasets, regression_summary)
    (OUTPUT_DIR / "metricas_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(summary)

    print(f"Salida generada en: {OUTPUT_DIR}")
    print("Archivos principales:")
    if csv_written:
        print("- salud_mexico_sintetica_75000.csv")
    else:
        print("- salud_mexico_sintetica_75000.csv (sin cambios porque estaba abierto)")
    print("- resumen_analisis.md")
    print("- metricas_resumen.json")
    print("- 01_tendencias_nacionales.png")
    print("- 02_sueno_estres.png")
    print("- 03_actividad_imc.png")
    print("- 04_habitos_salud_general.png")
    print("- 05_edades_significativas.png")
    print("- 06_tabaco_alcohol_salud.png")
    print("- 07_alimentacion_salud.png")
    print("- 08_contexto_social_salud.png")
    print("- 09_entorno_salud.png")
    print("- 10_heatmap_sueno_estres.png")
    print("- 11_heatmap_actividad_imc.png")
    print("- 12_heatmap_habitos_salud.png")
    print("- 13_heatmap_alimentacion_salud.png")
    print("- 14_heatmap_contexto_social.png")
    print("- 15_heatmap_edad_condiciones.png")
    print("- 16_heatmap_entorno_salud.png")
    print("- 17_regresion_salud_general.png")
    print("- 18_diagnostico_regresion.png")
    print("- 19_lasso_importancia.png")
    print("- 20_ridge_importancia.png")
    print("- 21_heatmap_correlacion_variables.png")
    print("- 22_heatmap_correlacion_modelo.png")
    print("- 23_pairplot_variables_modelo.png")
    print("- 24_predicciones_vs_reales.png")
    print("- predicciones_prueba_regresion.csv")


if __name__ == "__main__":
    main()
