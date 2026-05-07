from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "analisis_salud_mexico"
SOURCE_DATASET = SOURCE_DIR / "salud_mexico_sintetica_75000.csv"
OUTPUT_DIR = BASE_DIR / "analisis_salud_mexico_compuesto"
SEED = 42

TARGET = "salud_general_latente"
COMPOSITE_COLUMNS = [
    "indice_sueno_salud_mental",
    "indice_actividad_corporal",
    "indice_habitos_saludables",
    "indice_contexto_socioambiental",
    "indice_carga_clinica",
]
FORMULA = (
    "salud_general_latente ~ indice_sueno_salud_mental + indice_actividad_corporal + "
    "indice_habitos_saludables + indice_contexto_socioambiental + indice_carga_clinica"
)
PAIRPLOT_SAMPLE = 1200
GUI_ESTRATEGIA_ELIMINAR = "eliminar"
GUI_ESTRATEGIA_IMPUTAR = "imputar"
GUI_ESTRATEGIAS_VALIDAS = {GUI_ESTRATEGIA_ELIMINAR, GUI_ESTRATEGIA_IMPUTAR}


def load_base_dataset() -> pd.DataFrame:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset base del proyecto anterior: {SOURCE_DATASET}"
        )
    return pd.read_csv(SOURCE_DATASET)


def standardize(series: pd.Series) -> pd.Series:
    numeric = pd.Series(series, dtype=float)
    std = float(numeric.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - float(numeric.mean())) / std


def build_composite_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ses_score = df["nivel_socioeconomico"].map({"bajo": 1.0, "medio": 2.0, "alto": 3.0}).astype(float)
    composite = pd.DataFrame(
        {
            TARGET: df[TARGET].astype(float),
            "salud_general": df["salud_general"].astype(float),
            "indice_sueno_salud_mental": (
                standardize(df["horas_sueno"])
                + standardize(df["calidad_sueno"])
                - standardize(df["estres"])
                - standardize(df["depresion"])
            )
            / 4.0,
            "indice_actividad_corporal": (
                standardize(df["actividad_efectiva"])
                + standardize(df["min_actividad_semana"])
                - standardize(df["horas_pantalla"])
                - standardize(df["exceso_imc"])
            )
            / 4.0,
            "indice_habitos_saludables": (
                -standardize(df["porciones_azucar_dia"])
                - standardize(df["unidades_alcohol_semana"])
                - standardize(df["fuma"])
            )
            / 3.0,
            "indice_contexto_socioambiental": (
                standardize(ses_score)
                - standardize(df["contaminacion_aire"])
                - standardize(df["contaminacion_agua"])
                - standardize(df["contaminacion_suelo"])
                - standardize(df["contaminacion_auditiva"])
                - standardize(df["trafico"])
                - standardize(df["inseguridad"])
            )
            / 7.0,
            "indice_carga_clinica": (
                -standardize(df["diabetes"]) - standardize(df["hipertension"])
            )
            / 2.0,
        }
    )
    return composite


def make_pairplot(df: pd.DataFrame) -> Path:
    sample = df[[TARGET, *COMPOSITE_COLUMNS]].sample(min(len(df), PAIRPLOT_SAMPLE), random_state=SEED)
    plot = sns.pairplot(
        sample,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.25, "s": 18, "color": "#4c72b0"},
        diag_kws={"color": "#55a868"},
    )
    plot.figure.suptitle("Pairplot de variables compuestas", y=1.02)
    path = OUTPUT_DIR / "01_pairplot_indices_compuestos.png"
    plot.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(plot.figure)
    return path


def make_heatmap(df: pd.DataFrame) -> Path:
    corr = df[[TARGET, *COMPOSITE_COLUMNS]].corr()
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, square=True)
    plt.title("Correlacion entre indices compuestos")
    plt.tight_layout()
    path = OUTPUT_DIR / "02_heatmap_indices_compuestos.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def make_coefficients_plot(coef_series: pd.Series) -> Path:
    ordered = coef_series.sort_values().copy()
    labels = [label.replace("indice_", "").replace("_", " ").title() for label in ordered.index]
    colors = ["#c44e52" if value < 0 else "#55a868" for value in ordered.values]
    plt.figure(figsize=(9.5, 6))
    plt.barh(labels, ordered.values, color=colors)
    plt.axvline(0, color="black", linewidth=1.0, alpha=0.7)
    plt.title("Coeficientes del modelo con indices compuestos")
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    path = OUTPUT_DIR / "03_coeficientes_indices_compuestos.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def make_predictions_plot(y_true: pd.Series, y_pred: np.ndarray) -> Path:
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
    axes[1].set_xlabel("Observacion de prueba")
    axes[1].set_ylabel("Valor")
    axes[1].legend(loc="best")

    plt.tight_layout()
    path = OUTPUT_DIR / "04_predicciones_indices_compuestos.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def make_regression_diagnostics_plot(model: smf.ols) -> Path:
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
    path = OUTPUT_DIR / "05_diagnostico_regresion_indices_compuestos.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def fit_and_evaluate(df: pd.DataFrame) -> dict[str, object]:
    X = df[COMPOSITE_COLUMNS].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    prediction_df = X_test.reset_index(drop=True).copy()
    prediction_df["valor_real"] = y_test.reset_index(drop=True)
    prediction_df["valor_predicho"] = predictions
    prediction_df["residuo"] = prediction_df["valor_real"] - prediction_df["valor_predicho"]

    formula_model = smf.ols(FORMULA, data=df).fit()
    anova_table = (
        anova_lm(formula_model, typ=2)
        .reset_index()
        .rename(columns={"index": "factor", "sum_sq": "suma_cuadrados", "df": "gl", "F": "estadistico_f", "PR(>F)": "p_valor"})
    )

    reset_result = linear_reset(formula_model, power=2, use_f=True)
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(formula_model.resid, formula_model.model.exog)
    dw_stat = durbin_watson(formula_model.resid)
    jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(formula_model.resid)

    return {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": {
            "r2": round(float(r2_score(y_test, predictions)), 4),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "mse": round(float(mean_squared_error(y_test, predictions)), 4),
            "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        },
        "coefficients": pd.Series(model.coef_, index=COMPOSITE_COLUMNS),
        "intercept": round(float(model.intercept_), 4),
        "predictions_df": prediction_df.round(4),
        "formula_model": formula_model,
        "anova": anova_table,
        "assumptions": {
            "linealidad": {
                "prueba": "Ramsey RESET",
                "p_valor": round(float(reset_result.pvalue), 6),
                "cumple": bool(float(reset_result.pvalue) > 0.05),
            },
            "independencia": {
                "prueba": "Durbin-Watson",
                "estadistico": round(float(dw_stat), 4),
                "cumple": bool(1.5 <= float(dw_stat) <= 2.5),
            },
            "homocedasticidad": {
                "prueba": "Breusch-Pagan",
                "p_valor_f": round(float(bp_f_pvalue), 6),
                "cumple": bool(float(bp_f_pvalue) > 0.05),
            },
            "normalidad": {
                "prueba": "Jarque-Bera",
                "p_valor": round(float(jb_pvalue), 6),
                "asimetria": round(float(jb_skew), 4),
                "curtosis": round(float(jb_kurtosis), 4),
                "cumple": bool(float(jb_pvalue) > 0.05),
            },
        },
    }


def export_outputs(comp_df: pd.DataFrame, results: dict[str, object]) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    composite_csv = OUTPUT_DIR / "dataset_indices_compuestos.csv"
    metrics_json = OUTPUT_DIR / "metricas_indices_compuestos.json"
    predictions_csv = OUTPUT_DIR / "predicciones_indices_compuestos.csv"
    anova_csv = OUTPUT_DIR / "tabla_anova_indices_compuestos.csv"
    report_md = OUTPUT_DIR / "resumen_indices_compuestos.md"

    comp_df.to_csv(composite_csv, index=False, encoding="utf-8-sig")
    results["predictions_df"].to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    results["anova"].to_csv(anova_csv, index=False, encoding="utf-8-sig")

    pairplot_path = make_pairplot(comp_df)
    heatmap_path = make_heatmap(comp_df)
    coefficients_path = make_coefficients_plot(results["coefficients"])
    prediction_plot_path = make_predictions_plot(
        results["predictions_df"]["valor_real"], results["predictions_df"]["valor_predicho"].to_numpy()
    )
    diagnostics_path = make_regression_diagnostics_plot(results["formula_model"])

    assumptions = results["assumptions"]
    assumptions["cumplen_los_4_supuestos"] = bool(
        assumptions["linealidad"]["cumple"]
        and assumptions["independencia"]["cumple"]
        and assumptions["homocedasticidad"]["cumple"]
        and assumptions["normalidad"]["cumple"]
    )

    anova_rows = []
    for _, row in results["anova"].iterrows():
        anova_rows.append(
            {
                "factor": str(row["factor"]),
                "suma_cuadrados": None if pd.isna(row["suma_cuadrados"]) else round(float(row["suma_cuadrados"]), 4),
                "gl": None if pd.isna(row["gl"]) else round(float(row["gl"]), 4),
                "estadistico_f": None if pd.isna(row["estadistico_f"]) else round(float(row["estadistico_f"]), 4),
                "p_valor": None if pd.isna(row["p_valor"]) else round(float(row["p_valor"]), 6),
            }
        )
    top_anova = sorted(
        [row for row in anova_rows if row["factor"] != "Residual" and row["estadistico_f"] is not None],
        key=lambda row: row["estadistico_f"],
        reverse=True,
    )[:5]

    metrics_payload = {
        "variable_dependiente": TARGET,
        "variables_compuestas": {
            "indice_sueno_salud_mental": "Promedio estandarizado de horas de sueno, calidad de sueno, estres y depresion.",
            "indice_actividad_corporal": "Promedio estandarizado de actividad efectiva, minutos de actividad, pantalla y exceso de IMC.",
            "indice_habitos_saludables": "Promedio estandarizado inverso de azucar, alcohol y tabaquismo.",
            "indice_contexto_socioambiental": "Promedio estandarizado de nivel socioeconomico y contexto ambiental.",
            "indice_carga_clinica": "Promedio estandarizado inverso de diabetes e hipertension.",
        },
        "metodo_construccion_indices": (
            "Cada indice se construyo estandarizando primero las variables de su categoria y luego promediandolas "
            "con el signo segun su aporte esperado a la salud. Esto evita reutilizar directamente los pesos de la variable objetivo."
        ),
        "train_test_split": {
            "train_size": results["train_size"],
            "test_size": results["test_size"],
            "test_proportion": 0.2,
        },
        "metrics_test": results["metrics"],
        "intercept": results["intercept"],
        "coefficients": {
            key: round(float(value), 6) for key, value in results["coefficients"].items()
        },
        "assumptions": assumptions,
        "anova_top": top_anova,
        "formula": FORMULA,
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Proyecto alternativo: regresion con indices compuestos",
        "",
        "## Justificacion",
        "- Se creo un proyecto nuevo, separado del original, para reducir la cantidad de variables explicativas.",
        "- En lugar de modelar muchas variables individuales, se agruparon por categoria conceptual.",
        "- Se mantuvo un quinto indice de carga clinica para no perder la señal fuerte de diabetes e hipertension.",
        "- En esta version, los indices no usan pesos heredados del puntaje de salud: se forman con variables estandarizadas dentro de cada categoria.",
        "",
        "## Variables compuestas",
        "- indice_sueno_salud_mental: horas de sueno, calidad de sueno, estres y depresion estandarizados.",
        "- indice_actividad_corporal: actividad efectiva, minutos de actividad, horas de pantalla y exceso de IMC estandarizados.",
        "- indice_habitos_saludables: azucar, alcohol y tabaquismo estandarizados.",
        "- indice_contexto_socioambiental: nivel socioeconomico y condiciones ambientales estandarizados.",
        "- indice_carga_clinica: diabetes e hipertension estandarizadas.",
        "",
        "## Metodo de construccion",
        "- Cada variable se transformo a escala z dentro de su categoria.",
        "- Luego se promediaron con signo positivo para factores favorables y signo negativo para factores desfavorables.",
        "- Esto reduce la circularidad respecto al proyecto compuesto anterior.",
        "",
        "## Entrenamiento y prueba",
        f"- Entrenamiento: {results['train_size']}",
        f"- Prueba: {results['test_size']}",
        "",
        "## Metricas en prueba",
        f"- R2: {results['metrics']['r2']}",
        f"- MAE: {results['metrics']['mae']}",
        f"- MSE: {results['metrics']['mse']}",
        f"- RMSE: {results['metrics']['rmse']}",
        "",
        "## Coeficientes del modelo",
    ]
    report_lines.extend(
        f"- {name}: {value:.4f}" for name, value in results["coefficients"].sort_values(ascending=False).items()
    )
    report_lines.extend(
        [
            "",
            "## Supuestos del modelo",
            f"- Linealidad: {assumptions['linealidad']['cumple']} (p={assumptions['linealidad']['p_valor']})",
            f"- Independencia: {assumptions['independencia']['cumple']} (DW={assumptions['independencia']['estadistico']})",
            f"- Homocedasticidad: {assumptions['homocedasticidad']['cumple']} (p={assumptions['homocedasticidad']['p_valor_f']})",
            f"- Normalidad: {assumptions['normalidad']['cumple']} (p={assumptions['normalidad']['p_valor']})",
            f"- Cumplen los 4 supuestos: {assumptions['cumplen_los_4_supuestos']}",
            "",
            "## ANOVA",
        ]
    )
    report_lines.extend(
        f"- {row['factor']}: F={row['estadistico_f']}, p={row['p_valor']}" for row in top_anova
    )
    report_lines.extend(
        [
            "",
            "## Archivos generados",
            f"- {pairplot_path.name}",
            f"- {heatmap_path.name}",
            f"- {coefficients_path.name}",
            f"- {prediction_plot_path.name}",
            f"- {diagnostics_path.name}",
            f"- {predictions_csv.name}",
            f"- {anova_csv.name}",
            f"- {composite_csv.name}",
            f"- {metrics_json.name}",
        ]
    )
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "pairplot": pairplot_path,
        "heatmap": heatmap_path,
        "coefficients_plot": coefficients_path,
        "prediction_plot": prediction_plot_path,
        "diagnostics_plot": diagnostics_path,
        "predictions_csv": predictions_csv,
        "anova_csv": anova_csv,
        "composite_csv": composite_csv,
        "metrics_json": metrics_json,
        "report_md": report_md,
    }


def ensure_project_outputs() -> None:
    required = [
        OUTPUT_DIR / "dataset_indices_compuestos.csv",
        OUTPUT_DIR / "metricas_indices_compuestos.json",
        OUTPUT_DIR / "resumen_indices_compuestos.md",
    ]
    if not all(path.exists() for path in required):
        main()


def load_gui_context() -> dict[str, object]:
    ensure_project_outputs()
    metrics_path = OUTPUT_DIR / "metricas_indices_compuestos.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "output_dir": OUTPUT_DIR,
        "dataset_path": OUTPUT_DIR / "dataset_indices_compuestos.csv",
        "report_path": OUTPUT_DIR / "resumen_indices_compuestos.md",
        "metrics_path": metrics_path,
        "summary": summary,
        "artifacts": {
            "pairplot": OUTPUT_DIR / "01_pairplot_indices_compuestos.png",
            "heatmap": OUTPUT_DIR / "02_heatmap_indices_compuestos.png",
            "coeficientes": OUTPUT_DIR / "03_coeficientes_indices_compuestos.png",
            "predicciones": OUTPUT_DIR / "04_predicciones_indices_compuestos.png",
            "diagnostico": OUTPUT_DIR / "05_diagnostico_regresion_indices_compuestos.png",
            "anova": OUTPUT_DIR / "tabla_anova_indices_compuestos.csv",
        },
    }


def load_composite_or_source_dataset(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    ensure_project_outputs()
    csv_path = Path(ruta_csv) if ruta_csv else OUTPUT_DIR / "dataset_indices_compuestos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el CSV indicado: {csv_path}")
    df = pd.read_csv(csv_path)
    if TARGET in df.columns and all(column in df.columns for column in COMPOSITE_COLUMNS):
        return df
    return build_composite_dataset(df)


def prepare_gui_model_data(
    df: pd.DataFrame,
    strategy: str = GUI_ESTRATEGIA_ELIMINAR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if strategy not in GUI_ESTRATEGIAS_VALIDAS:
        raise ValueError(f"Estrategia invalida. Usa una de estas: {sorted(GUI_ESTRATEGIAS_VALIDAS)}")

    model_df = df[[TARGET, *COMPOSITE_COLUMNS]].copy()
    nulls_before = model_df.isna().sum().to_dict()
    total_before = len(model_df)

    if strategy == GUI_ESTRATEGIA_ELIMINAR:
        model_df = model_df.dropna().reset_index(drop=True)
        imputations: dict[str, object] = {}
        description = "Se eliminaron las filas con nulos en los indices compuestos."
    else:
        imputations = {}
        for column in [TARGET, *COMPOSITE_COLUMNS]:
            value = float(model_df[column].median())
            model_df[column] = model_df[column].fillna(value)
            if nulls_before.get(column, 0):
                imputations[column] = value
        description = "Se imputaron nulos con la mediana de cada indice compuesto."

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


def build_linear_equation(intercept: float, coefficients: pd.Series) -> str:
    parts = [f"{intercept:.4f}"]
    for name, coef in coefficients.items():
        sign = "+" if float(coef) >= 0 else "-"
        parts.append(f"{sign} {abs(float(coef)):.4f}*{name}")
    return " ".join(parts)


def safe_write_csv(df: pd.DataFrame, target_path: Path) -> Path:
    try:
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        return target_path
    except PermissionError:
        fallback_path = Path(__file__).resolve().parent / target_path.name
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        return fallback_path


def safe_write_text(text: str, target_path: Path) -> Path:
    try:
        target_path.write_text(text, encoding="utf-8")
        return target_path
    except PermissionError:
        fallback_path = Path(__file__).resolve().parent / target_path.name
        fallback_path.write_text(text, encoding="utf-8")
        return fallback_path


def run_gui_composite_model(
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

    composite_df = load_composite_or_source_dataset(ruta_csv)
    model_df, cleaning = prepare_gui_model_data(composite_df, strategy=strategy)

    X = model_df[COMPOSITE_COLUMNS].copy()
    y = model_df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    prediction_df = X_test.reset_index(drop=True).copy()
    prediction_df["valor_real"] = y_test.reset_index(drop=True)
    prediction_df["valor_predicho"] = predictions
    prediction_df["residuo"] = prediction_df["valor_real"] - prediction_df["valor_predicho"]

    coefficients = pd.Series(model.coef_, index=COMPOSITE_COLUMNS)
    coef_df = pd.DataFrame({"indice": coefficients.index, "coeficiente": coefficients.values})
    coef_df["abs_coef"] = coef_df["coeficiente"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }

    suffix = strategy.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = output_dir / f"predicciones_gui_compuesto_{suffix}_{timestamp}.csv"
    coefficients_path = output_dir / f"coeficientes_gui_compuesto_{suffix}_{timestamp}.csv"
    report_path = output_dir / f"reporte_gui_compuesto_{suffix}_{timestamp}.md"

    predictions_path = safe_write_csv(prediction_df, predictions_path)
    coefficients_path = safe_write_csv(coef_df, coefficients_path)

    equation = build_linear_equation(float(model.intercept_), coefficients)
    report_lines = [
        "# Modelo compuesto - Interfaz grafica",
        "",
        f"- Dataset: {Path(ruta_csv).name if ruta_csv else context['dataset_path'].name}",
        f"- Estrategia de nulos: {strategy}",
        f"- Registros antes: {cleaning['rows_before']}",
        f"- Registros despues: {cleaning['rows_after']}",
        f"- R2: {metrics['r2']:.4f}",
        f"- MAE: {metrics['mae']:.4f}",
        f"- MSE: {metrics['mse']:.4f}",
        f"- RMSE: {metrics['rmse']:.4f}",
        "",
        "## Variables del modelo",
        f"- Dependiente: {TARGET}",
        f"- Independientes: {', '.join(COMPOSITE_COLUMNS)}",
        "",
        "## Ecuacion",
        f"`{equation}`",
        "",
        "## Limpieza",
        f"- {cleaning['description']}",
    ]
    report_path = safe_write_text("\n".join(report_lines), report_path)

    return {
        "dataset_path": Path(ruta_csv) if ruta_csv else context["dataset_path"],
        "output_dir": output_dir,
        "strategy": strategy,
        "cleaning_summary": cleaning,
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
        "base_summary": context["summary"],
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    base_df = load_base_dataset()
    composite_df = build_composite_dataset(base_df)
    results = fit_and_evaluate(composite_df)
    outputs = export_outputs(composite_df, results)

    print(f"Proyecto compuesto generado en: {OUTPUT_DIR}")
    print("Archivos principales:")
    for key in [
        "pairplot",
        "heatmap",
        "coefficients_plot",
        "prediction_plot",
        "diagnostics_plot",
        "predictions_csv",
        "anova_csv",
        "composite_csv",
        "metrics_json",
        "report_md",
    ]:
        print(f"- {outputs[key].name}")


if __name__ == "__main__":
    main()
