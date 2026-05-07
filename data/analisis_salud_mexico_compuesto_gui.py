from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from analisis_salud_mexico_compuesto import (
    COMPOSITE_COLUMNS,
    GUI_ESTRATEGIA_ELIMINAR,
    GUI_ESTRATEGIA_IMPUTAR,
    OUTPUT_DIR,
    TARGET,
    load_gui_context,
    run_gui_composite_model,
)


class AplicacionAnalisisCompuesto(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Proyecto Salud Mexico - Interfaz de Indices Compuestos")
        self.geometry("1260x860")
        self.minsize(1080, 720)

        self.ruta_csv = tk.StringVar()
        self.ruta_salida = tk.StringVar(value=str(OUTPUT_DIR))
        self.estrategia_nulos = tk.StringVar(value=GUI_ESTRATEGIA_ELIMINAR)
        self.estado = tk.StringVar(value="Cargando contexto del proyecto compuesto...")
        self.contexto_base: dict[str, object] | None = None
        self.resultado_actual: dict[str, object] | None = None

        self._crear_interfaz()
        self._cargar_contexto_inicial()

    def _crear_interfaz(self) -> None:
        contenedor = ttk.Frame(self, padding=16)
        contenedor.pack(fill="both", expand=True)

        ttk.Label(
            contenedor,
            text="Proyecto de salud en Mexico - Regresion con indices compuestos",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            contenedor,
            text=(
                "La interfaz organiza el flujo en tres pasos: comprobacion de variables compuestas, "
                "construccion del modelo y resultados finales."
            ),
            wraplength=1180,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        notebook = ttk.Notebook(contenedor)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self.tab_variables = ttk.Frame(notebook, padding=14)
        self.tab_modelo = ttk.Frame(notebook, padding=14)
        self.tab_resultados = ttk.Frame(notebook, padding=14)
        notebook.add(self.tab_variables, text="1. Comprobacion de variables")
        notebook.add(self.tab_modelo, text="2. Construccion del modelo")
        notebook.add(self.tab_resultados, text="3. Resultados finales")

        self._crear_tab_variables()
        self._crear_tab_modelo()
        self._crear_tab_resultados()

    def _crear_tab_variables(self) -> None:
        acciones = ttk.Frame(self.tab_variables)
        acciones.pack(fill="x", pady=(0, 10))

        self.botones_artifacts: dict[str, ttk.Button] = {}
        botones = [
            ("reporte", "Abrir reporte base", self.abrir_reporte_base),
            ("pairplot", "Abrir pairplot", self.abrir_pairplot_base),
            ("heatmap", "Abrir heatmap", self.abrir_heatmap_base),
            ("coeficientes", "Abrir coeficientes", self.abrir_coeficientes_base),
            ("predicciones", "Abrir predicciones vs reales", self.abrir_predicciones_base),
            ("diagnostico", "Abrir diagnostico", self.abrir_diagnostico_base),
            ("anova", "Abrir tabla ANOVA", self.abrir_anova_base),
        ]
        for key, label, command in botones:
            button = ttk.Button(acciones, text=label, command=command, state="disabled")
            button.pack(side="left", padx=(0, 8))
            self.botones_artifacts[key] = button

        self.texto_variables = tk.Text(self.tab_variables, wrap="word", height=24)
        self.texto_variables.pack(fill="both", expand=True)
        self.texto_variables.configure(state="disabled")

    def _crear_tab_modelo(self) -> None:
        marco_rutas = ttk.LabelFrame(self.tab_modelo, text="Archivos y salida", padding=12)
        marco_rutas.pack(fill="x", pady=(0, 12))

        ttk.Label(marco_rutas, text="Dataset para modelar").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(marco_rutas, textvariable=self.ruta_csv, width=96).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(marco_rutas, text="Examinar", command=self.seleccionar_csv).grid(row=0, column=2, padx=(8, 0), pady=6)

        ttk.Label(marco_rutas, text="Carpeta de salida").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(marco_rutas, textvariable=self.ruta_salida, width=96).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(marco_rutas, text="Elegir carpeta", command=self.seleccionar_carpeta).grid(row=1, column=2, padx=(8, 0), pady=6)
        marco_rutas.columnconfigure(1, weight=1)

        marco_variables = ttk.LabelFrame(self.tab_modelo, text="Especificacion del modelo", padding=12)
        marco_variables.pack(fill="x", pady=(0, 12))
        ttk.Label(marco_variables, text=f"Variable dependiente: {TARGET}").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(marco_variables, text="Indices compuestos del modelo:").grid(row=1, column=0, sticky="w", pady=(8, 4))
        ttk.Label(
            marco_variables,
            text=", ".join(COMPOSITE_COLUMNS),
            wraplength=1100,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Label(marco_variables, text="Tratamiento de nulos").grid(row=3, column=0, sticky="w", pady=(8, 4))
        ttk.Combobox(
            marco_variables,
            textvariable=self.estrategia_nulos,
            state="readonly",
            values=[GUI_ESTRATEGIA_ELIMINAR, GUI_ESTRATEGIA_IMPUTAR],
            width=22,
        ).grid(row=4, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            marco_variables,
            text=(
                "La opcion elegida controla si se eliminan filas con nulos en los indices compuestos "
                "o si se imputan con mediana antes del entrenamiento."
            ),
            wraplength=1100,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

        marco_acciones = ttk.Frame(self.tab_modelo)
        marco_acciones.pack(fill="x", pady=(6, 10))
        ttk.Button(marco_acciones, text="Entrenar modelo compuesto", command=self.ejecutar_modelo).pack(side="left")

        ttk.Label(
            self.tab_modelo,
            textvariable=self.estado,
            foreground="#1f4e79",
            wraplength=1140,
            justify="left",
        ).pack(fill="x", pady=(4, 0))

    def _crear_tab_resultados(self) -> None:
        acciones = ttk.Frame(self.tab_resultados)
        acciones.pack(fill="x", pady=(0, 10))

        self.boton_reporte_modelo = ttk.Button(acciones, text="Abrir reporte del modelo", command=self.abrir_reporte_modelo, state="disabled")
        self.boton_reporte_modelo.pack(side="left", padx=(0, 8))
        self.boton_predicciones_modelo = ttk.Button(acciones, text="Abrir predicciones", command=self.abrir_predicciones_modelo, state="disabled")
        self.boton_predicciones_modelo.pack(side="left", padx=(0, 8))
        self.boton_coeficientes_modelo = ttk.Button(acciones, text="Abrir coeficientes", command=self.abrir_coeficientes_modelo, state="disabled")
        self.boton_coeficientes_modelo.pack(side="left", padx=(0, 8))
        self.boton_carpeta_modelo = ttk.Button(acciones, text="Abrir carpeta", command=self.abrir_carpeta_modelo, state="disabled")
        self.boton_carpeta_modelo.pack(side="left")

        self.texto_resultados = tk.Text(self.tab_resultados, wrap="word", height=16)
        self.texto_resultados.pack(fill="x", pady=(0, 12))
        self.texto_resultados.configure(state="disabled")

        contenedor_tablas = ttk.Frame(self.tab_resultados)
        contenedor_tablas.pack(fill="both", expand=True)
        contenedor_tablas.columnconfigure(0, weight=1)
        contenedor_tablas.columnconfigure(1, weight=1)
        contenedor_tablas.rowconfigure(0, weight=1)

        marco_pred = ttk.LabelFrame(contenedor_tablas, text="Vista previa de predicciones", padding=8)
        marco_pred.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.tree_pred = ttk.Treeview(marco_pred, show="headings")
        self.tree_pred.pack(side="left", fill="both", expand=True)
        pred_scroll = ttk.Scrollbar(marco_pred, orient="vertical", command=self.tree_pred.yview)
        pred_scroll.pack(side="right", fill="y")
        self.tree_pred.configure(yscrollcommand=pred_scroll.set)

        marco_coef = ttk.LabelFrame(contenedor_tablas, text="Coeficientes del modelo", padding=8)
        marco_coef.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.tree_coef = ttk.Treeview(marco_coef, show="headings")
        self.tree_coef.pack(side="left", fill="both", expand=True)
        coef_scroll = ttk.Scrollbar(marco_coef, orient="vertical", command=self.tree_coef.yview)
        coef_scroll.pack(side="right", fill="y")
        self.tree_coef.configure(yscrollcommand=coef_scroll.set)

    def _cargar_contexto_inicial(self) -> None:
        try:
            self.contexto_base = load_gui_context()
        except Exception as exc:
            messagebox.showerror("Error al cargar el proyecto", str(exc))
            self.estado.set("No se pudo cargar el contexto inicial del proyecto compuesto.")
            return

        dataset_path = Path(str(self.contexto_base["dataset_path"]))
        self.ruta_csv.set(str(dataset_path))
        self._llenar_resumen_variables()
        for button in self.botones_artifacts.values():
            button.configure(state="normal")
        self.estado.set("Contexto compuesto cargado. Puedes revisar variables o entrenar el modelo desde la pestaña 2.")

    def _llenar_resumen_variables(self) -> None:
        if not self.contexto_base:
            return
        summary = self.contexto_base["summary"]
        assumptions = summary["assumptions"]
        anova = summary["anova_top"]
        lines = [
            "Comprobacion de variables compuestas y evidencias del modelo base",
            "",
            f"Variable dependiente: {summary['variable_dependiente']}",
            "Indices del modelo:",
        ]
        lines.extend(f"- {key}: {value}" for key, value in summary["variables_compuestas"].items())
        lines.extend(
            [
                "",
                "Supuestos del modelo base:",
                f"- Linealidad: {assumptions['linealidad']['cumple']} (p={assumptions['linealidad']['p_valor']})",
                f"- Independencia: {assumptions['independencia']['cumple']} (DW={assumptions['independencia']['estadistico']})",
                f"- Homocedasticidad: {assumptions['homocedasticidad']['cumple']} (p={assumptions['homocedasticidad']['p_valor_f']})",
                f"- Normalidad: {assumptions['normalidad']['cumple']} (p={assumptions['normalidad']['p_valor']})",
                "",
                "Factores principales en ANOVA:",
            ]
        )
        lines.extend(f"- {row['factor']}: F={row['estadistico_f']}, p={row['p_valor']}" for row in anova)
        lines.extend(
            [
                "",
                "Artefactos disponibles desde esta pestaña:",
                "- Pairplot de indices compuestos",
                "- Heatmap de correlacion",
                "- Grafica de coeficientes",
                "- Predicciones vs reales",
                "- Diagnostico de regresion y tabla ANOVA",
            ]
        )

        self.texto_variables.configure(state="normal")
        self.texto_variables.delete("1.0", tk.END)
        self.texto_variables.insert("1.0", "\n".join(lines))
        self.texto_variables.configure(state="disabled")

    def seleccionar_csv(self) -> None:
        ruta = filedialog.askopenfilename(
            title="Selecciona un CSV",
            initialdir=str(Path(self.ruta_csv.get()).parent if self.ruta_csv.get() else OUTPUT_DIR),
            filetypes=[("Archivos CSV", "*.csv")],
        )
        if ruta:
            self.ruta_csv.set(ruta)

    def seleccionar_carpeta(self) -> None:
        carpeta = filedialog.askdirectory(
            title="Selecciona la carpeta de salida",
            initialdir=self.ruta_salida.get() or str(OUTPUT_DIR),
        )
        if carpeta:
            self.ruta_salida.set(carpeta)

    def ejecutar_modelo(self) -> None:
        ruta_csv = self.ruta_csv.get().strip()
        ruta_salida = self.ruta_salida.get().strip()
        estrategia = self.estrategia_nulos.get().strip()

        if not ruta_csv:
            messagebox.showwarning("CSV faltante", "Selecciona un dataset antes de entrenar el modelo.")
            return
        if not ruta_salida:
            messagebox.showwarning("Salida faltante", "Selecciona una carpeta para guardar los resultados.")
            return

        self.estado.set("Entrenando modelo compuesto y generando resultados...")
        self.update_idletasks()
        try:
            resultado = run_gui_composite_model(ruta_csv=ruta_csv, carpeta_salida=ruta_salida, strategy=estrategia)
        except Exception as exc:
            messagebox.showerror("Error al entrenar el modelo", str(exc))
            self.estado.set("La ejecucion fallo. Revisa el dataset y la configuracion.")
            return

        self.resultado_actual = resultado
        self._mostrar_resultados_modelo(resultado)
        self.estado.set("Modelo compuesto completado. Ya puedes revisar metricas y predicciones en la pestaña 3.")
        self.notebook.select(self.tab_resultados)
        messagebox.showinfo("Modelo finalizado", "El modelo con indices compuestos se entreno correctamente.")

    def _mostrar_resultados_modelo(self, resultado: dict[str, object]) -> None:
        metrics = resultado["metrics"]
        cleaning = resultado["cleaning_summary"]
        lines = [
            f"Dataset usado: {Path(str(resultado['dataset_path'])).name}",
            f"Estrategia de nulos: {resultado['strategy']}",
            f"Registros antes: {cleaning['rows_before']}",
            f"Registros despues: {cleaning['rows_after']}",
            f"Entrenamiento: {resultado['train_size']}",
            f"Prueba: {resultado['test_size']}",
            "",
            "Metricas en prueba:",
            f"- R2: {metrics['r2']:.4f}",
            f"- MAE: {metrics['mae']:.4f}",
            f"- MSE: {metrics['mse']:.4f}",
            f"- RMSE: {metrics['rmse']:.4f}",
            "",
            "Ecuacion del modelo:",
            resultado["equation"],
            "",
            "Limpieza aplicada:",
            cleaning["description"],
            "",
            f"Reporte: {resultado['report_path']}",
            f"Predicciones: {resultado['predictions_path']}",
            f"Coeficientes: {resultado['coefficients_path']}",
        ]

        self.texto_resultados.configure(state="normal")
        self.texto_resultados.delete("1.0", tk.END)
        self.texto_resultados.insert("1.0", "\n".join(lines))
        self.texto_resultados.configure(state="disabled")

        self._cargar_tabla(self.tree_pred, resultado["predictions_preview"])
        self._cargar_tabla(self.tree_coef, resultado["coefficients_preview"])

        self.boton_reporte_modelo.configure(state="normal")
        self.boton_predicciones_modelo.configure(state="normal")
        self.boton_coeficientes_modelo.configure(state="normal")
        self.boton_carpeta_modelo.configure(state="normal")

    def _cargar_tabla(self, tree: ttk.Treeview, datos: pd.DataFrame) -> None:
        tree.delete(*tree.get_children())
        tree["columns"] = list(datos.columns)
        for columna in datos.columns:
            tree.heading(columna, text=columna)
            tree.column(columna, width=140, anchor="center")
        for _, fila in datos.iterrows():
            valores = [fila[columna] for columna in datos.columns]
            tree.insert("", "end", values=valores)

    def abrir_reporte_base(self) -> None:
        if self.contexto_base:
            self._abrir_ruta(Path(str(self.contexto_base["report_path"])))

    def abrir_pairplot_base(self) -> None:
        self._abrir_artifact("pairplot")

    def abrir_heatmap_base(self) -> None:
        self._abrir_artifact("heatmap")

    def abrir_coeficientes_base(self) -> None:
        self._abrir_artifact("coeficientes")

    def abrir_predicciones_base(self) -> None:
        self._abrir_artifact("predicciones")

    def abrir_diagnostico_base(self) -> None:
        self._abrir_artifact("diagnostico")

    def abrir_anova_base(self) -> None:
        self._abrir_artifact("anova")

    def _abrir_artifact(self, key: str) -> None:
        if not self.contexto_base:
            return
        self._abrir_ruta(Path(str(self.contexto_base["artifacts"][key])))

    def abrir_reporte_modelo(self) -> None:
        if self.resultado_actual:
            self._abrir_ruta(Path(str(self.resultado_actual["report_path"])))

    def abrir_predicciones_modelo(self) -> None:
        if self.resultado_actual:
            self._abrir_ruta(Path(str(self.resultado_actual["predictions_path"])))

    def abrir_coeficientes_modelo(self) -> None:
        if self.resultado_actual:
            self._abrir_ruta(Path(str(self.resultado_actual["coefficients_path"])))

    def abrir_carpeta_modelo(self) -> None:
        if self.resultado_actual:
            self._abrir_ruta(Path(str(self.resultado_actual["output_dir"])))

    def _abrir_ruta(self, ruta: Path) -> None:
        if not ruta.exists():
            messagebox.showwarning("Ruta no encontrada", f"No se encontro la ruta: {ruta}")
            return
        os.startfile(str(ruta))


if __name__ == "__main__":
    app = AplicacionAnalisisCompuesto()
    app.mainloop()
