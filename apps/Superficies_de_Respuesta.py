import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import mean_squared_error, r2_score
    import io
    return go, io, mean_squared_error, mo, np, pl, r2_score


@app.cell
def _():
    from scipy.optimize import curve_fit
    return (curve_fit,)


@app.cell
def _():
    MAIN_Y = "w/c"
    MAIN_X = "CEM (kg/m3)"
    N_GRID = 251
    return MAIN_X, MAIN_Y, N_GRID


@app.cell
def _(mo):
    file_button = mo.ui.file(kind="button", label="Cargar archivo CSV", )
    file_button
    return (file_button,)


@app.cell
def _(file_button, io, mo, pl):
    if len(file_button.value) == 0:
        mo.md("## ⬆️ Por favor, carga un archivo Excel para comenzar el análisis.")
        mo.stop(True)

    import re

    file_content_bytes = file_button.value[0].contents

    try:
        sample_content = file_content_bytes[:4096].decode('utf-8')
    except UnicodeDecodeError:
        sample_content = file_content_bytes[:4096].decode('latin-1')

    if re.search(r'\d+,\d+', sample_content):
        decimal_sep = True
    else:
        decimal_sep = False

    df_complete = pl.read_csv(
        io.BytesIO(file_content_bytes),
        separator=";",
        decimal_comma=decimal_sep
    )
    df_complete = df_complete.with_columns(
        pl.col("w/c").forward_fill(),
        pl.col("CEM (kg/m3)").forward_fill()
    )
    print("Archivo cargado y procesado.")
    return (df_complete,)


@app.cell
def _(df_complete, outputs, pl):
    df = df_complete.group_by(["w/c", "CEM (kg/m3)"]).agg([pl.mean(col) for col in outputs])
    return (df,)


@app.cell
def _(MAIN_X, MAIN_Y, pl):
    def select_x_y_from_col(df, col):
        # Corregido: Usar filtered_df para todas las columnas para evitar desajustes
        filtered_df = df.filter(pl.col(col).is_not_null())
        val_col = filtered_df[col].to_numpy()
        x = filtered_df[MAIN_X].to_numpy()
        y = filtered_df[MAIN_Y].to_numpy()
        return x, y, val_col
    return (select_x_y_from_col,)


@app.cell
def _():
    # Columnas a excluir del análisis de propiedades
    quitar = set(["Compresión 2 d (MPa)", "Compresión 3 d (MPa)", "Compresión 4 d (MPa)", "Mix", "Acabado"])
    return (quitar,)


@app.cell
def _(MAIN_X, MAIN_Y, df_complete, quitar):
    outputs = set(df_complete.columns)-set([MAIN_X, MAIN_Y])-quitar
    return (outputs,)


@app.cell(hide_code=True)
def _(mo, outputs):
    drop_cond = mo.ui.dropdown(options=sorted(list(outputs)), value='Consistencia fresco (cm)', label="Propiedad a analizar")
    drop_cond
    return (drop_cond,)


@app.cell
def _(np):
    def modelo_lineal(X, a, b, c):
        x, y = X
        return a*x + b*y + c

    def modelo_cuadratico(X, a, b, c, d, e, f):
        x, y = X
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    def modelo_exponencial(X, a, b, c):
        x, y = X
        return a * np.exp(b*x + c*y)

    def modelo_logaritmico(X, a, b, c):
        x, y = X
        # Añadir un pequeño épsilon para evitar log(0)
        x_safe = np.where(x > 0, x, 1e-9)
        y_safe = np.where(y > 0, y, 1e-9)
        return a + b*np.log(x_safe) + c*np.log(y_safe)

    def modelo_potencial(X, a, b, c):
        x, y = X
        # Añadir un pequeño épsilon para evitar 0 en la base
        x_safe = np.where(x > 0, x, 1e-9)
        y_safe = np.where(y > 0, y, 1e-9)
        return a * (x_safe**b) * (y_safe**c)

    modelos = {
        'Lineal': (modelo_lineal, [1, 1, 1]),
        'Cuadrático': (modelo_cuadratico, [1, 1, 1, 1, 1, 1]),
        'Exponencial': (modelo_exponencial, [1, 0.01, 0.01]),
        'Logarítmico': (modelo_logaritmico, [1, 1, 1]),
        'Potencial': (modelo_potencial, [1, 1, 1])
    }
    return (modelos,)


@app.cell
def _():
    funciones={
        'Lineal': "$f(x, y) = ax + by + c$",
        'Cuadrático': "$f(x, y) = ax^2 + by^2 + cxy + dx + ey + f$",
        'Exponencial': "$f(x, y) = a \cdot e^{bx + cy}$",
        'Logarítmico': "$f(x, y) = a + b \ln(x) + c \ln(y)$",
        'Potencial': "$f(x, y) = a x^b y^c$"
    }
    return (funciones,)


@app.cell
def _(outputs):
    outputs_list = sorted(list(outputs))
    return (outputs_list,)


@app.cell
def _(
    curve_fit,
    df,
    mean_squared_error,
    modelos,
    np,
    outputs_list,
    pl,
    r2_score,
    select_x_y_from_col,
):
    best_ajustes = {}
    for out in outputs_list:
        x, y, val_col = select_x_y_from_col(df, out)
        if len(val_col) < 3: # Requiere al menos 3 puntos para un ajuste mínimo
            print(f"Propiedad: {out} - No hay suficientes datos para ajustar. Saltando.")
            continue

        print(f"Propiedad: {out}")
        resultados = []
        for modelo_name, (modelo_func, p0) in modelos.items():
            # El modelo logarítmico y potencial no puede manejar valores <= 0
            if modelo_name in ['Logarítmico', 'Potencial'] and (np.any(x <= 0) or np.any(y <= 0)):
                print(f"  Modelo: {modelo_name} no puede ajustarse con valores no positivos.")
                continue

            try:
                popt, pcov = curve_fit(modelo_func, (x, y), val_col, p0=p0, maxfev=10000)
                val_pred = modelo_func((x, y), *popt)
                rmse = np.sqrt(mean_squared_error(val_col, val_pred))
                r2 = r2_score(val_col, val_pred)
                resultados.append({
                    "Modelo": modelo_name,
                    "Parámetros": popt,
                    "RMSE": rmse,
                    "R2": r2
                })
                print(f"  Modelo: {modelo_name}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            except Exception as e:
                print(f"  Modelo: {modelo_name} no pudo ajustarse. Error: {e}")

        if not resultados:
            continue

        resultados_df = pl.DataFrame(resultados)
        mejores_modelos = resultados_df.sort('RMSE').row(0, named=True)
        best_ajustes[out] = {
            "Modelo": mejores_modelos["Modelo"],
            "Parámetros": mejores_modelos["Parámetros"],
            "RMSE": mejores_modelos["RMSE"],
            "R2": mejores_modelos["R2"]
        }
    return (best_ajustes,)


@app.cell
def _(best_ajustes, np, pl):
    lista_para_df = []
    for caracteristica, detalles in best_ajustes.items():
        parametros = list(detalles['Parámetros'])
        padded_params = (parametros + [np.nan] * 6)[:6]

        lista_para_df.append({
            'Caracteristicas': caracteristica,
            'Modelo': detalles['Modelo'],
            'RMSE': detalles['RMSE'],
            'R2': detalles['R2'],
            'a': padded_params[0],
            'b': padded_params[1],
            'c': padded_params[2],
            'd': padded_params[3],
            'e': padded_params[4],
            'f': padded_params[5]
        })

    # Crear el DataFrame de resultados
    if not lista_para_df:
        resultados_ajuste = pl.DataFrame()
    else:
        resultados_ajuste = pl.DataFrame(lista_para_df)
    return (resultados_ajuste,)


@app.cell
def _(MAIN_X, MAIN_Y, N_GRID, df, np):
    c_min, c_max = df[MAIN_X].min()-5, df[MAIN_X].max()+5
    w_min, w_max = df[MAIN_Y].min()-0.005, df[MAIN_Y].max()+0.005
    W = np.linspace(w_min, w_max, N_GRID)
    C = np.linspace(c_min, c_max, N_GRID)
    CC, WW = np.meshgrid(C, W, indexing="xy")
    return C, CC, W, WW


@app.cell
def _(
    CC,
    MAIN_X,
    MAIN_Y,
    WW,
    best_ajustes,
    df,
    df_complete,
    drop_cond,
    funciones,
    mo,
    modelos,
    np,
    pl,
):
    if drop_cond.value not in best_ajustes:
        md_model = mo.md(f"### No se pudo encontrar un modelo para **{drop_cond.value}**.")
        pred = np.array([])
        pred_real = np.array([])
        Zpred = np.array([[]])
    else:
        _model_name, _model_params,*_ = best_ajustes[drop_cond.value].values()
        _model_func, _ = modelos[_model_name]
        md_model = mo.md(f"### Mejor Modelo: **{_model_name}**\n\n{funciones[_model_name]}")
        _x_true = df_complete.filter(pl.col(drop_cond.value).is_not_null())[MAIN_X]
        _y_true = df_complete.filter(pl.col(drop_cond.value).is_not_null())[MAIN_Y]
        pred = _model_func((df[MAIN_X].to_numpy(), df[MAIN_Y].to_numpy()), *_model_params)
        pred_real = _model_func((_x_true.to_numpy(), _y_true.to_numpy()), *_model_params)
        Zpred = _model_func((CC.ravel(),WW.ravel()), *_model_params).reshape(WW.shape)
    return Zpred, md_model, pred, pred_real


@app.cell
def _(mean_squared_error, np, r2_score):


    def calculate_metrics(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_true_filtered) == 0:
            return np.nan, np.nan, np.nan

        rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
        r2 = r2_score(y_true_filtered, y_pred_filtered)

        # Cálculo de MAPE seguro
        non_zero_mask = y_true_filtered != 0
        mape = np.mean(np.abs((y_true_filtered[non_zero_mask] - y_pred_filtered[non_zero_mask]) / y_true_filtered[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0

        return mape, rmse, r2
    return (calculate_metrics,)


@app.cell
def _(calculate_metrics, df, drop_cond, mo, pl, pred):
    if pred.size == 0:
        md_error = mo.md("**Métricas no disponibles.**")
    else:
        mape_val, rmse_val, r2_val = calculate_metrics(df.filter(pl.col(drop_cond.value).is_not_null())[drop_cond.value], pred)
        md_error = mo.hstack([
            mo.stat("MAPE", f"{mape_val:.2f}%"), 
            mo.stat("RMSE", f"{rmse_val:.2f}"),
            mo.stat("R²", f"{r2_val:.2f}")
        ], justify="center", gap=2)
    return (md_error,)


@app.cell
def _(df, df_complete, drop_cond, go, mo, pl, pred, pred_real):
    y_true_series = df.filter(pl.col(drop_cond.value).is_not_null())[drop_cond.value]

    if y_true_series.is_empty():
        pred_plot = mo.md("No hay datos para el gráfico de predicción.")
    else:
        y_true_complete_series = df_complete.filter(pl.col(drop_cond.value).is_not_null())[drop_cond.value]
        _line = [y_true_series.min()*0.9, y_true_series.max()*1.1]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=y_true_series, y=pred, mode='markers', name='Media de datos', marker=dict(color="#6d6cf7", size=8, line=dict(color="blue"))))
        _fig.add_trace(go.Scatter(x=y_true_complete_series, y=pred_real, mode='markers', name='Datos completos', opacity=0.6, marker=dict(color="grey", line=dict(color="black"))))
        _fig.add_trace(go.Scatter(x=_line, y=_line, mode='lines', name='Línea 1:1', line=dict(dash='dash')))

        _fig.update_layout(
            xaxis_title="Valores reales",
            yaxis_title="Valores predichos",
            legend_title="Leyenda",
            width=600,
            height=600
        )
        pred_plot = mo.ui.plotly(_fig)
    return (pred_plot,)


@app.cell
def _(md_model):
    md_model
    return


@app.cell
def _(md_error):
    md_error
    return


@app.cell
def _(df_complete, drop_cond, mo, pl, select_x_y_from_col):
    _x, _y, _val_col = select_x_y_from_col(df_complete, drop_cond.value)
    if len(_val_col) > 0:
        aux = pl.DataFrame({"CEM (kg/m3)": _x, "w/c": _y, drop_cond.value: _val_col})
        ex_aux = mo.ui.data_editor(aux.to_pandas(), label="Datos de la propiedad seleccionada")
    else:
        ex_aux = mo.md("No hay datos para mostrar en la tabla.")
    return (ex_aux,)


@app.cell
def _(MAIN_X, MAIN_Y, df, df_complete, drop_cond, go, mo):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=df_complete[MAIN_X], y=df_complete[MAIN_Y], z=df_complete[drop_cond.value], 
                                mode='markers',opacity=0.8,
                                marker=dict(color='grey', size=5, line=dict(color="black", width=2)), name='Datos completos'))
    _fig.add_trace(go.Scatter3d(x=df[MAIN_X], y=df[MAIN_Y], z=df[drop_cond.value], mode='markers',
                                marker=dict(color='#6d6cf7', size=5, line=dict(color="blue", width=2)), name='Media de datos'))
    _fig.update_layout(
        scene=dict(
            xaxis_title=MAIN_X,
            yaxis_title=MAIN_Y,
            zaxis_title=drop_cond.value
        ),
        legend_title="Leyenda"
    )
    _fig.update_layout(width=500, height=500)
    d3_plot = mo.ui.plotly(_fig)
    return (d3_plot,)


@app.cell
def _(C, MAIN_X, MAIN_Y, W, Zpred, df, drop_cond, go, mo, select_x_y_from_col):
    if Zpred.size == 0:
        _fig = go.Figure()
        _fig.update_layout(title_text=f"No hay datos para mostrar para {drop_cond.value}")
    else:
        _x, _y, _val_col = select_x_y_from_col(df, drop_cond.value)
        if len(_val_col) == 0:
             _fig = go.Figure()
             _fig.update_layout(title_text=f"No hay datos de media para mostrar para {drop_cond.value}")
        else:
            z_min = Zpred.min()
            z_max = Zpred.max()
            _fig = go.Figure(data=[
                go.Heatmap(
                    z=Zpred,
                    x=C,
                    y=W,
                    colorscale="viridis",
                    colorbar=dict(title=drop_cond.value)
                ),
                go.Contour(
                    z=Zpred,
                    x=C,
                    y=W,
                    colorscale='greys',
                    line_width=2,
                    contours=dict(
                        coloring='lines',
                        showlabels=True,
                    )
                )
            ])
            _fig.add_trace(go.Scatter(x=_x, y=_y, mode='markers',
                                      marker=dict(color=_val_col, colorscale='Viridis', showscale=False, cmin=z_min, cmax=z_max,line=dict(color="black", width=1)),name=drop_cond.value, hovertemplate=f'{MAIN_X}: %{{x}}<br>{MAIN_Y}: %{{y}}<br>{drop_cond.value}: %{{marker.color}}'
                                     ))
            _fig.update_layout(width=900, height=700, title=f"Superficie de Respuesta para {drop_cond.value}")
            _fig.update_xaxes(range=[min(_x)-5, max(_x)+5])
            _fig.update_yaxes(range=[min(_y)-0.005, max(_y)+0.005])

    surf_plot = mo.ui.plotly(_fig)
    return (surf_plot,)


@app.cell
def _(d3_plot, ex_aux, mo, pred_plot, resultados_ajuste, surf_plot):
    if resultados_ajuste.is_empty():
        tabla_resultados = mo.md("No se han podido generar resultados de ajuste.")
    else:
        tabla_resultados = mo.ui.table(resultados_ajuste.to_pandas(), page_size=20, selection=None, show_data_types=False, freeze_columns_left=["Caracteristicas"], label="Tabla de resultados del mejor ajuste por propiedad")

    mo.ui.tabs({
        "Visualización del Ajuste": mo.vstack([
            mo.hstack([d3_plot, surf_plot], gap=0, justify="start", align="center"),
        ]),
        "Análisis de Predicción": mo.hstack([pred_plot,ex_aux],gap=1,widths=[2, 3], justify="center", align="center"),
        "Resultados Globales": tabla_resultados
    })
    return


if __name__ == "__main__":
    app.run()
