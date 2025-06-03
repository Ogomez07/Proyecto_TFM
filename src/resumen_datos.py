import pandas as pd

def resumir_movimientos(filepath: str) -> str:
    """
    Carga un archivo CSV con movimientos bancarios y genera un resumen de importes mensuales
    por categoría (incluyendo ingresos) para los últimos 18 meses.
    """
    # Cargar los datos
    df = pd.read_csv(filepath)

    # Convertir fecha y redondear al fin de mes
    df["fecha_operacion"] = pd.to_datetime(df["fecha_operacion"])
    df["fecha_operacion"] = df["fecha_operacion"] + pd.offsets.MonthEnd(0)

    # Filtrar últimos 18 meses desde la fecha más reciente
    fecha_max = df["fecha_operacion"].max()
    fecha_min = fecha_max - pd.DateOffset(months=18)
    df_filtrado = df[df["fecha_operacion"] > fecha_min]

    # Agrupar por mes y categoría
    gastos_mensuales = (
        df_filtrado.groupby([pd.Grouper(key="fecha_operacion", freq="ME"), "categoria"])["importe"].sum().unstack(fill_value=0))

    # Generar resumen de texto para IA
    resumen = "Resumen de importes mensuales por categoría en los últimos 18 meses:\n"
    for categoria in gastos_mensuales.columns:
        valores = gastos_mensuales[categoria].round(2).tolist()
        resumen += f"\n {categoria}: {valores}\n"

    return resumen.strip()
