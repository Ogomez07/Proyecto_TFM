import pandas as pd

def resumir_movimientos(df):

    # Aseguramos que fecha_operacion está en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(df['fecha_operacion']):
        df['fecha_operacion'] = pd.to_datetime(df['fecha_operacion'], errors='coerce')

    # Crear columna 'fecha_mes' si no existe
    if 'fecha_mes' not in df.columns:
        df['fecha_mes'] = df['fecha_operacion'].dt.to_period("M").dt.to_timestamp()

    # Seleccionar solo columnas necesarias (asegura que todas existen)
    columnas_necesarias = ['categoria', 'importe', 'fecha_mes']
    df = df[[col for col in columnas_necesarias if col in df.columns]]

    return df

def contexto_valido(df):
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

def extraer_movimientos(df):
    if df.empty:
        return 0.0, 0.0

    df = df.copy()
    df['fecha_mes'] = pd.to_datetime(df['fecha_operacion']).dt.to_period('M').dt.to_timestamp()

    # Ingresos mensuales promedio
    ingresos_mensuales = df[df['categoria'].str.lower() == 'ingreso'] \
        .groupby('fecha_mes')['importe'].sum().mean()

    # Gastos fijos mensuales promedio
    categorias_fijas = ['supermercado', 'facturas', 'préstamo', 'transporte']
    gastos_mensuales = df[df['categoria'].str.lower().isin(categorias_fijas)] \
        .groupby('fecha_mes')['importe'].sum().mean()

    return ingresos_mensuales, gastos_mensuales

