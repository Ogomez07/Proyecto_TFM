import pandas as pd

def resumir_movimientos(data):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data  # si es DataFrame, úsalo directamente

    # Asegúrate de que 'fecha_operacion' esté en datetime
    df['fecha_operacion'] = pd.to_datetime(df['fecha_operacion'])

    df_filtrado = df[df['tipo'] == 'gasto']

    resumen_texto = (
        f"En el periodo analizado has tenido un gasto total de {df_filtrado['importe'].sum():.2f} €.\n"
        f"Gastos por categoría:\n"
        f"{df_filtrado.groupby('categoria')['importe'].sum().to_string()}\n\n"
        f"Gastos mensuales por categoría:\n"
        f"{df_filtrado.groupby([pd.Grouper(key='fecha_operacion', freq='ME'), 'categoria'])['importe'].sum().unstack(fill_value=0)}"
    )

    return resumen_texto
