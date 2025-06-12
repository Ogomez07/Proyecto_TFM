import pandas as pd

def resumir_movimientos(data):
    import pandas as pd

    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data  # Si ya es DataFrame, úsalo directamente

    df['fecha_operacion'] = pd.to_datetime(df['fecha_operacion'])

    # Separar ingresos y gastos
    df_gastos = df[df['tipo'] == 'gasto']
    df_ingresos = df[df['tipo'] == 'ingreso']

    # Calcular ingreso mensual promedio
    n_meses = df['fecha_operacion'].dt.to_period("M").nunique()
    ingresos_mensuales = df_ingresos['importe'].sum() / n_meses if n_meses > 0 else 0

    # Agrupar gastos por categoría
    gastos_por_categoria = df_gastos.groupby("categoria")["importe"].sum().to_dict()

    return {
        "ingresos_mensuales": ingresos_mensuales,
        "gastos": gastos_por_categoria
    }
