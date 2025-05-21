import pandas as pd


def reemplazar_meses(fecha: str) -> str:
    """Convierte meses en español a número en strings."""
    meses = {
        "ene": "01", "feb": "02", "mar": "03", "abr": "04",
        "may": "05", "jun": "06", "jul": "07", "ago": "08",
        "sep": "09", "oct": "10", "nov": "11", "dic": "12"
    }
    if not isinstance(fecha, str):
        return fecha  # Deja datetime o NaT tal cual
    for esp, num in meses.items():
        if esp in fecha:
            fecha = fecha.replace(esp, num)
    return fecha


def formatear_fecha(df: pd.DataFrame, columna_fecha: str = "Fecha_operacion") -> pd.DataFrame:
    """Convierte fechas tipo texto al formato datetime, ignorando las que ya lo son."""
    df = df.copy()
    
    # Detectar las que siguen siendo strings
    es_string = df[columna_fecha].apply(lambda x: isinstance(x, str))

    # Paso 1: Intentar formato con barra
    df.loc[es_string, columna_fecha] = df.loc[es_string, columna_fecha].str.replace(" ", "/", n=2, regex=False)
    df.loc[es_string, columna_fecha] = pd.to_datetime(df.loc[es_string, columna_fecha], format="%d/%m/%Y", errors="coerce")

    # Paso 2: Detectar errores y probar con guiones y año corto
    es_string_fallido = df[columna_fecha].isna() & df[columna_fecha].astype(str).str.match(r"\d{2}-\d{2}-\d{2}")
    df.loc[es_string_fallido, columna_fecha] = pd.to_datetime(df.loc[es_string_fallido, columna_fecha], format="%d-%m-%y", errors="coerce")

    return df

""" REVISAR MAÑANA"""
# def convertir_fechas_sin_romper(df: pd.DataFrame, columna: str = "Fecha_operacion") -> pd.DataFrame: 
#     """Convierte solo los strings con meses en español, sin tocar datetimes."""
#     df = df.copy()

#     # Identificar qué entradas son strings
#     es_str = df[columna].apply(lambda x: isinstance(x, str))

#     # Reemplazar meses en español (solo si es string)
#     df.loc[es_str, columna] = df.loc[es_str, columna].apply(reemplazar_meses_espanol)

#     # Cambiar espacios por barras solo en strings
#     df.loc[es_str, columna] = df.loc[es_str, columna].str.replace(" ", "/", n=2, regex=False)

#     # Aplicar conversión SOLO a los strings
#     df.loc[es_str, columna] = pd.to_datetime(
#         df.loc[es_str, columna], format="%d/%m/%Y", errors="coerce"
#     )

    #return df

def agregar_movimiento(df: pd.DataFrame, fecha: str, concepto: str, importe: float, saldo=None) -> pd.DataFrame:
    """Agrega manualmente un movimiento al DataFrame."""
    nueva_fila = pd.DataFrame([{
        "Fecha_operacion": fecha,
        "Operacion": concepto,
        "Importe": importe,
        "Saldo": saldo
    }])
    df = pd.concat([df, nueva_fila], ignore_index=True)
    df["Fecha_operacion"] = pd.to_datetime(df["Fecha_operacion"])
    return df
