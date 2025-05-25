import pandas as pd
from datetime import datetime
import re


def reemplazar_meses(fecha: str) -> str:
    """Convierte meses en español a números en strings."""
    meses = {
        "ene": "01", "feb": "02", "mar": "03", "abr": "04",
        "may": "05", "jun": "06", "jul": "07", "ago": "08",
        "sep": "09", "oct": "10", "nov": "11", "dic": "12"
    }
    if not isinstance(fecha, str):
        return fecha
    for esp, num in meses.items():
        if esp in fecha.lower():
            fecha = fecha.lower().replace(esp, num)
    return fecha

def convertir_fecha(fecha_str: str) -> pd.Timestamp:
    """Convierte una fecha en string o ISO a datetime."""
    try:
        fecha_str = reemplazar_meses(fecha_str)
        if isinstance(fecha_str, str) and "/" not in fecha_str and "-" not in fecha_str:
            return datetime.strptime(fecha_str, "%d %m %Y")
        return pd.to_datetime(fecha_str, errors='coerce')
    except Exception:
        return pd.NaT

def convertir_a_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma la columna Fecha_operacion a datetime robusto."""
    df["Fecha_operacion"] = df["Fecha_operacion"].apply(convertir_fecha)
    return df

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

def normalizar_columnas(df):
    """ Convierte los nombres de las columnas a minúsculas y reemplaza espacios por guiones basjos"""
    df.columns = df.columns.str.lower().str.replace(" ","_")
    return df

def limpiar_texto(texto):
    """ Limpia el texto eliminando URLs, números y caracteres especiales"""    
    texto = str(texto).lower()
    texto = re.sub(r'https?://\S+', '', texto)                  # quitar URLs
    texto = re.sub(r'[\d]+', '', texto)                         # quitar números
    texto = re.sub(r'[\*\.,:;/\-_"\'\(\)]+', ' ', texto)        # quitar símbolos comunes
    texto = re.sub(r'\s+', ' ', texto).strip()                  # quitar espacios repetidos
    return texto

def eliminar_palabras(texto):
    """Elimina palabras/frases específicas del texto."""
    palabras = ["pago movil", "tarj"]  # Personaliza esta lista
    texto = str(texto).lower()
    for palabra in palabras:
        texto = re.sub(re.escape(palabra), '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', ' ', texto).strip()  # limpiar espacios extra
    return texto

# def mover_outliers_a_gastos_extra(df):
#     df = df.copy()

#     for cat in df['categoria'].unique():
#         datos = df[df['categoria'] == cat]['importe']
#         q1, q3 = datos.quantile([0.25, 0.9])
#         iqr = q3 - q1
#         bajo = q1 - 1.5 * iqr
#         alto = q3 + 1.5 * iqr

#         es_outlier = (df['categoria'] == cat) & ((df['importe'] < bajo) | (df['importe'] > alto))
#         df.loc[es_outlier, 'categoria'] = 'Gastos extraordinarios'

#     return df

def mover_outliers_a_gastos_extra(df):
    df = df.copy()

    for cat in df['categoria'].unique():
        # Ignorar cualquier categoría si el tipo asociado a ella es "ingreso"
        if df[df['categoria'] == cat]['tipo'].iloc[0].lower() == "ingreso":
            continue

        datos = df[df['categoria'] == cat]['importe']
        q1, q3 = datos.quantile([0.25, 0.9])
        iqr = q3 - q1
        bajo = q1 - 1.5 * iqr
        alto = q3 + 1.5 * iqr

        es_outlier = (df['categoria'] == cat) & ((df['importe'] < bajo) | (df['importe'] > alto))
        df.loc[es_outlier, 'categoria'] = 'Gastos extraordinarios'

    return df

def eliminar_outliers_prestamo(df, columna_categoria='categoria', columna_valor='importe'):
    """
    Elimina outliers (valores atípicos) dentro de la categoría 'Préstamo' 
    usando el método del rango intercuartílico (IQR). 
    No modifica otras categorías.
    """
    df_filtrado = df.copy()
    df_filtrado[columna_valor] = pd.to_numeric(df_filtrado[columna_valor], errors='coerce')

    # Aislar los valores de 'Préstamo'
    prestamos = df_filtrado[df_filtrado[columna_categoria] == 'Préstamo']

    q1 = prestamos[columna_valor].quantile(0.25)
    q3 = prestamos[columna_valor].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    # Filtrar los 'Préstamo' que están dentro del rango
    prestamos_filtrados = prestamos[
        (prestamos[columna_valor] >= limite_inferior) &
        (prestamos[columna_valor] <= limite_superior)
    ]

    # Conservar todas las demás categorías sin tocar
    df_restantes = df_filtrado[df_filtrado[columna_categoria] != 'Préstamo']

    # Reunir ambos
    df_final = pd.concat([df_restantes, prestamos_filtrados], ignore_index=True)

    return df_final


