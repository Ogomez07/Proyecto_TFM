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

