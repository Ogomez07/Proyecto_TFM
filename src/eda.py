import fitz  # PyMuPDF
import pandas as pd
import re
import pandas as pd
from io import BytesIO
from typing import List, Optional
# Si no recibe bien otros formatos streamlit usar esta función -->

# def abrir_pdf_y_extraer_texto(pdf_input) 
#     """
#     Extrae el texto completo de un PDF.
#     Acepta una ruta de archivo (str) o un archivo subido en memoria (BytesIO).
#     """

#     if isinstance(pdf_input, str):
#         doc = fitz.open(pdf_input)
#     elif isinstance(pdf_input, BytesIO):
#         pdf_input.seek(0) 
#         doc = fitz.open(stream=pdf_input.read(), filetype="pdf")
#     else:
#         raise ValueError("❌ Formato de entrada no soportado para el PDF.")

#     texto = "".join([pagina.get_text() for pagina in doc])
#     doc.close()
#     return texto

def abrir_pdf_y_extraer_texto(pdf_path):
    """Extrae el texto completo de un PDF."""
    with fitz.open(pdf_path) as doc:
        texto = "".join([pagina.get_text() for pagina in doc])
    return texto

def extraer_movimientos_formato1(texto):
    """Extrae movimientos para el formato 1 (con 'F. valor')."""
    patron = re.compile(
        r"(?P<fecha>\d{1,2} \w{3} 20\d{2})\s+F\. valor:.*?\n(?P<operacion>.*?)(?P<importe>[−\-]?\d{1,3}(?:\.\d{3})*,\d{2})€\s+(?P<saldo>\d{1,3}(?:\.\d{3})*,\d{2})€",
        re.DOTALL
    )

    datos = []
    for match in patron.finditer(texto):
        try:
            fecha = match.group("fecha").strip()
            operacion = match.group("operacion").replace('\n', ' ').strip()
            importe = float(match.group("importe").replace('−', '-').replace('.', '').replace(',', '.'))
            saldo = float(match.group("saldo").replace('.', '').replace(',', '.'))
            datos.append([fecha, operacion, importe, saldo])
        except ValueError:
            continue

    return pd.DataFrame(datos, columns=["Fecha_operacion", "Operacion", "Importe", "Saldo"])

def extraer_movimientos_formato2(texto):
    """Extrae movimientos para el formato 2 (con fechas tipo dd-mm-yy)."""
    lineas = texto.split('\n')
    movimientos = []
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        match = re.match(
            r"^(\d{2}-\d{2}-\d{2})\s+(?:\d{2}-\d{2}-\d{2}\s+){1,2}\d+\s+(.*?)(\d{1,3}(?:,\d{2}))\s+([DH])(?:\s+(\d{1,3}(?:,\d{2}))\s+H)?", 
            linea
        )
        if match:
            fecha = match.group(1)
            descripcion = match.group(2).strip()
            importe = float(match.group(3).replace(',', '.'))
            signo = -1 if match.group(4) == 'D' else 1
            importe *= signo
            saldo = match.group(5)
            saldo = float(saldo.replace(',', '.')) if saldo else None

            concepto_extra = lineas[i + 1].strip() if i + 1 < len(lineas) else ""
            if concepto_extra and not re.match(r"^\d{2}-\d{2}-\d{2}", concepto_extra):
                descripcion += " " + concepto_extra
                i += 1

            movimientos.append([fecha, descripcion, importe, saldo])
        i += 1

    df = pd.DataFrame(movimientos, columns=["Fecha_operacion", "Operacion", "Importe", "Saldo"])
    df["Fecha_operacion"] = pd.to_datetime(df["Fecha_operacion"], format="%d-%m-%y")
    return df

def extraer_movimientos_formato3(texto):
    import pandas as pd
    import re

    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    movimientos = []

    i = 0
    while i + 3 < len(lineas):
        concepto = lineas[i]
        fecha = lineas[i + 1]
        importe = lineas[i + 2]
        saldo = lineas[i + 3]

        if re.match(r"\d{2}/\d{2}/\d{4}", fecha) and "€" in importe and "€" in saldo:
            try:
                fecha_dt = pd.to_datetime(fecha, dayfirst=True)
                importe_val = float(importe.replace('−', '-').replace('–', '-').replace('.', '').replace(',', '.').replace('€', ''))
                saldo_val = float(saldo.replace('.', '').replace(',', '.').replace('€', ''))
                movimientos.append([fecha_dt, concepto, importe_val, saldo_val])
                i += 4
            except Exception:
                i += 1
        else:
            i += 1

    if not movimientos:
        raise ValueError("❌ No se pudieron extraer movimientos del PDF de Caixa.")

    return pd.DataFrame(movimientos, columns=["Fecha_operacion", "Operacion", "Importe", "Saldo"])



def detectar_formato(texto):
    """Detecta el formato del PDF analizando el texto."""
    if "F. valor" in texto:
        return "formato1"
    elif re.search(r"\d{2}-\d{2}-\d{2}.*[DH]", texto):
        return "formato2"
    elif re.search(r"Concepto\s*Fecha\s*Importe\s*Saldo", texto):
        return "formato3"
    else:
        raise ValueError("❌ No se pudo detectar el formato del PDF.")

def extraer_movimientos(pdf_path, formato = "auto"):
    """Extrae los movimientos bancarios desde un PDF según su formato."""
    texto = abrir_pdf_y_extraer_texto(pdf_path)

    if formato == "auto":
        formato = detectar_formato(texto)

    if formato == "formato1":
        return extraer_movimientos_formato1(texto)
    elif formato == "formato2":
        return extraer_movimientos_formato2(texto)
    elif formato == "formato3":
        return extraer_movimientos_formato3(texto)
    else:
        raise ValueError("❌ Formato no reconocido. Usa 'formato1', 'formato2', 'formato3' o 'auto'.")
