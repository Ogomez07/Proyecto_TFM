# streamlit/app.py

import streamlit as st
import pandas as pd
import tempfile
import os
from src import eda

st.title("🧾 Extractor de movimientos bancarios desde PDF")

st.write("Sube tu archivo PDF con movimientos bancarios. El sistema detectará automáticamente el formato.")

archivo_pdf = st.file_uploader("📤 Sube tu PDF", type=["pdf"])

if archivo_pdf:
    # Guardar temporalmente el archivo PDF subido
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(archivo_pdf.read())
        ruta_pdf = tmp.name

    try:
        df = eda.extraer_movimientos(ruta_pdf, formato="auto")
        st.success(f"✔ Se extrajeron {len(df)} movimientos.")
        st.dataframe(df)

        # Permitir descarga
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 Descargar CSV", csv, file_name="movimientos_extraidos.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {e}")

    finally:
        os.unlink(ruta_pdf)  # Limpiar archivo temporal
