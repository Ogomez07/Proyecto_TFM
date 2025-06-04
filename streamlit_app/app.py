# streamlit/app.py

import streamlit as st
import pandas as pd
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.eda as eda
import src.etl as etl
import models.categorizacion as categorizar
import models.prediccion as prediccion
import src.visualizaciones as visualizaciones  # Para graficar


st.title("🧾 Extractor de movimientos bancarios desde PDF")

st.write("Sube tu archivo PDF con movimientos bancarios. El sistema detectará automáticamente el formato.")


archivos_pdf = st.file_uploader(
    "📤 Sube tus PDFs", 
    type=["pdf"], 
    accept_multiple_files=True,  # <-- aquí permites subir varios archivos
    key="uploader_pdfs"
)

if archivos_pdf:
    dataframes = []

    for archivo in archivos_pdf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(archivo.read())
        ruta_pdf = tmp.name
        tmp.close()  # Muy importante

        try:
            df = eda.extraer_movimientos(ruta_pdf, formato="auto")
            dataframes.append(df)

        except Exception as e:
            st.error(f"❌ Error al procesar {archivo.name}: {e}")

        finally:
            os.unlink(ruta_pdf)  # Ahora sí puedes borrarlo

    # Combinar todos los DataFrames
    df_total = pd.concat(dataframes, ignore_index=True)

    st.success(f"✔ Se extrajeron {len(df_total)} movimientos de {len(dataframes)} archivos.")
    st.dataframe(df_total)

    # ========================
    # 🔥 LIMPIEZA DE DATOS
    # ========================
    st.subheader("🧹 Limpieza de movimientos")
    df_total = etl.convertir_a_datetime(df_total)  # ¡Aplica a df_total!
    df_total = etl.normalizar_columnas(df_total)

    df_total['operacion_limpia'] = df_total['operacion'].apply(etl.limpiar_texto)
    df_total['tipo'] = df_total['importe'].apply(lambda i: "ingreso" if i > 0 else "gasto")
    df_total['año_mes'] = df_total['fecha_operacion'].dt.to_period("M").astype(str)

    if 'saldo' in df_total.columns:
        df_total = df_total.drop(columns=['saldo'])

    st.write("🔍 Datos procesados:")
    st.dataframe(df_total.head())

    # ========================
    # 🏷️ CATEGORIZACIÓN DE MOVIMIENTOS
    # ========================
    st.subheader("🏷️ Clasificación automática de movimientos")

    df_total['categoria'] = df_total['operacion_limpia'].apply(categorizar.clasificar_por_reglas)

    st.write("📚 Movimientos clasificados:")
    st.dataframe(df_total[['fecha_operacion', 'operacion', 'importe', 'categoria']].head())

    # ========================
    # ⬇️ DESCARGAR CSV CATEGORIZADO
    # ========================
    csv_categorizado = df_total.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 Descargar CSV categorizado",
        csv_categorizado,
        file_name="movimientos_categorizados.csv",
        mime="text/csv"
    )

    # ========================
    # 🔮 PREDICCIÓN DE GASTOS
    # ========================
    st.subheader("🔮 Predicción de gastos futuros por categoría")

    # Filtrar solo gastos
    df_gastos = df_total[df_total['tipo'] == 'gasto']
    df_gastos['importe'] = df_gastos['importe'].abs()

    # Agrupar gastos mensuales por categoría
    gastos_mensuales = df_gastos.groupby(['año_mes', 'categoria'])['importe'].sum().unstack(fill_value=0)

    categorias_disponibles = gastos_mensuales.columns.tolist()

    categoria_seleccionada = st.selectbox("Selecciona una categoría para predecir:", categorias_disponibles)

    if categoria_seleccionada:
        serie = gastos_mensuales[categoria_seleccionada]

        serie_train, fechas, pred, reales = prediccion.predecir_naive_media(serie)

        st.write(f"📈 Predicción de gastos futuros para **{categoria_seleccionada}**:")

        fig = visualizaciones.graficar_predicciones(
            serie_train, 
            fechas, 
            reales, 
            pred, 
            categoria_seleccionada, 
            6    # <--- pasas el 6 directamente, sin "meses_pred="
        )

        df_resultados = prediccion.mostrar_resultado(fechas, [pred]*len(fechas), reales)
        st.dataframe(df_resultados)
