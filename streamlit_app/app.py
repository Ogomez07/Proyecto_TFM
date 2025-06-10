# streamlit/app.py

import streamlit as st
import importlib
import pandas as pd
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.eda as eda
import src.etl as etl
import models.categorizacion as categorizar
import models.prediccion as prediccion
import viz_app as viz 
import models.ia_asesor as ia_asesor
importlib.reload(ia_asesor)

import src.resumen_datos as resumen
import streamlit_app.historial as historial

# ========================
# Inicializar session_state
# ========================
if 'df_total' not in st.session_state:
    st.session_state['df_total'] = None


# Título general
st.title("Asistente Financiero Inteligente")

# Barra de navegación
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una página", [
    "Extracción de movimientos",
    "Predicciones por categorías",
    "Asesor financiero"
])


# ========================
# 🚀 Página 1: Extracción de movimientos
# ========================
if page == "Extracción de movimientos":
    st.header("🧾 Extractor de movimientos")
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
        # st.dataframe(df_total)

        # ========================
        # 🔥 LIMPIEZA DE DATOS
        # ========================
        # st.subheader("🧹 Limpieza de movimientos")
        df_total = etl.convertir_a_datetime(df_total)  # ¡Aplica a df_total!
        df_total = etl.normalizar_columnas(df_total)

        df_total['operacion_limpia'] = df_total['operacion'].apply(etl.limpiar_texto)
        df_total['tipo'] = df_total['importe'].apply(lambda i: "ingreso" if i > 0 else "gasto")
        df_total['año_mes'] = df_total['fecha_operacion'].dt.to_period("M").astype(str)

        if 'saldo' in df_total.columns:
            df_total = df_total.drop(columns=['saldo'])

        # Clasificación automática por reglas
        df_total['categoria'] = df_total['operacion_limpia'].apply(categorizar.clasificar_por_reglas)

        # Guarda df_total en session_state
        st.session_state.df_total = df_total
        st.session_state.df_total_limpio = df_total.copy()
                

        # st.write("🔍 Datos procesados:")
        # st.dataframe(df_total.head())

# ========================
# 🚀 Página 2: Predicciones por categorías
# ========================
elif page == "Predicciones por categorías":
    st.header("Predicciones por categoría")

    if st.session_state.df_total is None:
        st.error("⚠️ No has subido aún ningún PDF. Por favor ve a 'Extracción de movimientos' primero.")
    else:
        df_total = st.session_state.df_total.copy()

        # st.subheader("Clasificación automática de movimientos")
        # Guarda df_total en session_state
        st.session_state.df_total = df_total
        # Guarda una copia limpia antes de modificar
        st.session_state.df_total_limpio = df_total.copy()

        x_train, x_test, y_train, y_test, x_text_train, x_text_test, vectorizer = categorizar.preparar_datos_modelo(df_total)
        modelo = categorizar.entrenar_modelo_clasificador(x_train, y_train)

        evaluacion = categorizar.evaluar_modelo(modelo, x_test, y_test, x_text_test)
        # st.text(f"📊 Evaluación del modelo:\n{evaluacion}")

        df_sin_etiquetar = categorizar.filtrar_movimientos_sin_categoria(df_total)

        if not df_sin_etiquetar.empty:
            df_sin_etiquetar = categorizar.predecir_categorias(df_sin_etiquetar, vectorizer, modelo)
            df_total = categorizar.actualizar_categorias(df_total, df_sin_etiquetar)

        # ELIMINACIÓN DE OUTLIERS Y AJUSTES DE IMPORTE
        df_total = etl.mover_outliers_a_gastos_extra(df_total)
        df_total = etl.eliminar_outliers_prestamo(df_total)
        df_total['importe'] = df_total['importe'].abs()

        st.success("✔️ Movimientos clasificados correctamente.")
        #st.dataframe(df_total[['fecha_operacion', 'operacion', 'importe', 'categoria']].head())

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

        df_gastos = df_total[df_total['tipo'] == 'gasto']
        df_gastos['importe'] = df_gastos['importe'].abs()

        st.markdown("## 📊 **Resumen de gastos e ingresos para el periodo elegido**")
        # Extraer año y mes en columnas auxiliares
        df_total['año'] = df_total['fecha_operacion'].dt.year
        df_total['mes'] = df_total['fecha_operacion'].dt.month

        # Crear listas de años y meses disponibles
        años_disponibles = df_total['año'].sort_values().unique().tolist()
        meses_disponibles = list(range(1, 13))  # 1=enero, 2=febrero, ...

        # Selectboxes para elegir año y mes
        st.markdown("### 🗓️ *Filtrar por año y mes*")
        año_seleccionado = st.selectbox("Selecciona el año", años_disponibles)
        mes_seleccionado = st.selectbox("Selecciona el mes", meses_disponibles)

        # Filtro de datos por año y mes seleccionados
        df_filtrado = df_total[(df_total['año'] == año_seleccionado) & (df_total['mes'] == mes_seleccionado)]

        # ========================
        # Tabla Resumen
        # ========================
        tabla_resumen = df_filtrado.groupby('categoria')['importe'].sum().reset_index()
        tabla_resumen = tabla_resumen.sort_values(by='importe', ascending=False)
        st.dataframe(tabla_resumen)


        gastos_mensuales = df_gastos.groupby(['año_mes', 'categoria'])['importe'].sum().unstack(fill_value=0)

        categorias_disponibles = gastos_mensuales.columns.tolist()

        # Lista de categorías a excluir
        categorias_a_excluir = ['Ingreso', 'Gastos extraordinarios']

        # Filtrar categorías
        categorias_filtradas = [cat for cat in categorias_disponibles if cat not in categorias_a_excluir]


        categoria_seleccionada = st.selectbox("Selecciona una categoría para predecir:", categorias_filtradas)

        if categoria_seleccionada:
            serie = gastos_mensuales[categoria_seleccionada]

            serie_train, fechas, pred, reales = prediccion.predecir_naive_media(serie)

            st.write(f"📈 Predicción de gastos futuros para **{categoria_seleccionada}**:")

            fig = viz.graficar_predicciones(
                    serie_train, 
                    fechas, 
                    reales, 
                    pred, 
                    categoria_seleccionada, 
                    6
                )
            st.pyplot(fig)

            df_resultados = prediccion.mostrar_resultado(fechas, [pred]*len(fechas), reales)
            st.dataframe(df_resultados)

# ========================
# 🚀 Página 3: Asesor financiero
# ========================
elif page == "Asesor financiero":
    st.header("💬 Asesor Financiero Inteligente")

    if st.session_state.df_total is None:
        st.error("⚠️ No has subido aún ningún PDF. Por favor ve a 'Extracción de movimientos' primero.")
    else:
        df_total = st.session_state.df_total.copy()

        # ✅ AÑADIDO NECESARIO: columna 'fecha' para compatibilidad con ia_asesor
        df_total['fecha'] = pd.to_datetime(df_total['fecha_operacion'])

        # ✅ Inicializar contexto si no existe
        if 'contexto' not in st.session_state:
            st.session_state.contexto = historial.resumir_movimientos(df_total)

        contexto = st.session_state.contexto

        st.markdown("### 🧠 ¿Qué deseas consultar?")
        opcion = st.selectbox("Selecciona un servicio:", [
            "📈 Proyección de gastos futuros",
            "💬 Chat con el Asesor Financiero IA",
            "📅 Planificador de Ahorro Personalizado",
            "🚨 Alerta de gasto excesivo",
            "🏡 Asesoría para compra de vivienda",
            "🏦 Estrategia para pago de deudas"
        ])

        if opcion == "📈 Proyección de gastos futuros":
            st.subheader("📈 Proyección de gastos futuros")
            categoria = st.selectbox("Selecciona categoría a proyectar:", df_total['categoria'].unique())

            if st.button("Generar proyección"):
                respuesta = ia_asesor.proyeccion_gastos_futuros(categoria, df_total)
                st.success(respuesta)

        elif opcion == "💬 Chat con el Asesor Financiero IA":
            st.subheader("💬 Chat con el Asesor Financiero IA")
            pregunta_usuario = st.text_input("¿Qué deseas preguntar al asesor financiero?")

            if st.button("Enviar pregunta"):
                if pregunta_usuario:
                    respuesta_ia = ia_asesor.asesor_con_contexto(pregunta_usuario, contexto)
                    st.success(respuesta_ia)
                else:
                    st.warning("⚠️ Por favor, escribe tu pregunta antes de enviar.")

        elif opcion == "📅 Planificador de Ahorro Personalizado":
            st.subheader("📅 Planificador de Ahorro")
            monto_objetivo = st.number_input("💰 ¿Cuánto dinero deseas ahorrar?", min_value=0.0)
            tiempo_disponible = st.number_input("🗓️ ¿En cuántos meses quieres ahorrar esa cantidad?", min_value=1)
            ingresos = st.number_input("📈 ¿Cuáles son tus ingresos mensuales?", min_value=0.0)
            gastos = st.number_input("📉 ¿Cuáles son tus gastos mensuales?", min_value=0.0)

            if st.button("Generar plan de ahorro"):
                plan_ahorro = ia_asesor.plan_ahorro_objetivo(
                    monto_objetivo, tiempo_disponible, ingresos, gastos,
                    usar_contexto=True, contexto_gastos=contexto
                )
                st.success(plan_ahorro)

        elif opcion == "🚨 Alerta de gasto excesivo":
            st.subheader("🚨 Alerta de gasto excesivo")
            if st.button("Analizar gastos"):
                alertas = ia_asesor.alerta_gasto_excesivo(df_total)
                if not alertas.empty:
                    st.dataframe(alertas)
                    st.markdown("### 📝 Resumen del Asesor:")
                    resumen_alerta = ia_asesor.resumen_alerta_gastos(alertas)
                    st.success(resumen_alerta)
                else:
                    st.success("✅ No se detectaron gastos excesivos.")

        elif opcion == "🏡 Asesoría para compra de vivienda":
            st.subheader("🏡 Asesoría para compra de vivienda")
            decision = st.selectbox("¿Qué deseas evaluar?", ["compra", "alquiler", "independizarme"])
            usar_chatgpt = st.checkbox("¿Deseas una recomendación avanzada con ChatGPT?", value=False)

            if st.button("Generar asesoría"):
                respuesta = ia_asesor.asesoria_vivienda(
                    opcion=decision,
                    usar_contexto=True,
                    contexto_gastos=contexto,
                    usar_chatgpt=usar_chatgpt
                )
                st.success(respuesta)

        elif opcion == "🏦 Estrategia para pago de deudas":
            st.subheader("🏦 Estrategia para pago de deudas")

            deudas_demo = [
                {'nombre': 'Tarjeta crédito', 'saldo': 3000, 'interes': 28},
                {'nombre': 'Préstamo personal', 'saldo': 8000, 'interes': 7},
                {'nombre': 'Crédito estudios', 'saldo': 15000, 'interes': 4}
            ]

            ingresos_disponibles = st.number_input("💶 Ingresos disponibles al mes para pagar deudas:", min_value=1)

            if st.button("Simular estrategias de pago"):
                df_nieve = ia_asesor.generar_tabla_orden_pago(deudas_demo, 'bola de nieve')
                df_avalancha = ia_asesor.generar_tabla_orden_pago(deudas_demo, 'avalancha')

                st.write("### Bola de Nieve:")
                st.dataframe(df_nieve)

                st.write("### Avalancha:")
                st.dataframe(df_avalancha)

                rep_nieve, rep_aval, recomendacion = ia_asesor.comparar_estrategias_deuda(deudas_demo, ingresos_disponibles)
                st.markdown(f"### 📄 Reporte Bola de Nieve\n{rep_nieve}")
                st.markdown(f"### 📄 Reporte Avalancha\n{rep_aval}")
                st.markdown(f"### ✅ Recomendación:\n{recomendacion}")
