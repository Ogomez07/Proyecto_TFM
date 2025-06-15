import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

import src.eda as eda
import src.etl as etl
import src.visualizaciones as viz
import src.resumen_datos as rd
import models.categorizacion as categorizar
import models.prediccion as predecir
import src.resumen_datos as resumen
import models.ia_asesor as ia_asesor 


if __name__ == "__main__":

    # Lista con la ruta de los archivos PDF a procesar.
    pdf_path = [
            "File.pdf",
            "movimientosgenerados_07052025.pdf"
    ]

    # Lista para almacenar los dataframes extraidos
    dataframes = []

    for path in pdf_path:
        print(f'Extrayendo movimientos de: {path}')
    # Ejecuta la extracción de los datos del PDF con la función de detección automática del formato.
        df = eda.extraer_movimientos(path, formato='auto')
        dataframes.append(df)
    
    # Combinar todos los DataFrames
    df_total = pd.concat(dataframes, ignore_index=True)

    # Guardar resultado combinado
    #df_total.to_csv("data/movimientos_combinados.csv", index=False, encoding="utf-8-sig")

    print(f"✔ Se extrajeron {len(df_total)} movimientos combinados.")
    #print(df_total.head())

    # Abrir CSV de los movimientos combinados
    df_fechas = pd.read_csv('data/movimientos_combinados.csv', parse_dates=["Fecha_operacion"])

    #Convertir fechas en formato datetime

    df_fechas = etl.convertir_a_datetime(df_fechas)

    # Agregar manualmente un movimiento 
    df_fechas = etl.agregar_movimiento(df_fechas,'2025-03-31', 'Prestamo', -197.19, None)
    #df_fechas.info()

    # Cambiar los nombres de las columnas
    df_fechas = etl.normalizar_columnas(df_fechas)
    
    # Eliminar palabras de la columna Operacion
    df_fechas['operacion'] = df_fechas['operacion'].apply(etl.eliminar_palabras)


    # Limpiar el texto de la columna operacion
    df_fechas['operacion_limpia'] = df_fechas['operacion'].apply(etl.limpiar_texto)

    # Creación de la columna tipo que separa los datos en función Ingresos de dinero y Gasto
    df_fechas["tipo"] = df_fechas["importe"].apply(lambda i: "ingreso" if i > 0 else "gasto")

    # Añadir columna año_mes
    df_fechas["año_mes"] = df_fechas["fecha_operacion"].dt.to_period("M").astype(str)

    # Eliminar columna saldo
    df_fechas_limpio = df_fechas.drop(columns=["saldo"])

    # Guardamos el CSV limpio
    df_fechas_limpio.to_csv("data/movimientos_limpios.csv", index=False)

    # Abrimos el nuevo CSV 
    df_categorizacion = pd.read_csv('data/movimientos_limpios.csv', parse_dates=['fecha_operacion'])

    # Categorizamos los movimientos
    df_categorizacion['categoria'] = df_categorizacion['operacion_limpia'].apply(categorizar.clasificar_por_reglas)

    #  Preparación del modelo de clasificación
    # Preparar datos
    x_train, x_test, y_train, y_test, x_text_train, x_text_test, vectorizer = categorizar.preparar_datos_modelo(df_categorizacion)

    # Entrenar modelo
    modelo = categorizar.entrenar_modelo_clasificador(x_train, y_train)

    # Evaluación del modelo
    evaluacion = categorizar.evaluar_modelo(modelo, x_test, y_test, x_text_test)
    print(evaluacion)

    # Filtrar las operaciones sin etiquetar
    df_sin_etiquetar = categorizar.filtrar_movimientos_sin_categoria(df_categorizacion)

    # Predecir categorias faltantes
    df_sin_etiquetar = categorizar.predecir_categorias(df_sin_etiquetar, vectorizer, modelo)

    # Vizualizar resultados
    print(f' Movimientos sin etiquetar: {df_sin_etiquetar.shape[0]}')
    display(df_sin_etiquetar[['operacion_limpia', 'categoria_predicha']].sample(15))

    # Actualizamos las categorías en el Dataframe original
    df_actualizado = categorizar.actualizar_categorias(df_categorizacion, df_sin_etiquetar)
    print(f" Aún sin categorizar: {df_actualizado[df_actualizado['categoria'] == 'Sin categorizar'].shape[0]}")
    print(df_actualizado.info())

    # Visualizamos el reparto de categorías
    viz.mostrar_distribucion_categorias(df_actualizado)

    # Vemos como se comportan sus importes por categoría con un Boxplot
    viz.mostrar_boxplots_por_categoria(df_actualizado)

    # Limpiamos lo valores atípicos de cada categoría y los agrupamos en una categoría nueva llamada "Gastos extras"
    df_actualizado = etl.mover_outliers_a_gastos_extra(df_actualizado)
    display(df_actualizado['categoria'].value_counts())

    # Eliminamos el valor 0 de préstamo (puede generar problemas en las predicciones)
    df_actualizado = etl.eliminar_outliers_prestamo(df_actualizado)

    # Pasamos los valores a positivo
    df_actualizado['importe'] = df_actualizado['importe'].abs()

    # Revisamos como se encuentra cada categoría con un boxplot
    viz.mostrar_boxplots_por_categoria(df_actualizado)

    # Guardar el CSV
    df_actualizado.to_csv("data/Movimientos_categorizados.csv", index=False)

    # Abrir el CSV de movimientos categorizados
    df_prediccion =pd.read_csv('data/Movimientos_categorizados.csv', parse_dates=['fecha_operacion'])

    # Seleccionamos el tipo de movimiento que queremos predecir
    df_prediccion = df_prediccion[df_prediccion['tipo'] == 'gasto'].copy()
    gastos_mensuales = df_prediccion.groupby(['año_mes', 'categoria'])['importe'].sum().unstack(fill_value=0)
    display(gastos_mensuales)

    # Predecir el gasto mensual por categoría
    categoria ='Préstamo'
    serie = gastos_mensuales[categoria]
    serie_train, fechas, pred, reales = predecir.predecir_naive_media(serie)
    print(f"Predicción para la categoría {categoria}: {pred:.2f} €")
    df_resultado = predecir.mostrar_resultado(fechas, [pred]*len(fechas), reales)
    #viz.graficar_predicciones(serie_train, fechas, reales, pred, 'Restauración', 6)
    predecir.calcular_metricas(reales, [pred]*len(fechas))


    # Prueba API ok
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ No se ha cargado la clave API.')
    else:
        print('✅ Clave API cargada correctamente.')

    # # Generamos la ruta a seguir del asesor a los datos
    ruta_csv = 'data/Movimientos_categorizados.csv'

    # Funciones de asesor financiero
    """Funcion 1"""

    # Generamos resumen de los datos
    contexto = resumen.resumir_movimientos(ruta_csv)

    # Pregunta del usuario
    print("¿Qué deseas consultar?: ")
    pregunta = input("> ")

    # Preparar respuesta de la IA
    respuesta = ia_asesor.asesor_con_contexto(pregunta, contexto)

    # Respuesta generada
    print("\n Respuesta: \n")
    print(respuesta)


    """Función 2"""
    print(ia_asesor.plan_ahorro_objetivo(5000, 10, 1300, 600, usar_contexto=True, contexto_gastos=contexto))

    """Función 3"""
    # Cargar CSV de movimientos
    df_movimientos = pd.read_csv("data/Movimientos_categorizados.csv")
    df_movimientos['fecha'] = pd.to_datetime(df_movimientos['fecha_operacion'])
    respuesta_3 = ia_asesor.proyeccion_gastos_futuros("Restauración", df_movimientos)

    """Función 4 """
    print(ia_asesor.proyeccion_gastos_totales(df_movimientos))


    """Fucnión 5"""
    df_contexto = pd.read_csv("data/Movimientos_categorizados.csv", parse_dates=["fecha_operacion"])

    print(ia_asesor.recomendacion_emergencia(1300, 500, usar_contexto=True, contexto_gastos=df_contexto))

    """Función 6"""
    alerta_df = ia_asesor.alerta_gasto_excesivo(df_movimientos)
    informe = ia_asesor.resumen_alerta_gastos(alerta_df)
    print(informe)
    
    """Función 7 (Gestión de deudas)"""
    # Valores de ejemplo para hacer significativas las diferencias entre modelos
    deudas = [
        {'nombre': 'Micropréstamo', 'saldo': 5000, 'interes': 35},    # ahora más saldo pero poco interés
        {'nombre': 'Tarjeta Crédito', 'saldo': 3000, 'interes': 28},
        {'nombre': 'Préstamo personal', 'saldo': 8000, 'interes': 7},
        {'nombre': 'Crédito estudios', 'saldo': 15000, 'interes': 4},
        {'nombre': 'Préstamo consumo', 'saldo': 1000, 'interes': 5},   # pequeño pero alto interés
    ]


    ingresos_disponibles = 500
    
    # Generar DataFrames de orden de pago
    df_nieve = ia_asesor.generar_tabla_orden_pago(deudas, 'bola de nieve')
    df_avalancha = ia_asesor.generar_tabla_orden_pago(deudas, 'avalancha')

    # Comparar estrategias
    reporte_nieve, reporte_avalancha, recomendacion = ia_asesor.comparar_estrategias_deuda(deudas, ingresos_disponibles)
    
    # Nueva simulación con historial para gráficos
    intereses_nieve, meses_nieve, historial_nieve, intereses_avalancha, meses_avalancha, historial_avalancha = ia_asesor.simular_estrategias(deudas, ingresos_disponibles)


    # Mostrar reporte Bola de Nieve
    print(" Estrategia: Bola de Nieve ")
    print(df_nieve.to_string(index=False))
    print(reporte_nieve)

    # Mostrar reporte Avalancha
    print(" Estrategia: Avalancha ")
    print(df_avalancha.to_string(index=False))
    print(reporte_avalancha)

    # Mostrar recomendación final
    print(" Recomendación Final ")
    print(recomendacion)
    # Grafico evolucion deudas
    viz.graficar_evolucion_deuda(historial_nieve, historial_avalancha)

    """Funcion 8 (Asesor vivienda)"""
    # Cargando contexto financiero real
    contexto = etl.preparar_contexto(ruta_csv)

    # Asesoría normal basada en lógica
    print(ia_asesor.asesoria_vivienda("compra", usar_contexto=True, contexto_gastos=contexto))

    # Asesoría avanzada con ChatGPT
    print(ia_asesor.asesoria_vivienda("compra", usar_contexto=True, contexto_gastos=contexto, usar_chatgpt=True))







    
