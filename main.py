import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt


import src.eda as eda
import src.etl as etl
import src.categorizacion as categorizar
import src.prediccion as pred
import src.visualizaciones as viz


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
    df_total.to_csv("data/movimientos_combinados.csv", index=False, encoding="utf-8-sig")

    print(f"✔ Se extrajeron {len(df_total)} movimientos combinados.")
    print(df_total.head())

    # Abrir CSV de los movimientos combinados
    df_fechas = pd.read_csv('data/movimientos_combinados.csv', parse_dates=["Fecha_operacion"])

    #Convertir fechas en formato datetime

    df_fechas = etl.convertir_a_datetime(df_fechas)

    # Agregar manualmente un movimiento 
    df_fechas = etl.agregar_movimiento(df_fechas,'2025-03-31', 'Prestamo', -197.19, None)
    df_fechas.info()

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

 
