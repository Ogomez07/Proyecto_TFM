import pandas as pd
from IPython.display import display


import src.eda as eda
import src.etl as etl
import src.entrenamiento as entrenamiento

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

    # Procesar las fechas del Dataset
    df_fechas = etl.convertir_a_datetime(df_fechas)

    # Agregar manualmente un movimiento 
    df_fechas = etl.agregar_movimiento(df_fechas,'2025-03-31', 'Prestamo', 197.19, None)
    df_fechas.info()
    display(df_fechas.sample(25))

    display(df_fechas["Fecha_operacion"].isna().sum())

 
