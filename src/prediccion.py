import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

def predecir_naive_media(serie, fecha_corte='2024-12-31', n_meses=6, meses_pred=4):
    """
    Calcula predicción Naïve para una serie temporal.
    """
    serie = serie.sort_index()
    if not isinstance(serie.index, pd.DatetimeIndex):
        serie.index = pd.to_datetime(serie.index.astype(str)) + pd.offsets.MonthEnd(0)
    else:
        serie.index = serie.index + pd.offsets.MonthEnd(0)

    serie_train = serie[serie.index <= fecha_corte]
    media = serie_train[-n_meses:].mean()

    fechas_futuras = pd.date_range(start=pd.to_datetime(fecha_corte) + pd.offsets.MonthEnd(1),
                                    periods=meses_pred, freq='M')
    valores_reales = [serie.get(fecha, np.nan) for fecha in fechas_futuras]

    return serie_train, fechas_futuras, media, valores_reales

def mostrar_resultado(fechas, predicciones, reales):
    df_resultado = pd.DataFrame({
        'Fecha': fechas,
        'Predicción (€)': np.round(predicciones, 2),
        'Real (€)': np.round(reales, 2),
    })
    display(df_resultado)
    return df_resultado



def calcular_metricas(reales, predicciones):
    y_true = pd.Series(reales)
    y_pred = pd.Series(predicciones)
    mask = ~y_true.isna()
    if mask.sum() == 0:
        print("⚠️ No hay valores reales disponibles para calcular métricas.")
        return
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true[mask], y_pred[mask])
    print(f"✅ MAE:  {mae:.2f} €")
    print(f"✅ RMSE: {rmse:.2f} €")
    print(f"✅ MAPE: {mape:.2f} %")
    print(f"✅ R²:   {r2:.2f}")

