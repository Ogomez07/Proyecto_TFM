import pandas as pd
import matplotlib.pyplot as plt


def graficar_predicciones(serie_train, fechas_futuras, reales, pred, categoria, n_meses):
    serie_ext = pd.concat([
        serie_train,
        pd.Series(reales, index=fechas_futuras)
    ])

    # Crear figure y axis explícitamente
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(serie_ext.index, serie_ext.values, marker='o', color='orange', label='Real')
    ax.plot(fechas_futuras, [pred] * len(fechas_futuras), color='blue', marker='o', label='Predicción')
    ax.set_title(f"Predicción de gastos - {categoria} (Naïve, media últimos {n_meses} meses)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("€")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig