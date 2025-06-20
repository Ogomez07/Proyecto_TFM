import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def mostrar_matriz_confusion(y_test, y_pred, labels):
    """Muestra la matriz de confusión como heatmap."""
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()


def mostrar_distribucion_categorias(df, columna='categoria', titulo='Distribución de Categorías'):
    """
    Muestra un gráfico de barras con la distribución de valores en una columna categórica.
    
    Parámetros:
    - df: DataFrame que contiene la columna a analizar.
    - columna: nombre de la columna categórica (por defecto 'categoria').
    - titulo: título opcional del gráfico.
    """
    conteo = df[columna].value_counts()
    conteo.plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')

    plt.title(titulo)
    plt.xlabel(columna.capitalize())
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def mostrar_boxplots_por_categoria(df, columna_categoria='categoria', columna_valor='importe', titulo_base='Boxplot'):
    """
    Muestra un boxplot del valor numérico para cada categoría única.
    """
    categorias = df[columna_categoria].dropna().unique()

    for cat in categorias:
        datos = df[df[columna_categoria] == cat]

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=datos, y=columna_valor)
        plt.title(f'{titulo_base} - {cat}')
        plt.ylabel(columna_valor.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def graficar_predicciones(serie_train, fechas_futuras, reales, pred, categoria, n_meses):
    serie_ext = pd.concat([
        serie_train,
        pd.Series(reales, index=fechas_futuras)
    ])
    plt.figure(figsize=(10, 5))
    plt.plot(serie_ext.index, serie_ext.values, marker='o', color='orange', label='Real')
    plt.plot(fechas_futuras, [pred] * len(fechas_futuras), color='blue', marker='o', label='Predicción')
    plt.title(f"Predicción de gastos - {categoria} (Naïve, media últimos {n_meses} meses)")
    plt.xlabel("Fecha")
    plt.ylabel("€")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_evolucion_deuda(historial_nieve, historial_avalancha):
    """
    Genera un gráfico comparando la evolución de la deuda entre Bola de Nieve y Avalancha.
    """
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(historial_nieve) + 1), historial_nieve, label='Bola de Nieve', marker='o')
    plt.plot(range(1, len(historial_avalancha) + 1), historial_avalancha, label='Avalancha', marker='x')
    plt.xlabel('Meses')
    plt.ylabel('Deuda Total Restante (€)')
    plt.title('Evolución de la Deuda: Bola de Nieve vs Avalancha')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

