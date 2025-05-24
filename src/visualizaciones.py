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

    Parámetros:
    - df: DataFrame con los datos.
    - columna_categoria: nombre de la columna que contiene las categorías.
    - columna_valor: nombre de la columna con valores numéricos (e.g. importe).
    - titulo_base: prefijo para el título de cada gráfico.
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


