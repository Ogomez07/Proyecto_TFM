from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import src.visualizaciones as viz

# Creamos funcion para crear las categorias y entrenar el modelo de categorización
def clasificar_por_reglas(texto):
    texto = str(texto).lower()

    # Ingresos
    if any(p in texto for p in ['transferencia inmediata de', 'transferencia de', 'transferencia recibida', 'reintegro']):
        return 'Ingreso'
    if 'bizum de' in texto:
        return 'Ingreso'

    # Cuentas pagadas / personales
    if 'bizum a' in texto or 'transferencia a' in texto or 'traspaso' in texto:
        return 'Transferencia personal'
    
    # Facturas
    if 'recibo' in texto or 'academia' in texto or 'univ' in texto:
        return 'Facturas'

    # Préstamos
    if 'liquidacion periodica' in texto:
        return 'Préstamo'
    
    # Compras no esenciales
    compras_no_esenciales = [
        'mediamarkt', 'media markt', 'fnac','ikea', 'leroy merlin', 'decathlon',
        'zara', 'shein', 'aliexpress', 'pull&bear', 'stradivarius', 'amazon', 'playstation',
        'viveros', 'zara home', 'idea market', 'playstation', 'tabaco', 'kiwoko',
        'primor', 'agroquimica', 'druni','kiwoko'
    ]
    if any(c in texto for c in compras_no_esenciales):
        return 'Compras no esenciales'

    # Restauración
    restaurantes = [
        'la sureña', 'goiko', 'vips', 'heladeria','pizza', 'hamburguesa','heladería', 'picoteo',
        'mcdonald', 'mc donald', 'burger king', 'burguer king', 'los charcones',
        'quinta cumbre', 'bk', 'camaleon', 'tagliatella', 'bar', 'cafeteria',
        'cafetería', 'food', 'tb metropo', 'compra uber', "llaollao", "compra sumup",
        'pasteleria', 'poke','las casitas', 'gelato', 'cantina', 'nothing', 'oakberry', 'restaurante'
    ]
    if any(r in texto for r in restaurantes):
        return 'Restauración'
    
    # Veterinario
    if 'el corral' in texto:
        return 'Facturas'

    # Supermercado
    supermercado = ['mercadona', 'aldi', 'lidl', 'hiperdino', 'carrefour', 'alcampo', 'farmacia']
    if any(s in texto for s in supermercado):
        return 'Supermercado'

    # Suscripciones / ocio
    ocio = ['netflix', 'spotify', 'hbo', 'disney+', 'disney plus', 'amazon prime', 'filmin', 'nintendo', 'game pass', 'cine', 'artesiete', 'apple com']
    if any(o in texto for o in ocio):
        return 'Ocio / Suscripciones'

    
    # Transporte público
    if any(p in texto for p in ['guaguas', 'salcai', 'global']):
        return 'Transporte'

    # Transporte privado / gasolina
    gasolina = ['repsol', 'cepsa', 'bp', 'shell', 'gasolinera', 'disa', 'parking', 'motor telde', 'coche', 'rueda', 'gasolina', 'petroprix']
    if any(g in texto for g in gasolina):
        return 'Transporte'


    return 'Sin categorizar'

def preparar_datos_modelo(df, columna_texto='operacion_limpia', columna_etiqueta='categoria', max_features=1000):
    """
    Prepara los datos para entrenamiento: vectoriza texto, separa etiquetas y texto original.
    """
    
    df_etiquetado = df[df[columna_etiqueta] != 'Sin categorizar'].copy()
    x_texto = df_etiquetado[columna_texto]
    y_etiqueta = df_etiquetado[columna_etiqueta]
    textos_originales = x_texto.tolist()

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_vect = vectorizer.fit_transform(x_texto)

    x_train, x_test, y_train, y_test, x_text_train, x_text_test = train_test_split(
        X_vect, y_etiqueta, textos_originales, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test, x_text_train, x_text_test, vectorizer

def entrenar_modelo_clasificador(x_train, y_train):
    """
    Entrena un modelo RandomForest y lo devuelve.
    """
    modelo = RandomForestClassifier()
    modelo.fit(x_train, y_train)
    return modelo



def obtener_predicciones(modelo, x_test):
    """Devuelve las predicciones del modelo."""
    return modelo.predict(x_test)

def mostrar_metricas(y_test, y_pred):
    """Imprime accuracy y reporte de clasificación."""
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n Reporte de clasificación:")
    print(classification_report(y_test, y_pred))


def obtener_errores(x_text_test, y_test, y_pred):
    """Devuelve un DataFrame con los errores de predicción."""
    errores = pd.DataFrame({
        'texto': x_text_test,
        'real': y_test.values,
        'predicho': y_pred
    })
    return errores[errores['real'] != errores['predicho']]


def evaluar_modelo(modelo, x_test, y_test, x_text_test=None, mostrar_errores=True):
    """
    Evalúa un modelo clasificando y visualiza métricas.
    Devuelve errores si `mostrar_errores=True` y `x_text_test` no es None.
    """
    y_pred = obtener_predicciones(modelo, x_test)
    mostrar_metricas(y_test, y_pred)
    viz.mostrar_matriz_confusion(y_test, y_pred, modelo.classes_)

    if x_text_test is not None and mostrar_errores:
        errores = obtener_errores(x_text_test, y_test, y_pred)
        print("\n❌ Errores de clasificación:")
        display(errores.head(15))
        return errores

    return None

def filtrar_movimientos_sin_categoria(df, columna_etiqueta="categoria"):
    """
    Devuelve un DataFrame con los movimientos sin categoría.
    """
    return df[df[columna_etiqueta] == "Sin categorizar"].copy()


def predecir_categorias(df_sin_etiquetar, vectorizer, modelo, columna_texto="operacion_limpia"):
    """
    Usa un modelo entrenado y un vectorizador para predecir categorías de movimientos sin etiquetar.
    Añade la columna 'categoria_predicha'.
    """
    x_nuevos = vectorizer.transform(df_sin_etiquetar[columna_texto])
    df_sin_etiquetar["categoria_predicha"] = modelo.predict(x_nuevos)
    return df_sin_etiquetar

def actualizar_categorias(df_original, df_predicho, columna_etiqueta='categoria'):
    """
    Actualiza el DataFrame original con las categorías predichas,
    y marca el origen de la categoría ('manual' o 'modelo').

    - df_original: DataFrame con todas las operaciones.
    - df_predicho: DataFrame con operaciones 'Sin categorizar' ya predichas.
    """
    # Actualizar las categorías
    df_original.loc[df_predicho.index, columna_etiqueta] = df_predicho['categoria_predicha'].values

    # Añadir columna de origen
    df_original['origen'] = 'manual'
    df_original.loc[df_predicho.index, 'origen'] = 'modelo'

    return df_original

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





