🧠 AFI – Asesor Financiero Inteligente
Domina tus gastos, predice tu futuro.

Este proyecto combina extracción automática de movimientos bancarios, categorización inteligente, predicción de gastos y asesoría financiera personalizada. Incluye una interfaz interactiva en Streamlit y un entorno de desarrollo para pruebas, exploración y mejora continua.

- Estructura del Proyecto

AFI/
├── app.py                     # Interfaz Streamlit de usuario
├── main.py                   # Script de ejecución principal (modo consola)
├── models/
│   ├── categorizacion.py     # Clasificador de movimientos
│   ├── ia_asesor.py          # Módulo con funciones del asesor financiero
│   └── prediccion.py         # Predicción de gastos por categoría
├── src/
│   ├── eda.py                # Extracción de datos desde PDFs
│   ├── etl.py                # Limpieza y transformación de datos
│   ├── resumen_datos.py      # Creación de contexto financiero
│   └── visualizaciones.py    # Funciones gráficas
├── streamlit_app/
│   ├── assets/               # Recursos visuales para la app
│   └── historial.py          # Funciones de resumen histórico
├── data/                     # Archivos CSV procesados
├── .env                      # Archivo de entorno para API Key
└── requirements.txt          # Dependencias del proyecto


- Cómo Ejecutar
1. Clonar el repositorio

git clone https://github.com/Ogomez07/Proyecto_TFM.git

2. Crear entorno virtual e instalar dependencias

python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
Nota: Crea un archivo .env con tu clave de OpenAI:


- Modo Consola (Análisis paso a paso)
Ejecuta:

python main.py
Esto procesará archivos PDF bancarios, limpiará los datos, clasificará los movimientos, predecirá gastos y generará estrategias de ahorro, deuda o vivienda. Todos los pasos están comentados en el script.

- Modo Interactivo (Interfaz Streamlit)
Ejecuta:

streamlit run app.py
Esto abre una interfaz gráfica con navegación en tres secciones:

Extracción de movimientos: Sube PDFs bancarios y limpia los datos automáticamente.

Predicciones por categorías: Clasifica gastos, descarga CSV y predice valores futuros.

Asesor financiero: Accede a funcionalidades avanzadas como:

Chat IA con contexto financiero.

Recomendación de fondo de emergencia.

Planificador de ahorro.

Detección de gastos excesivos.

Estrategia de pago de deudas (bola de nieve vs avalancha).

Asesoría personalizada sobre vivienda.

- Tecnologías usadas
Python (pandas, scikit-learn, matplotlib)

Streamlit (interfaz web)

OpenAI API (IA generativa para recomendaciones)

Regex y reglas heurísticas (para clasificación)

Modelos Naïve y métricas (para predicción de gastos)

- Ejemplo de predicción
Predicción mensual de gastos en "Restauración" basada en la media de los últimos meses.

📄 Licencia
MIT – Puedes usarlo, modificarlo y distribuirlo con libertad. Dale crédito al autor si lo compartes.