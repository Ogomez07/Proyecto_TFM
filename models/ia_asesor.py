# models/ia_asesor.py
import numpy as np
import pandas as pd
import os
import openai
from openai import ChatCompletion
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

# Cargar la clave desde el .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def asesor_con_contexto(mensaje_usuario, contexto):
    """ Genera una respuesta usando la IA con el contexto extraído de los datos."""
    prompt = (
        f"Contexto financiero del usuario: \n {contexto}\n\n"
        f"Pregunta del usuario: {mensaje_usuario}\n\n"
        f"Responde de forma clara, empática. Con recomendaciones útiles y personalizadas."
    )

    try:
        respuesta = openai.ChatCompletion.create(
            model ="gpt-3.5-turbo",
            messages= [
                {"role": "system", "content": "Eres un asesor financiero experto en economía y finanzas"},
                {"role": "user", "content": prompt}
            ], 
            temperature=0.7,
            max_tokens=325
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        return f" Error al generar la respuesta: {e}"
    
def gestion_gastos(consulta):
    """ Genera una respuesta usando la IA con el contexto extraído de los datos."""
    prompt = (
        f"Pregunta del usuario: {consulta}\n\n"
        f"Responde de forma clara, con recomendaciones útiles y personalizadas, la mejor forma en la que se pueden repartir los gastos (en proporción), maximizando el ahorro."
    )

    try:
        respuesta = openai.ChatCompletion.create(
            model ="gpt-3.5-turbo",
            messages= [
                {"role": "system", "content": "Eres un asesor financiero experto y empático. Tu objetivo es ayudar a los usuarios a distribuir sus gastos mensuales de forma eficiente, adaptada a sus necesidades reales, maximizando sus ahorros sin afectar su bienestar"},
                {"role": "user", "content": prompt}
            ], 
            temperature=0.7,
            max_tokens=280
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        return f" Error al generar la respuesta: {e}"
    


def plan_ahorro_objetivo(cantidad_objetivo, meses, ingresos, gastos_fijos, usar_contexto=False, contexto_gastos=None):
    """
    Genera un plan de ahorro mensual para alcanzar un objetivo económico.
    Si el usuario lo solicita, la IA sugiere en qué categorías recortar basándose en su historial de gastos.
    """
    ahorro_mensual_necesario = cantidad_objetivo / meses
    dinero_disponible = ingresos - gastos_fijos
    porcentaje_ahorro = (ahorro_mensual_necesario / ingresos) * 100

    mensaje = (
        f" Para alcanzar tu objetivo de ahorrar {cantidad_objetivo:.2f} € en {meses} meses, "
        f"necesitas guardar {ahorro_mensual_necesario:.2f} € al mes.\n"
        f"Esto representa un {porcentaje_ahorro:.1f}% de tus ingresos mensuales.\n\n"
    )

    if dinero_disponible >= ahorro_mensual_necesario:
        mensaje += (
            f" Con tus ingresos actuales ({ingresos:.2f} €) y gastos fijos de {gastos_fijos:.2f} €, "
            f"puedes lograrlo si controlas los gastos variables.\n"
        )
    else:
        diferencia = ahorro_mensual_necesario - dinero_disponible
        mensaje += (
            f" Actualmente solo dispones de {dinero_disponible:.2f} € al mes tras tus gastos fijos.\n"
            f"Te faltarían {diferencia:.2f} € mensuales para alcanzar ese objetivo.\n"
        )

    if usar_contexto and contexto_gastos:
        prompt = (
            f"Contexto financiero del usuario (historial de gastos por categoría):\n{contexto_gastos}\n\n"
            f"Quiere ahorrar {cantidad_objetivo:.2f} € en {meses} meses, lo que implica "
            f"un ahorro mensual de {ahorro_mensual_necesario:.2f} €.\n\n"
            f"Revisa sus gastos históricos y recomienda en qué categorías debería recortar y cuánto, "
            f"para lograr ese ahorro mensual sin afectar demasiado su calidad de vida."
        )

        try:
            respuesta = ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asesor financiero profesional, cercano y práctico. Ayudas al usuario a ahorrar."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            sugerencias = respuesta.choices[0].message.content.strip()
            mensaje += f"\n\n Recomendación personalizada de la IA:\n{sugerencias}"
        except Exception as e:
            mensaje += f"\n\n No se pudo obtener la recomendación automática: {e}"

    elif usar_contexto and not contexto_gastos:
        mensaje += "\n\nHas activado el análisis con IA, pero no se ha proporcionado contexto financiero."

    return mensaje


def predecir_naive_media(serie, fecha_corte='2024-12-31', n_meses=6, meses_pred=4):
    serie = serie.sort_index()
    if not isinstance(serie.index, pd.DatetimeIndex):
        serie.index = pd.to_datetime(serie.index.astype(str)) + pd.offsets.MonthEnd(0)
    else:
        serie.index = serie.index + pd.offsets.MonthEnd(0)

    serie_train = serie[serie.index <= fecha_corte]
    media = serie_train[-n_meses:].mean()

    fechas_futuras = pd.date_range(
        start=pd.to_datetime(fecha_corte) + pd.offsets.MonthEnd(1),
        periods=meses_pred, freq='ME'
    )
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
        print("No hay valores reales disponibles para calcular métricas.")
        return
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true[mask], y_pred[mask])
    print(f"✅ MAE:  {mae:.2f} €")
    print(f"✅ RMSE: {rmse:.2f} €")


def proyeccion_gastos_futuros(categoria, historial_gastos, fecha_corte='2024-12-31', n_meses=6, meses_pred=4):
    """
    Proyecta gasto futuro usando Naïve, muestra resultados y métricas.
    """

    df_categoria = historial_gastos[historial_gastos['categoria'] == categoria].copy()
    serie_mensual = df_categoria.groupby(pd.Grouper(key='fecha', freq='ME'))['importe'].sum()

    serie_train, fechas_futuras, prediccion_media, valores_reales = predecir_naive_media(
        serie_mensual, fecha_corte, n_meses, meses_pred
    )

    predicciones = [prediccion_media] * meses_pred

    # Mostrar resultado visual
    print(f"\n Proyección de gastos en '{categoria}' próximos {meses_pred} meses:")
    resultado_df = mostrar_resultado(fechas_futuras, predicciones, valores_reales)

    # Mostrar métricas
    calcular_metricas(valores_reales, predicciones)

    promedio_prediccion = np.mean(predicciones)
    return (
        f"\n Se estima que gastarás aproximadamente {promedio_prediccion:.2f}€ mensuales "
        f"en '{categoria}' los próximos {meses_pred} meses."
    )


def proyeccion_gastos_totales(historial_gastos, fecha_corte='2024-12-31', n_meses=6, meses_pred=4):
    """
    Proyecta el gasto mensual total de todas las categorías (excepto ingresos y gastos extraordinarios),
    usando el modelo Naïve.
    """

    df_gastos = historial_gastos[
        (historial_gastos['tipo'] != 'ingreso') & 
        (historial_gastos['categoria'].str.lower() != 'gastos extraordinarios')
    ].copy()

    serie_mensual = df_gastos.groupby(pd.Grouper(key='fecha', freq='ME'))['importe'].sum()

    serie_train, fechas_futuras, prediccion_media, valores_reales = predecir_naive_media(
        serie_mensual, fecha_corte, n_meses, meses_pred
    )

    predicciones = [prediccion_media] * meses_pred

    df_resultado = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Predicción (€)': np.round(predicciones, 2),
        'Real (€)': np.round(valores_reales, 2)
    })

    print("\n Proyección de Gastos Totales (sin extraordinarios) próximos meses:")
    display(df_resultado)

    y_true = pd.Series(valores_reales)
    y_pred = pd.Series(predicciones)
    mask = ~y_true.isna()

    if mask.sum() > 0:
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        r2 = r2_score(y_true[mask], y_pred[mask])

        print("\n Métricas del modelo (sin extraordinarios):")
        print(f"✅ MAE:  {mae:.2f} €")
        print(f"✅ RMSE: {rmse:.2f} €")

    else:
        print("No hay suficientes valores reales para calcular métricas.")

    importe_total_proyectado = np.sum(predicciones)

    return (
        f"\nImporte total proyectado (sin extraordinarios) para los próximos {meses_pred} meses: "
        f"{importe_total_proyectado:.2f}€ (promedio mensual: {prediccion_media:.2f}€)."
    )


def recomendacion_emergencia(ingreso_mensual, gastos_fijos, usar_contexto=False, contexto_gastos=None):
    """
    Recomienda un fondo de emergencia adaptado según ingreso y gastos fijos.
    """
    fondo_minimo = gastos_fijos * 3
    fondo_optimo = gastos_fijos * 6
    
    capacidad_ahorro = ingreso_mensual - gastos_fijos

    mensaje_base = (
        f" Fondo mínimo recomendado: {fondo_minimo:.2f}€ (3 meses de gastos fijos).\n"
        f" Fondo óptimo recomendado: {fondo_optimo:.2f}€ (6 meses de gastos fijos).\n\n"
    )
    
    if capacidad_ahorro <= 0:
        mensaje_base += (
            f"Actualmente, con ingresos de {ingreso_mensual:.2f}€ y gastos fijos de {gastos_fijos:.2f}€, "
            "no tienes capacidad de ahorro mensual. Es esencial revisar tus gastos o generar ingresos adicionales "
            "para construir un fondo de emergencia deberías contar con un  mínimo recomendado de al menos "
            f"{fondo_minimo:.2f}€."
        )

    meses_para_minimo = fondo_minimo / capacidad_ahorro
    meses_para_optimo = fondo_optimo / capacidad_ahorro

    mensaje_base += (
        f"\n Para tu seguridad financiera, se recomienda tener un fondo de emergencia mínimo de {fondo_minimo:.2f}€ "
        f"(equivalente a 3 meses de gastos fijos).\n"
        f"El fondo óptimo recomendado sería de {fondo_optimo:.2f}€ (6 meses de gastos).\n\n"
        f"Con tu capacidad actual de ahorro mensual de {capacidad_ahorro:.2f}€, tardarías aproximadamente:\n"
        f"   - {np.ceil(meses_para_minimo)} meses en conseguir el fondo mínimo.\n"
        f"   - {np.ceil(meses_para_optimo)} meses para alcanzar el fondo óptimo.\n\n"
        " Estrategias recomendadas:\n"
        "- Reducir gastos prescindibles.\n"
        "- Destinar ingresos extra puntuales (pagas extras) al fondo.\n"
        "- Generar ingresos adicionales temporales (trabajos freelance, venta de objetos en desuso)."
    )

    if usar_contexto and contexto_gastos:
            prompt = (
                f"El usuario tiene ingresos mensuales de {ingreso_mensual:.2f}€ y gastos fijos mensuales de {gastos_fijos:.2f}€.\n"
                f"Quiere construir un fondo de emergencia mínimo de {fondo_minimo:.2f}€ rápidamente.\n"
                f"Contexto histórico de gastos:\n{contexto_gastos}\n\n"
                f"Sugiere específicamente qué categorías de gasto debería reducir y en qué medida "
                "para acelerar la creación de este fondo de emergencia sin afectar demasiado su calidad de vida."
            )

            try:
                respuesta = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Eres un asesor financiero experto en finanzas personales, cercano y muy práctico."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=400
                )

                sugerencias = respuesta.choices[0].message.content.strip()

                mensaje_base += (
                    "\n **Análisis personalizado según tus gastos históricos:**\n"
                    f"{sugerencias}"
                )

            except Exception as e:
                mensaje_base += f"\n No se pudo obtener recomendación automática: {e}"

    elif usar_contexto and not contexto_gastos:
            mensaje_base += "\n Activaste el contexto histórico, pero no proporcionaste los datos necesarios."

    return mensaje_base


def alerta_gasto_excesivo(historial_gastos, meses_media=3, umbral_alerta=0.3):
    """
    Detecta categorías con incremento significativo de gasto respecto a la media reciente.
    """
    df = historial_gastos.copy()

    df = df[
        (df["tipo"] != "ingreso") &
        (df['categoria'] .str.lower() != 'gastos extraordinarios')]

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["fecha"] = df["fecha"] + pd.offsets.MonthEnd(0)

    resumen = df.groupby([pd.Grouper(key="fecha", freq="ME"), "categoria"])["importe"].sum().reset_index()

    ultimo_mes = resumen["fecha"].max()

    meses_previos = np.sort(resumen["fecha"].unique())
    meses_previos = meses_previos[meses_previos < ultimo_mes][-meses_media:]
    media_anterior = resumen[resumen["fecha"].isin(meses_previos)].groupby("categoria")["importe"].mean()
    gasto_ultimo = resumen[resumen["fecha"] == ultimo_mes].set_index("categoria")["importe"]

    df_alerta = pd.DataFrame({
        "Media_móvil": media_anterior,
        "Gasto_último_mes": gasto_ultimo
    })

    df_alerta.dropna(inplace=True)
    df_alerta["Diferencia_%"] = ((df_alerta["Gasto_último_mes"] - df_alerta["Media_móvil"]) / df_alerta["Media_móvil"]) * 100
    df_alerta["Alerta"] = df_alerta["Diferencia_%"] > (umbral_alerta * 100)

    df_alerta.sort_values("Diferencia_%", ascending=False, inplace=True)

    return df_alerta.reset_index()

def resumen_alerta_gastos(alertas_df):
    """
    Genera un informe textual narrativo de las categorías con gasto excesivo.
    """
    if alertas_df.empty:
        return "Todo en orden. No se han detectado gastos excesivos este mes."

    alertas_activas = alertas_df[alertas_df["Alerta"] == True]

    if alertas_activas.empty:
        return "Todo bajo control. Ninguna categoría ha superado el umbral de gasto excesivo."

    informe = "*Alerta de gasto excesivo detectada en las siguientes categorías:*\n\n"

    for _, row in alertas_activas.iterrows():
        categoria = row["categoria"]
        gasto_actual = row["Gasto_último_mes"]
        media = row["Media_móvil"]
        dif = row["Diferencia_%"]

        informe += (
            f"🔸 **{categoria}**: has gastado {gasto_actual:.2f} €, mientras que la media de los últimos meses era "
            f"{media:.2f} € → incremento de **{dif:.1f}%**.\n"
        )

    informe += "\n Revisa si estos aumentos fueron necesarios o si puedes ajustar tu presupuesto el próximo mes."

    return informe