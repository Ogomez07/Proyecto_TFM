# models/ia_asesor.py
import numpy as np
import pandas as pd
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

# Cargar la clave desde el .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def asesor_con_contexto(mensaje_usuario, contexto):
    """Genera una respuesta usando IA con el contexto proporcionado."""
    prompt = (
        f"Contexto financiero del usuario:\n{contexto}\n\n"
        f"Pregunta del usuario:\n{mensaje_usuario}\n\n"
        f"Responde de forma clara, empática y con recomendaciones personalizadas."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asesor financiero experto y empático."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error al generar la respuesta de IA: {e}"
    
def gestion_gastos(consulta):
    """Ofrece distribución de gastos en proporción, maximizando el ahorro."""
    prompt = (
        f"Consulta del usuario:\n{consulta}\n\n"
        f"Ofrece una estrategia clara, útil y bien argumentada para repartir los gastos y ahorrar más."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto financiero especializado en presupuestos personales, debes dar valores numéricos en caso de indicar que debe reducir alguna categoría."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=320
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error al generar respuesta de IA: {e}"


def plan_ahorro_objetivo(cantidad_objetivo, meses, ingresos, gastos_fijos, usar_contexto=False, contexto_gastos=None):
    """Calcula cuánto debe ahorrar al mes un usuario para lograr una meta económica."""
    ahorro_mensual = cantidad_objetivo / meses
    disponible = ingresos - gastos_fijos
    porcentaje_ahorro = (ahorro_mensual / ingresos) * 100 if ingresos else 0

    mensaje = (
        f"💰 Para ahorrar {cantidad_objetivo:.2f}€ en {meses} meses necesitas ahorrar {ahorro_mensual:.2f}€/mes.\n"
        f"Eso representa el {porcentaje_ahorro:.1f}% de tus ingresos mensuales.\n"
    )

    if disponible >= ahorro_mensual:
        mensaje += f"✅ Puedes lograrlo, ya que dispones de {disponible:.2f}€/mes tras tus gastos fijos.\n"
    else:
        mensaje += f"⚠️ Te faltarían {ahorro_mensual - disponible:.2f}€ mensuales para lograrlo con tus ingresos actuales.\n"

    if usar_contexto and contexto_gastos:
        contexto_texto = (
            f"Tus ingresos son {ingresos}€, tus gastos fijos {gastos_fijos}€ y quieres ahorrar {ahorro_mensual:.2f}€/mes.\n"
            f"Este es tu contexto de gastos:\n{contexto_gastos}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asesor financiero que recomienda cómo ahorrar más dinero."},
                    {"role": "user", "content": f"{contexto_texto}\n\nIndica en qué partidas podría recortar para conseguir su objetivo sin afectar mucho su calidad de vida."}
                ],
                temperature=0.7,
                max_tokens=400
            )
            mensaje += "\n📌 Recomendación de IA:\n" + response.choices[0].message.content.strip()
        except Exception as e:
            mensaje += f"\n⚠️ No se pudo obtener recomendación de IA: {e}"

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
        f"\n Se estima que tendrás unos ingresos aproximados de: {promedio_prediccion:.2f}€ mensuales "
        f"en '{categoria}' los próximos {meses_pred} meses."
        if categoria.lower() == 'ingreso'
        else
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

def ordenar_deudas(deudas, estrategia):
    """
    Ordena las deudas según la estrategia seleccionada.
    """
    if estrategia == 'bola de nieve':
        return sorted(deudas, key=lambda d: d['saldo'])
    elif estrategia == 'avalancha':
        return sorted(deudas, key=lambda d: d['interes'], reverse=True)
    else:
        raise ValueError("Estrategia no válida. Usa 'bola de nieve' o 'avalancha'.")

def simular_pago_dinamico(deudas, ingreso_mensual, estrategia):
    """
    Simula el pago dinámico de deudas, devolviendo además la evolución del saldo total mes a mes.
    """
    plan = ordenar_deudas(deudas, estrategia)
    total_intereses = 0
    meses = 0
    historial_saldo_total = []

    while plan:
        dinero_mes = ingreso_mensual

        for deuda in plan:
            interes_mensual = deuda['interes'] / 100 / 12
            saldo = deuda['saldo']

            interes_mes = saldo * interes_mensual
            pago_aplicado = min(dinero_mes, saldo + interes_mes)

            deuda['saldo'] = saldo + interes_mes - pago_aplicado
            total_intereses += interes_mes
            dinero_mes -= pago_aplicado

        # Guardar el saldo total al final del mes
        saldo_total_mes = sum(d['saldo'] for d in plan)
        historial_saldo_total.append(saldo_total_mes)

        # Eliminar deudas pagadas
        plan = [d for d in plan if d['saldo'] > 0]
        meses += 1

    return total_intereses, meses, historial_saldo_total



def generar_reporte_asesoramiento(intereses_totales, meses_totales, estrategia):
    """
    Genera un resumen del plan para reducir las deudas.
    """
    if estrategia == 'bola de nieve':
        estrategia_texto = ("La estrategia utilizada es **Bola de Nieve**, priorizando pagar las deudas con menor saldo "
                            "para que sientas avances rápidos y aumentes tu motivación.")
    elif estrategia == 'avalancha':
        estrategia_texto = ("La estrategia utilizada es **Avalancha**, enfocándote en pagar primero las deudas con "
                            "mayor interés para ahorrar más dinero en intereses a largo plazo.")
    else:
        estrategia_texto = "La estrategia utilizada no está definida claramente."

    reporte = (
        f"{estrategia_texto}\n\n"
        f"- Tiempo estimado total para liberarte de todas las deudas: {meses_totales} meses.\n"
        f"- Intereses totales estimados que pagarás: {round(intereses_totales, 2)} €.\n\n"
        "*Consejo*: Considera no adquirir nuevas deudas durante este proceso y, si puedes, incrementar los pagos mensuales "
        "cuando sea posible para acortar los plazos."
    )

    return reporte


def comparar_intereses(intereses_nieve, intereses_avalancha):
    """
    Compara los intereses pagados entre las dos estrategias y da una recomendación.
    """
    if intereses_avalancha < intereses_nieve:
        return (
            f" Recomendación: Te conviene utilizar la estrategia **Avalancha**, ya que pagarías menos intereses "
            f"({intereses_avalancha:.2f} € vs {intereses_nieve:.2f} €). Esta estrategia es ideal si quieres minimizar el coste total."
        )
    else:
        return (
            f" Recomendación: Te conviene utilizar la estrategia **Bola de Nieve**, aunque pagarás un poco más en intereses "
            f"({intereses_nieve:.2f} € vs {intereses_avalancha:.2f} €). Es ideal si prefieres mantener alta tu motivación eliminando deudas rápidamente."
        )

def comparar_estrategias_deuda(deudas, ingresos_disponibles):
    """
    Compara las estrategias Bola de Nieve y Avalancha, generando reportes y recomendación.
    """
    # IMPORTANTE: Copias profundas para que no se modifiquen las deudas originales
    deudas_nieve = [deuda.copy() for deuda in deudas]
    deudas_avalancha = [deuda.copy() for deuda in deudas]

    intereses_nieve, meses_nieve, _ = simular_pago_dinamico(deudas_nieve, ingresos_disponibles, 'bola de nieve')
    intereses_avalancha, meses_avalancha, _ = simular_pago_dinamico(deudas_avalancha, ingresos_disponibles, 'avalancha')


    reporte_nieve = generar_reporte_asesoramiento(intereses_nieve, meses_nieve, 'bola de nieve')
    reporte_avalancha = generar_reporte_asesoramiento(intereses_avalancha, meses_avalancha, 'avalancha')

    recomendacion = comparar_intereses(intereses_nieve, intereses_avalancha)

    return reporte_nieve, reporte_avalancha, recomendacion

def generar_tabla_orden_pago(deudas, estrategia):
    """
    Genera una tabla con el orden de pago de las deudas según la estrategia.
    """
    # Ordenamos las deudas
    deudas_ordenadas = ordenar_deudas(deudas, estrategia)
    
    # Creamos un DataFrame
    df = pd.DataFrame(deudas_ordenadas)
    df = df[['nombre', 'saldo', 'interes']]  # Solo columnas que nos interesan
    df.columns = ['Deuda', 'Saldo Inicial (€)', 'Interés Anual (%)']

    # Añadimos columna de orden de pago
    df.insert(0, 'Orden de Pago', range(1, len(df) + 1))

    return df

def simular_estrategias(deudas, ingreso_mensual):
    """
    Simula las dos estrategias (bola de nieve y avalancha) y devuelve intereses, meses y historial de cada una.
    """
    # Copiamos deudas para cada simulación
    deudas_nieve = [deuda.copy() for deuda in deudas]
    deudas_avalancha = [deuda.copy() for deuda in deudas]

    intereses_nieve, meses_nieve, historial_nieve = simular_pago_dinamico(deudas_nieve, ingreso_mensual, 'bola de nieve')
    intereses_avalancha, meses_avalancha, historial_avalancha = simular_pago_dinamico(deudas_avalancha, ingreso_mensual, 'avalancha')

    return intereses_nieve, meses_nieve, historial_nieve, intereses_avalancha, meses_avalancha, historial_avalancha





# Función para procesar contexto
def procesar_contexto(contexto_gastos):
    ingreso = contexto_gastos.get('ingresos_mensuales', 0)
    gastos_totales = sum(contexto_gastos.get('gastos', {}).values())
    ahorro_estimado = ingreso - gastos_totales
    gasto_alquiler = contexto_gastos.get('gastos', {}).get('Alquiler', 0)
    return {
        'ingreso': ingreso,
        'gastos_totales': gastos_totales,
        'ahorro_estimado': ahorro_estimado,
        'gasto_alquiler': gasto_alquiler
    }


# Funciones auxiliares para lógica manual
def generar_asesoria_compra(datos):
    cuota_maxima = datos['ingreso'] * 0.35
    if datos['ahorro_estimado'] > cuota_maxima:
        return (
            f"Comprar podría ser viable.\n"
            f" - Máxima cuota hipotecaria recomendada: {cuota_maxima:.2f} €/mes.\n"
            f" - Asegúrate de contar con al menos el 20% del precio de la vivienda como entrada.\n"
            f" - Tu capacidad de ahorro es suficiente para asumir los gastos iniciales y afrontar imprevistos."
        )
    else:
        return (
            "Actualmente tu capacidad de ahorro no parece suficiente para afrontar una hipoteca de forma segura.\n"
            "Se recomienda mejorar tu ahorro mensual antes de plantearte comprar vivienda."
        )


def generar_asesoria_alquiler(datos):
    if datos['gasto_alquiler'] > 0:
        porcentaje_alquiler = datos['gasto_alquiler'] / datos['ingreso']
        if porcentaje_alquiler <= 0.3:
            return (
                f"Tu gasto en alquiler representa un {porcentaje_alquiler*100:.1f}% de tus ingresos, "
                "lo cual es un valor saludable.\n"
                "Seguir de alquiler parece una opción adecuada en tu situación actual."
            )
        else:
            return (
                f"Tu gasto en alquiler representa un {porcentaje_alquiler*100:.1f}% de tus ingresos, "
                "lo cual es elevado.\n"
                "Considera renegociar el alquiler, mudarte o valorar una compra si es viable."
            )
    else:
        return (
            "No se detecta gasto en alquiler actualmente.\n"
            "¿Seguro que deseas evaluar continuar alquilado? Puede que no estés pagando alquiler ahora mismo."
        )


def generar_asesoria_independizar(datos):
    if datos['gasto_alquiler'] > 0:
        return (
            "Actualmente ya tienes un gasto de alquiler. Pareces estar independizado.\n"
            "Revisa si tu gasto es adecuado o si puedes optimizarlo."
        )
    else:
        alquiler_recomendado = datos['ingreso'] * 0.3
        if datos['ahorro_estimado'] > 300:
            return (
                f"Actualmente vives sin gasto de alquiler.\n"
                f"Podrías permitirte un alquiler de hasta {alquiler_recomendado:.2f} € al mes "
                "manteniendo un margen sano.\n"
                "Si decides independizarte, asegúrate de considerar los gastos extra: fianza, mudanza, muebles."
            )
        else:
            return (
                "Actualmente tu ahorro mensual es bajo.\n"
                "Independizarte podría comprometer tu estabilidad financiera.\n"
                "Recomendación: aumenta tu ahorro mensual antes de dar el paso."
            )


# Función para generar asesoría manual
def generar_asesoria_manual(opcion, datos):
    resumen = (
        f"Ingreso mensual promedio: {datos['ingreso']:.2f} €\n"
        f"Gastos mensuales aproximados: {datos['gastos_totales']:.2f} €\n"
        f"Ahorro mensual estimado: {datos['ahorro_estimado']:.2f} €\n\n"
    )
    if opcion == "compra":
        resumen += generar_asesoria_compra(datos)
    elif opcion == "alquiler":
        resumen += generar_asesoria_alquiler(datos)
    elif opcion in ["independizarme", "independizar"]:
        resumen += generar_asesoria_independizar(datos)
    return resumen


# Función para generar asesoría usando ChatGPT
def generar_asesoria_chatgpt(opcion, datos):
    prompt_usuario = f"""
Eres un asesor financiero experto.
Basándote en estos datos reales de un usuario:
- Ingreso mensual: {datos['ingreso']:.2f} €
- Ahorro mensual estimado: {datos['ahorro_estimado']:.2f} €
- Gasto mensual en alquiler (si existe): {datos['gasto_alquiler']:.2f} €
- Opción deseada: {opcion}

Asesora de forma profesional y razonada si debería comprar vivienda, seguir alquilado o si puede independizarse.
Sé claro, sencillo, razonado y adapta la recomendación a la situación financiera.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Asistente financiero experto en asesoría de compra y alquiler de vivienda."},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error al generar asesoría con ChatGPT: {str(e)}"


# Función principal
def asesoria_vivienda(opcion, usar_contexto=True, contexto_gastos=None, usar_chatgpt=False):
    """
    Aconseja si conviene comprar, seguir de alquiler o independizarse usando datos reales o generando la asesoría con ChatGPT.
    """
    opciones_validas = ["compra", "alquiler", "independizarme", "independizar"]

    if opcion.lower() not in opciones_validas:
        return "Error: opción no válida. Debes elegir entre: 'compra', 'alquiler' o 'independizarme'."

    if not usar_contexto or contexto_gastos is None:
        return "Actualmente esta función requiere usar contexto financiero real."

    datos = procesar_contexto(contexto_gastos)

    if usar_chatgpt:
        return generar_asesoria_chatgpt(opcion.lower(), datos)
    else:
        return generar_asesoria_manual(opcion.lower(), datos)









