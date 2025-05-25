# models/ia_asesor.py
import os
import openai
from dotenv import load_dotenv

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