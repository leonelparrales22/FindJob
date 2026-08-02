import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Job Scanner", layout="centered")

st.title("AI Job Scanner")
st.markdown("### Escáner de ofertas de Data Engineering")

# Cargar API key desde .env o ingreso manual
api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key:
    st.session_state["api_key"] = api_key
    st.sidebar.success("Clave cargada de forma segura desde el archivo .env")
else:
    st.sidebar.warning("No se encontró clave en el archivo .env")
    manual_key = st.sidebar.text_input(
        "Ingresa tu API Key de DeepSeek",
        type="password",
    )
    if manual_key:
        st.session_state["api_key"] = manual_key
        st.sidebar.success("Clave ingresada manualmente")

if st.sidebar.button("Iniciar Escaneo"):
    if st.session_state.get("api_key"):
        st.sidebar.info("Escaneo iniciado. Pronto mostraremos resultados.")
    else:
        st.sidebar.error(
            "No hay clave disponible. Cárgala desde el .env o ingrésela manualmente."
        )

# Placeholder para mostrar resultados en fases posteriores
placeholder = st.empty()
with placeholder.container():
    st.info("Aquí se mostrarán los resultados de las ofertas de trabajo en las siguientes fases.")
