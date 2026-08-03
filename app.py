import io
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import filter_new_jobs, get_all_jobs, inicializar_db, save_jobs
from evaluator import evaluar_lote_ofertas, evaluar_oferta
from scraper import obtener_empleos_getonboard

load_dotenv()
inicializar_db()

ROLES_KEYWORDS = [
    "ingeniero de datos",
    "arquitecto de datos",
    "ingeniero de inteligencia artificial",
    "ingeniero ia",
    "data engineer",
    "data architect",
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "analytics engineer",
    "datos",
    "data",
]

STACK_KEYWORDS = [
    "python",
    "sql",
    "spark",
    "databricks",
    "aws",
    "azure",
    "llm",
    "rag",
    "mcp",
    ".net",
    "etl",
    "el",
    "big data",
    "data warehouse",
    "lakehouse",
]

DESCARTAR_KEYWORDS = [
    "junior",
    "trainee",
    "intern",
    "pasante",
    "becario",
    "entry level",
    "practicante",
    "pasantía",
    "english required",
    "advanced english",
    "fluent english",
    "english proficiency",
    "proficiency in english",
    "native english",
    "english speaker",
    "inglés requerido",
    "inglés avanzado",
    "inglés fluido",
    "inglés nativo",
]

LOTE_TAMANO = 5


def _quitar_acentos(texto: str) -> str:
    """Convierte a minúsculas y elimina acentos."""
    if not texto:
        return ""
    texto = texto.lower()
    vocales = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunaeiouun")
    return texto.translate(vocales)


def _es_ubicacion_valida(modalidad: str, ubicacion: str) -> bool:
    """Acepta remoto siempre; presencial o híbrido solo si es en Quito o Ecuador."""
    modalidad_limpia = _quitar_acentos(modalidad or "")
    ubicacion_limpia = _quitar_acentos(ubicacion or "")
    if "remoto" in modalidad_limpia or "remote" in modalidad_limpia:
        return True
    if any(u in ubicacion_limpia for u in ["quito", "ecuador"]):
        return True
    # Si no hay ubicación clara, dejar pasar para que DeepSeek decida
    return not ubicacion_limpia


def _cumple_prefiltro(row) -> bool:
    """Verifica que la oferta tenga al menos un rol, un stack, no requiera inglés y sea ubicación válida."""
    texto = _quitar_acentos(
        f"{row.get('titulo', '')} {row.get('descripcion', '')} {row.get('empresa', '')} {row.get('ubicacion', '')}"
    )
    tiene_rol = any(kw in texto for kw in ROLES_KEYWORDS)
    tiene_stack = any(kw in texto for kw in STACK_KEYWORDS)
    descartar = any(kw in texto for kw in DESCARTAR_KEYWORDS)
    ubicacion_ok = _es_ubicacion_valida(row.get("modalidad", ""), row.get("ubicacion", ""))
    return tiene_rol and tiene_stack and not descartar and ubicacion_ok


st.set_page_config(page_title="AI Job Scanner", layout="wide")

st.title("AI Job Scanner")
st.markdown("### Escáner multi-fuente de ofertas de Datos e IA")

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

placeholder = st.empty()

if st.sidebar.button("Iniciar Escaneo"):
    if not st.session_state.get("api_key"):
        st.sidebar.error(
            "No hay clave disponible. Cárgala desde el .env o ingrésela manualmente."
        )
    else:
        with st.spinner("Buscando ofertas en GetOnBoard..."):
            df_nuevas = obtener_empleos_getonboard()
            if not df_nuevas.empty:
                df_nuevas["fuente"] = "GetOnBoard"

            df_existentes = get_all_jobs()
            if not df_nuevas.empty and not df_existentes.empty:
                df_nuevas = df_nuevas[~df_nuevas["url"].isin(df_existentes["url"])]

            df_combinado = pd.concat([df_existentes, df_nuevas], ignore_index=True).fillna("")
            st.session_state["jobs_df"] = df_combinado
            st.session_state.pop("evaluated_df", None)

        st.sidebar.success(
            f"Se encontraron {len(df_nuevas)} ofertas nuevas y {len(df_combinado)} en total."
        )

if st.sidebar.button("Filtrar con IA (DeepSeek)"):
    if not st.session_state.get("api_key"):
        st.sidebar.error("No hay clave de DeepSeek disponible.")
    elif "jobs_df" not in st.session_state or st.session_state["jobs_df"].empty:
        st.sidebar.error("Primero ejecuta 'Iniciar Escaneo'.")
    else:
        df = st.session_state["jobs_df"].copy()
        df["pasa_prefiltro"] = df.apply(_cumple_prefiltro, axis=1)
        df_prefiltrado = df[df["pasa_prefiltro"]].copy()
        descartadas = len(df) - len(df_prefiltrado)
        st.sidebar.info(
            f"Pre-filtrado: {descartadas} descartadas, {len(df_prefiltrado)} pasan al análisis."
        )

        df_nuevas = filter_new_jobs(df_prefiltrado)

        if df_nuevas.empty:
            st.sidebar.info(
                "Todas las ofertas pre-filtradas ya fueron evaluadas. Cargando resultados guardados."
            )
            st.session_state["evaluated_df"] = get_all_jobs()
        else:
            ofertas_nuevas = df_nuevas.to_dict("records")
            resultados = []
            barra = st.sidebar.progress(0)
            total_lotes = (len(ofertas_nuevas) + LOTE_TAMANO - 1) // LOTE_TAMANO
            for i in range(0, len(ofertas_nuevas), LOTE_TAMANO):
                lote = ofertas_nuevas[i : i + LOTE_TAMANO]
                evaluaciones = evaluar_lote_ofertas(lote, st.session_state["api_key"])
                resultados.extend(evaluaciones)
                barra.progress(int((i + len(lote)) / len(ofertas_nuevas) * 100))
            barra.empty()

            df_nuevas = df_nuevas.copy()
            df_nuevas["aprobado"] = [r.get("aprobado", False) for r in resultados]
            df_nuevas["score"] = [r.get("score", 0) for r in resultados]
            df_nuevas["tipo_contrato_estimado"] = [
                r.get("tipo_contrato_estimado", "") for r in resultados
            ]
            df_nuevas["razon"] = [r.get("razon", "") for r in resultados]

            df_existentes = get_all_jobs()
            df_evaluado = pd.concat([df_existentes, df_nuevas], ignore_index=True).fillna("")
            save_jobs(df_evaluado)
            st.session_state["evaluated_df"] = df_evaluado
            aprobadas = sum(1 for r in resultados if r.get("aprobado"))
            st.sidebar.success(
                f"Evaluación: {aprobadas} de {len(df_nuevas)} aprobadas. "
                f"Llamadas API: {total_lotes} (vs {len(df_nuevas)} individuales)"
            )

with placeholder.container():
    if "evaluated_df" in st.session_state and not st.session_state["evaluated_df"].empty:
        df = st.session_state["evaluated_df"].copy()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score_min = st.slider("Score mínimo", 0, 100, 0)
        with col2:
            busqueda = st.text_input("Buscar palabra clave", "")
        with col3:
            orden = st.selectbox("Ordenar por score", ["Mayor a menor", "Menor a mayor"])
        with col4:
            vista = st.radio("Vista", ["Todas", "Solo aprobadas"], horizontal=True)

        if "aprobado" in df.columns:
            df["aprobado"] = df["aprobado"].astype(bool)

        if "score" in df.columns:
            df = df[df["score"] >= score_min]

        if busqueda:
            patron = busqueda.lower()
            columnas_buscar = ["titulo", "empresa", "descripcion", "razon"]
            df = df[
                df[columnas_buscar]
                .fillna("")
                .apply(lambda x: x.str.lower().str.contains(patron, na=False))
                .any(axis=1)
            ]

        if vista == "Solo aprobadas" and "aprobado" in df.columns:
            df = df[df["aprobado"] == True]

        if "score" in df.columns:
            df = df.sort_values(by="score", ascending=(orden == "Menor a mayor"))

        st.dataframe(df, use_container_width=True)

        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Exportar a CSV",
                csv,
                "ofertas_aprobadas.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_xlsx:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            st.download_button(
                "Exportar a Excel",
                buffer.getvalue(),
                "ofertas_aprobadas.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    elif "jobs_df" in st.session_state and not st.session_state["jobs_df"].empty:
        st.dataframe(st.session_state["jobs_df"], use_container_width=True)
    else:
        st.info(
            "Aquí se mostrarán los resultados de las ofertas de trabajo en las siguientes fases."
        )
