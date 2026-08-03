import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import filter_new_jobs, get_all_jobs, inicializar_db, save_jobs
from evaluator import evaluar_lote_ofertas, evaluar_oferta
from prefilter import (
    SALARIO_MINIMO,
    _cumple_prefiltro,
    _motivo_descarte,
    _quitar_acentos,
)
from scraper import obtener_empleos_getonboard, obtener_empleos_remoteok

load_dotenv()
inicializar_db()

def _content_hash(row) -> str:
    """Genera un hash del contenido relevante de la oferta."""
    texto = (
        f"{row.get('titulo', '')}{row.get('descripcion', '')}"
        f"{row.get('salario', '')}{row.get('ubicacion', '')}"
    ).lower()
    return hashlib.md5(texto.encode()).hexdigest()[:16]


def _salario_minimo_actual() -> int:
    """Devuelve el salario mínimo configurado en la UI."""
    return st.session_state.get("salario_minimo", SALARIO_MINIMO)


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

st.sidebar.divider()
st.sidebar.subheader("Configuración")
st.session_state["salario_minimo"] = st.sidebar.slider(
    "Salario mínimo (USD)", 0, 5000, 2000, step=100
)
st.session_state["umbral_aprobacion"] = st.sidebar.slider(
    "Score mínimo para aprobación", 0, 100, 70, step=5
)
st.session_state["lote_tamano"] = st.sidebar.slider(
    "Ofertas por lote (DeepSeek)", 1, 15, 5, step=1
)
st.session_state["usar_getonboard"] = st.sidebar.checkbox("Usar GetOnBoard", True)
st.session_state["usar_remoteok"] = st.sidebar.checkbox("Usar RemoteOK", False)
st.session_state["modalidades_mostrar"] = st.sidebar.multiselect(
    "Modalidades a mostrar", ["Remoto", "Híbrido", "Presencial"], default=["Remoto", "Híbrido", "Presencial"]
)

placeholder = st.empty()

if st.sidebar.button("Iniciar Escaneo"):
    if not st.session_state.get("api_key"):
        st.sidebar.error(
            "No hay clave disponible. Cárgala desde el .env o ingrésela manualmente."
        )
    else:
        with st.spinner("Buscando ofertas en fuentes activas..."):
            dfs = []
            if st.session_state.get("usar_getonboard", True):
                df_gob = obtener_empleos_getonboard()
                if not df_gob.empty:
                    df_gob["fuente"] = "GetOnBoard"
                    dfs.append(df_gob)
            if st.session_state.get("usar_remoteok", False):
                df_rok = obtener_empleos_remoteok()
                if not df_rok.empty:
                    df_rok["fuente"] = "RemoteOK"
                    dfs.append(df_rok)

            if dfs:
                df_nuevas = pd.concat(dfs, ignore_index=True)
                df_nuevas["content_hash"] = df_nuevas.apply(_content_hash, axis=1)
            else:
                df_nuevas = pd.DataFrame(columns=["titulo", "empresa", "modalidad", "ubicacion", "salario", "descripcion", "url", "fuente"])

            df_existentes = get_all_jobs()
            if not df_nuevas.empty and not df_existentes.empty:
                df_nuevas = df_nuevas[~df_nuevas["url"].isin(df_existentes["url"])]

            df_combinado = pd.concat([df_existentes, df_nuevas], ignore_index=True).fillna("")
            st.session_state["jobs_df"] = df_combinado
            st.session_state.pop("evaluated_df", None)

        st.session_state["metricas"] = {
            "escaneadas": len(df_combinado),
            "prefiltradas": len(df_combinado),
            "evaluadas": 0,
            "aprobadas": 0,
            "llamadas_api": 0,
        }
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
        salario_minimo = _salario_minimo_actual()
        df["pasa_prefiltro"] = df.apply(
            lambda row: _cumple_prefiltro(row, salario_minimo=salario_minimo), axis=1
        )
        df["motivo_descarte"] = df.apply(
            lambda row: "; ".join(_motivo_descarte(row, salario_minimo=salario_minimo)), axis=1
        )
        df_prefiltrado = df[df["pasa_prefiltro"]].copy()
        st.session_state["debug_df"] = df
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
            lote_tamano = st.session_state.get("lote_tamano", 5)
            lotes = [
                (i, ofertas_nuevas[i : i + lote_tamano])
                for i in range(0, len(ofertas_nuevas), lote_tamano)
            ]
            total_lotes = len(lotes)

            def _evaluar_lote(args):
                idx, lote = args
                umbral = st.session_state.get("umbral_aprobacion", 70)
                return idx, evaluar_lote_ofertas(
                    lote, st.session_state["api_key"], umbral=umbral
                )

            resultados_parciales = [None] * total_lotes
            prompt_tokens = 0
            completion_tokens = 0
            barra = st.sidebar.progress(0)
            completados = 0

            max_workers = min(4, max(1, total_lotes))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futuros = {
                    executor.submit(_evaluar_lote, item): item[0]
                    for item in lotes
                }
                for futuro in as_completed(futuros):
                    i, respuesta = futuro.result()
                    idx_lote = i // lote_tamano
                    resultados_parciales[idx_lote] = respuesta.get("resultados", [])
                    uso = respuesta.get("usage", {})
                    prompt_tokens += uso.get("prompt_tokens", 0)
                    completion_tokens += uso.get("completion_tokens", 0)
                    completados += 1
                    barra.progress(int(completados / total_lotes * 100))
            barra.empty()

            resultados = [r for sub in resultados_parciales for r in sub]

            df_nuevas = df_nuevas.copy()
            df_nuevas["aprobado"] = [r.get("aprobado", False) for r in resultados]
            df_nuevas["score"] = [r.get("score", 0) for r in resultados]
            df_nuevas["tipo_contrato_estimado"] = [
                r.get("tipo_contrato_estimado", "") for r in resultados
            ]
            df_nuevas["razon"] = [r.get("razon", "") for r in resultados]
            df_nuevas = df_nuevas.drop(
                columns=["pasa_prefiltro", "motivo_descarte"], errors="ignore"
            )

            df_existentes = get_all_jobs()
            df_evaluado = pd.concat([df_existentes, df_nuevas], ignore_index=True).fillna("")
            save_jobs(df_evaluado)
            st.session_state["evaluated_df"] = df_evaluado
            aprobadas = sum(1 for r in resultados if r.get("aprobado"))

            tokens_total = prompt_tokens + completion_tokens
            costo_usd = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000

            st.session_state["metricas"] = {
                "escaneadas": len(df),
                "prefiltradas": len(df_prefiltrado),
                "evaluadas": len(df_nuevas),
                "aprobadas": aprobadas,
                "llamadas_api": total_lotes,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_total": tokens_total,
                "costo_usd": costo_usd,
            }
            st.sidebar.success(
                f"Evaluación: {aprobadas} de {len(df_nuevas)} aprobadas. "
                f"Llamadas API: {total_lotes} (vs {len(df_nuevas)} individuales)"
            )

st.sidebar.divider()
st.sidebar.subheader("Métricas")
metricas = st.session_state.get("metricas", {})
if metricas:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Escaneadas", metricas.get("escaneadas", 0))
    c2.metric("Pre-filtradas", metricas.get("prefiltradas", 0))
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Evaluadas", metricas.get("evaluadas", 0))
    c4.metric("Aprobadas", metricas.get("aprobadas", 0))
    st.sidebar.info(f"Llamadas API: {metricas.get('llamadas_api', 0)}")
    tokens = metricas.get("tokens_total", 0)
    if tokens:
        st.sidebar.info(
            f"Tokens: {tokens} "
            f"(${metricas.get('costo_usd', 0):.4f} USD aprox.)"
        )

with placeholder.container():
    tab_resultados, tab_debug = st.tabs(["Resultados", "Depuración"])

    with tab_resultados:
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

            modalidades_mostrar = st.session_state.get("modalidades_mostrar", ["Remoto", "Híbrido", "Presencial"])
            if modalidades_mostrar and "modalidad" in df.columns:
                modalidades_limpias = [_quitar_acentos(m) for m in modalidades_mostrar]
                df = df[
                    df["modalidad"]
                    .fillna("")
                    .apply(lambda x: any(m in _quitar_acentos(x) for m in modalidades_limpias))
                ]

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

            formato = st.radio(
                "Formato", ["Tabla", "Tarjetas"], horizontal=True, key="vista_formato"
            )

            if formato == "Tabla":
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
            else:
                st.subheader(f"Mostrando {len(df)} ofertas")
                for _, row in df.iterrows():
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"### {row.get('titulo', '')}")
                            st.markdown(
                                f"**{row.get('empresa', '')}** · "
                                f"{row.get('modalidad', '')} · {row.get('ubicacion', '')}"
                            )
                            if row.get("salario"):
                                st.markdown(f"💰 {row.get('salario')}")
                            st.markdown(
                                f"Score: **{row.get('score', 0)}** | "
                                f"Contrato: {row.get('tipo_contrato_estimado', '')}"
                            )
                            st.markdown(f"📝 {row.get('razon', '')}")
                        with col_b:
                            url = row.get("url", "")
                            if url:
                                st.markdown(f"🔗 [Abrir oferta]({url})")
                        st.divider()

        elif "jobs_df" in st.session_state and not st.session_state["jobs_df"].empty:
            st.dataframe(st.session_state["jobs_df"], use_container_width=True)
        else:
            st.info(
                "Aquí se mostrarán los resultados de las ofertas de trabajo en las siguientes fases."
            )

    with tab_debug:
        if "debug_df" in st.session_state and not st.session_state["debug_df"].empty:
            debug = st.session_state["debug_df"].copy()
            descartadas = debug[debug["pasa_prefiltro"] == False]
            st.markdown(f"**Ofertas descartadas:** {len(descartadas)}")
            columnas_debug = [c for c in ["titulo", "empresa", "modalidad", "ubicacion", "salario", "motivo_descarte"] if c in descartadas.columns]
            st.dataframe(descartadas[columnas_debug], use_container_width=True)
        else:
            st.info(
                "Ejecuta 'Filtrar con IA' para ver el detalle de ofertas descartadas."
            )
