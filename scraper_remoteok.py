import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://remoteok.com/api"


def _limpiar_html(texto: str) -> str:
    """Elimina etiquetas HTML y normaliza espacios."""
    if not texto:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    soup = BeautifulSoup(texto, "html.parser")
    limpio = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", limpio).strip()


def _extraer(oferta: dict, alternativas: list[str]):
    """Busca el primer campo disponible entre las alternativas."""
    for clave in alternativas:
        if clave in oferta and oferta[clave]:
            return oferta[clave]
    return ""


def _normalizar_url(url):
    if not url:
        return ""
    url = str(url)
    if not url.startswith("http"):
        url = "https://remoteok.com" + (url if url.startswith("/") else "/" + url)
    return url


def _normalizar_modalidad(modalidad):
    if not modalidad:
        return "Remoto"
    if isinstance(modalidad, list):
        modos = [str(m) for m in modalidad if m]
        if any("remote" in m.lower() for m in modos):
            return "Remoto"
        return ", ".join(modos)
    if isinstance(modalidad, str):
        return modalidad
    return str(modalidad)


def obtener_empleos_remoteok(query: str = "data engineer") -> pd.DataFrame:
    """Consulta RemoteOK y devuelve un DataFrame estandarizado."""
    columnas = ["titulo", "empresa", "modalidad", "descripcion", "url", "fuente"]

    try:
        respuesta = requests.get(
            BASE_URL,
            params={"tag": query},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
    except requests.RequestException as e:
        print(f"Error al consultar RemoteOK: {e}")
        return pd.DataFrame(columns=columnas)
    except ValueError as e:
        print(f"Error al decodificar RemoteOK: {e}")
        return pd.DataFrame(columns=columnas)

    if not isinstance(datos, list):
        print("Respuesta de RemoteOK no es una lista.")
        return pd.DataFrame(columns=columnas)

    # El primer elemento suele ser metadatos; filtrar diccionarios con oferta
    ofertas = [x for x in datos if isinstance(x, dict) and (x.get("position") or x.get("title"))]

    registros = []
    for item in ofertas:
        titulo = _extraer(item, ["position", "title", "job_title"])
        empresa = _extraer(item, ["company", "company_name"])
        modalidad = _normalizar_modalidad(_extraer(item, ["location", "modality", "remote"]))
        descripcion = _limpiar_html(_extraer(item, ["description", "job_description"]))
        url = _normalizar_url(_extraer(item, ["url", "apply_url", "link", "slug"]))

        if not titulo:
            continue

        registros.append(
            {
                "titulo": titulo,
                "empresa": empresa,
                "modalidad": modalidad,
                "descripcion": descripcion,
                "url": url,
                "fuente": "RemoteOK",
            }
        )

    return pd.DataFrame(registros, columns=columnas)
