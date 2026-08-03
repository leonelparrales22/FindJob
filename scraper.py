import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.getonbrd.com/api/v0/search/jobs"


def _normalizar_valor(valor):
    """Convierte valores anidados o JSON serializado en texto plano."""
    if valor is None:
        return ""
    if isinstance(valor, dict):
        for clave in ["name", "value", "title", "label", "text"]:
            if clave in valor:
                return str(valor[clave]) if valor[clave] is not None else ""
        for clave in ["data"]:
            if clave in valor:
                return _normalizar_valor(valor[clave])
        for v in valor.values():
            if isinstance(v, (str, int, float, bool)):
                return str(v)
            if isinstance(v, (dict, list)):
                return _normalizar_valor(v)
        return ""
    if isinstance(valor, list):
        if not valor:
            return ""
        return _normalizar_valor(valor[0])
    if isinstance(valor, str):
        valor_limpio = valor.strip()
        if valor_limpio.startswith("{") or valor_limpio.startswith("["):
            try:
                parsed = json.loads(valor_limpio)
                return _normalizar_valor(parsed)
            except json.JSONDecodeError:
                pass
        return valor_limpio
    return str(valor)


def _limpiar_html(texto: str) -> str:
    """Elimina etiquetas HTML de un texto y normaliza espacios."""
    if not texto:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    soup = BeautifulSoup(texto, "html.parser")
    limpio = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", limpio).strip()


def _extraer_campo(item: dict, alternativas: list[str]):
    """Busca el primer campo disponible entre las alternativas."""
    for clave in alternativas:
        if clave in item:
            return item[clave]
    for clave in alternativas:
        for subvalor in item.values():
            if isinstance(subvalor, dict) and clave in subvalor:
                return subvalor[clave]
    return None


def obtener_empleos_getonboard(query: str = "data engineer") -> pd.DataFrame:
    """Consulta ofertas de trabajo en GetOnBoard y devuelve un DataFrame."""
    columnas = ["titulo", "empresa", "modalidad", "ubicacion", "salario", "descripcion", "url"]

    try:
        respuesta = requests.get(BASE_URL, params={"query": query}, timeout=15)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except requests.RequestException as e:
        print(f"Error al consultar GetOnBoard: {e}")
        return pd.DataFrame(columns=columnas)
    except ValueError as e:
        print(f"Error al decodificar la respuesta JSON: {e}")
        return pd.DataFrame(columns=columnas)

    empleos = datos
    if isinstance(datos, dict):
        for clave in ["data", "jobs", "results", "items", "offers"]:
            if clave in datos and isinstance(datos[clave], list):
                empleos = datos[clave]
                break

    if not isinstance(empleos, list):
        print("No se encontró una lista de empleos en la respuesta.")
        if isinstance(datos, dict):
            print(f"Claves disponibles en la raíz: {list(datos.keys())}")
        return pd.DataFrame(columns=columnas)

    registros = []
    for item in empleos:
        if not isinstance(item, dict):
            continue

        titulo = _extraer_campo(item, ["title", "name", "titulo", "job_title"])
        empresa = _extraer_campo(
            item,
            ["company_name", "company", "empresa", "organization", "name"],
        )
        modalidad = _extraer_campo(
            item,
            ["modality", "work_modality", "remote", "modalidad", "work_mode", "location_type"],
        )
        ubicacion = _extraer_campo(
            item,
            ["location", "city", "country", "place", "region", "address"],
        )
        salario = _extraer_campo(
            item,
            ["salary", "salario", "compensation", "remuneration", "pay", "income"],
        )
        descripcion = _extraer_campo(
            item,
            ["description", "job_description", "descripcion", "desc"],
        )
        url_oferta = _extraer_campo(
            item,
            ["url", "link", "public_url", "job_url", "apply_url", "permalink"],
        )

        registros.append(
            {
                "titulo": titulo or "",
                "empresa": _normalizar_valor(empresa),
                "modalidad": _normalizar_valor(modalidad),
                "ubicacion": _normalizar_valor(ubicacion),
                "salario": _normalizar_valor(salario),
                "descripcion": _limpiar_html(descripcion) if descripcion else "",
                "url": url_oferta or "",
            }
        )

    return pd.DataFrame(registros, columns=columnas)


def obtener_empleos_remoteok(query: str = "data engineer") -> pd.DataFrame:
    """Consulta ofertas de trabajo en RemoteOK y devuelve un DataFrame."""
    columnas = ["titulo", "empresa", "modalidad", "ubicacion", "salario", "descripcion", "url"]

    try:
        respuesta = requests.get(
            "https://remoteok.com/api",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
    except requests.RequestException as e:
        print(f"Error al consultar RemoteOK: {e}")
        return pd.DataFrame(columns=columnas)
    except ValueError as e:
        print(f"Error al decodificar la respuesta JSON de RemoteOK: {e}")
        return pd.DataFrame(columns=columnas)

    empleos = datos
    if isinstance(datos, dict):
        for clave in ["jobs", "data", "results", "items", "offers"]:
            if clave in datos and isinstance(datos[clave], list):
                empleos = datos[clave]
                break

    if not isinstance(empleos, list):
        print("No se encontró una lista de empleos en la respuesta de RemoteOK.")
        return pd.DataFrame(columns=columnas)

    query_limpia = query.lower()
    registros = []
    for item in empleos:
        if not isinstance(item, dict):
            continue

        titulo = _extraer_campo(item, ["position", "title", "name", "titulo"])
        descripcion = _extraer_campo(
            item, ["description", "job_description", "descripcion", "desc"]
        )
        tags = item.get("tags", [])
        tags_texto = " ".join(str(t) for t in tags).lower() if isinstance(tags, list) else ""

        if (
            query_limpia not in (titulo or "").lower()
            and query_limpia not in (descripcion or "").lower()
            and query_limpia not in tags_texto
        ):
            continue

        empresa = _extraer_campo(item, ["company", "company_name", "empresa", "organization", "name"])
        ubicacion = _extraer_campo(item, ["location", "city", "country", "place", "region"])
        salario = _extraer_campo(item, ["salary", "salario", "compensation", "remuneration", "pay"])
        url_oferta = _extraer_campo(
            item, ["url", "link", "public_url", "job_url", "apply_url", "source"]
        )

        registros.append(
            {
                "titulo": titulo or "",
                "empresa": _normalizar_valor(empresa),
                "modalidad": _normalizar_valor("Remoto"),
                "ubicacion": _normalizar_valor(ubicacion),
                "salario": _normalizar_valor(salario),
                "descripcion": _limpiar_html(descripcion) if descripcion else "",
                "url": url_oferta or "",
            }
        )

    return pd.DataFrame(registros, columns=columnas)
