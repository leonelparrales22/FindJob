import re

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
]

ROLES_GENERICOS = [
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

IDIOMA_KEYWORDS = [
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

JERARQUIA_KEYWORDS = [
    "junior",
    "trainee",
    "intern",
    "pasante",
    "becario",
    "entry level",
    "practicante",
    "pasantía",
]

NO_RELEVANTES_KEYWORDS = [
    "data entry",
    "customer data",
    "operador de datos",
    "captura de datos",
]

DESCARTAR_KEYWORDS = IDIOMA_KEYWORDS + JERARQUIA_KEYWORDS + NO_RELEVANTES_KEYWORDS

SALARIO_MINIMO = 2000


def _quitar_acentos(texto) -> str:
    """Convierte a minúsculas y elimina acentos."""
    if texto is None:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    if not texto:
        return ""
    texto = texto.lower()
    vocales = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunaeiouun")
    return texto.translate(vocales)


def _es_ubicacion_valida(modalidad: str, ubicacion: str) -> bool:
    """Acepta remoto siempre; presencial o híbrido solo si es en Quito o Ecuador."""
    modalidad_limpia = _quitar_acentos(modalidad)
    ubicacion_limpia = _quitar_acentos(ubicacion)
    if "remoto" in modalidad_limpia or "remote" in modalidad_limpia:
        return True
    if any(u in ubicacion_limpia for u in ["quito", "ecuador"]):
        return True
    # Si no hay ubicación clara, dejar pasar para que DeepSeek decida
    return not ubicacion_limpia


def _extraer_salario_minimo(texto) -> float | None:
    """Extrae el salario numerico mas bajo encontrado en el texto."""
    if texto is None:
        return None
    if not isinstance(texto, str):
        texto = str(texto)
    if not texto:
        return None
    numeros = []
    for palabra in texto.split():
        limpio = palabra.replace(",", "").strip("$?")
        if limpio.lower().startswith("usd") or limpio.lower() in ["dolares", "d?lares"]:
            continue
        try:
            numeros.append(float(limpio))
        except ValueError:
            continue
    return min(numeros) if numeros else None



def _es_salario_valido(row, salario_minimo: int = SALARIO_MINIMO) -> bool:
    """Descarta ofertas cuyo salario explícito esté por debajo del mínimo."""
    salario_texto = f"{row.get('salario', '')} {row.get('descripcion', '')}"
    salario_min = _extraer_salario_minimo(salario_texto)
    if salario_min is None:
        return True
    return salario_min >= salario_minimo


def _cumple_prefiltro(row, salario_minimo: int = SALARIO_MINIMO) -> bool:
    """Verifica rol, stack, idioma, ubicación y salario."""
    texto = _quitar_acentos(
        f"{row.get('titulo', '')} {row.get('descripcion', '')} {row.get('empresa', '')} {row.get('ubicacion', '')} {row.get('salario', '')}"
    )
    tiene_rol_especifico = any(kw in texto for kw in ROLES_KEYWORDS)
    tiene_rol_generico = any(kw in texto for kw in ROLES_GENERICOS)
    tiene_stack = any(kw in texto for kw in STACK_KEYWORDS)
    descartar = any(kw in texto for kw in DESCARTAR_KEYWORDS)
    ubicacion_ok = _es_ubicacion_valida(row.get("modalidad", ""), row.get("ubicacion", ""))
    salario_ok = _es_salario_valido(row, salario_minimo)
    return (tiene_rol_especifico or tiene_rol_generico) and tiene_stack and not descartar and ubicacion_ok and salario_ok


def _motivo_descarte(row, salario_minimo: int = SALARIO_MINIMO) -> list[str]:
    """Devuelve la lista de razones por las que una oferta no pasa el pre-filtro."""
    texto = _quitar_acentos(
        f"{row.get('titulo', '')} {row.get('descripcion', '')} {row.get('empresa', '')} {row.get('ubicacion', '')} {row.get('salario', '')}"
    )
    tiene_rol_especifico = any(kw in texto for kw in ROLES_KEYWORDS)
    tiene_rol_generico = any(kw in texto for kw in ROLES_GENERICOS)
    tiene_stack = any(kw in texto for kw in STACK_KEYWORDS)
    ubicacion_ok = _es_ubicacion_valida(row.get("modalidad", ""), row.get("ubicacion", ""))
    salario_ok = _es_salario_valido(row, salario_minimo)

    motivos = []
    if not (tiene_rol_especifico or tiene_rol_generico):
        motivos.append("falta_rol")
    if not tiene_stack:
        motivos.append("falta_stack")
    if any(kw in texto for kw in IDIOMA_KEYWORDS):
        motivos.append("idioma")
    if any(kw in texto for kw in JERARQUIA_KEYWORDS):
        motivos.append("jerarquia")
    if any(kw in texto for kw in NO_RELEVANTES_KEYWORDS):
        motivos.append("no_relevante")
    if not ubicacion_ok:
        motivos.append("ubicacion")
    if not salario_ok:
        motivos.append("salario")
    return motivos
