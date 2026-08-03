import json

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
UMBRAL_APROBACION = 70

_SYSTEM_PROMPT = """Eres un asesor experto en carreras de datos, arquitectura de datos e inteligencia artificial.

Tu tarea es evaluar ofertas laborales en español o inglés y decidir si se alinean con el siguiente perfil:

Perfil buscado:
- Roles aceptados: Ingeniero de Datos, Arquitecto de Datos, Ingeniero de IA, AI Engineer, Data Engineer, Data Architect, ML Engineer, Analytics Engineer. También Lead o Senior de los anteriores.
- Stack relevante: Python, PySpark, Databricks, SQL, .NET Core, AWS, Azure, LLMs, RAG, MCP, ETL, EL, Data Warehouse, Lakehouse, Big Data.

Criterios salariales/jerárquicos:
- Posiciones Senior/Lead con potencial de alcanzar:
  * $2,800+ USD mensuales en relación de dependencia.
  * $3,500+ USD mensuales como Contractor/Freelance.

Rúbrica de score (0-100):
- 80-100: Stack muy alineado, claramente Senior/Lead, salario/jerarquía compatible.
- 50-79: Algunos aspectos coinciden, pero hay dudas de stack, senioridad o salario.
- 0-49: Poco alineado: stack diferente, junior, salario bajo o rol ajeno.

Regla de idioma:
- El usuario no habla inglés. Si la oferta requiere inglés avanzado, fluido, nativo o lo menciona como requisito (ej. "English required", "Fluent English", "Native English"), penaliza fuertemente: score máximo 25 y aprobado=false.

Regla de ubicación:
- El usuario vive en Quito, Ecuador. Si la modalidad es presencial o híbrida, solo aprueba si la ubicación es Quito o Ecuador. Si es remota/100% remoto, puede aprobar sin importar ubicación. Si no se especifica ubicación y la modalidad no es remota, penaliza el score.

Regla de salario:
- Si la oferta menciona un salario numérico, aprúebala solo si cumple o supera los objetivos: $2,800+ USD mensuales en dependencia o $3,500+ USD como contractor. Si el salario es inferior a $2,000 USD, penaliza fuertemente el score.

Ejemplos guía:

Oferta 1 - Alta alineación:
Título: Senior Data Engineer - Databricks, PySpark, AWS
Descripción: Buscamos Senior Data Engineer con 5+ años. Stack: Python, PySpark, Databricks, SQL, AWS. Salario $3,000-$4,000 USD/mes. Remoto.
Evaluación esperada: { "aprobado": true, "score": 92, "tipo_contrato_estimado": "Dependencia", "razon": "Senior, stack completo con Databricks/PySpark/AWS, salario en rango." }

Oferta 2 - Baja alineación:
Título: Junior Data Analyst
Descripción: Analista junior recién egresado. Stack: Excel, SQL básico. Pasantía.
Evaluación esperada: { "aprobado": false, "score": 15, "tipo_contrato_estimado": "Pasantía", "razon": "Junior/pasantía, stack básico, no cumple senioridad ni salario." }

Oferta 3 - Parcial:
Título: Data Engineer
Descripción: Experiencia con Python y SQL. Trabajo híbrido. No especifica senioridad ni salario.
Evaluación esperada: { "aprobado": false, "score": 55, "tipo_contrato_estimado": "Indefinido", "razon": "Stack correcto pero falta senioridad y salario; riesgo de no cumplir." }

Reglas de salida:
- Responde EXACTAMENTE con un objeto JSON.
- No añadas texto explicativo fuera del JSON.
- Las claves obligatorias son:
  * aprobado (booleano): true solo si score >= 70 y la oferta parece cumplir los criterios.
  * score (entero entre 0 y 100): confianza de alineación según la rúbrica.
  * tipo_contrato_estimado (string breve): "Dependencia", "Contractor", "Freelance", "Indefinido" o similar.
  * razon (string breve): máximo 150 caracteres explicando por qué apruebas o rechazas.
"""


def _parsear_lote(contenido: str, total: int) -> list[dict]:
    """Parsea la respuesta JSON de un lote y retorna una lista de evaluaciones."""
    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as e:
        raise ValueError(f"Respuesta no es JSON válido: {e}")

    if "resultados" in datos and isinstance(datos["resultados"], list):
        resultados = datos["resultados"]
    elif isinstance(datos, list):
        resultados = datos
    else:
        raise ValueError("La respuesta no contiene la clave 'resultados' ni una lista")

    if len(resultados) != total:
        raise ValueError(f"Se esperaban {total} resultados, se recibieron {len(resultados)}")

    evaluaciones = []
    for r in resultados:
        score = int(r.get("score", 0))
        aprobado = bool(r.get("aprobado", False)) and score >= UMBRAL_APROBACION
        evaluaciones.append(
            {
                "aprobado": aprobado,
                "score": score,
                "tipo_contrato_estimado": str(r.get("tipo_contrato_estimado", "")),
                "razon": str(r.get("razon", "")),
            }
        )
    return evaluaciones


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _llamar_deepseek_lote(cliente, ofertas_texto: str, total: int) -> list[dict]:
    """Realiza una llamada a DeepSeek y retorna el lote de evaluaciones."""
    respuesta = cliente.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": ofertas_texto},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1200,
        seed=42,
    )
    contenido = respuesta.choices[0].message.content
    return _parsear_lote(contenido, total)


def evaluar_lote_ofertas(ofertas: list[dict], api_key: str) -> list[dict]:
    """Evalúa un lote de ofertas con DeepSeek y devuelve una lista de evaluaciones."""
    if not ofertas:
        return []

    cliente = OpenAI(base_url=BASE_URL, api_key=api_key)

    texto_ofertas = []
    for i, oferta in enumerate(ofertas, 1):
        texto_ofertas.append(
            f"""Oferta {i}:
Título: {oferta.get('titulo', '')}
Empresa: {oferta.get('empresa', '')}
Modalidad: {oferta.get('modalidad', '')}
Ubicación: {oferta.get('ubicacion', '')}
Salario: {oferta.get('salario', '')}
Descripción: {oferta.get('descripcion', '')[:800]}
"""
        )

    user_prompt = (
        "Evalúa las siguientes ofertas laborales y devuelve un JSON con la clave 'resultados', "
        "que sea un array de objetos. Cada objeto debe tener exactamente las claves: "
        "aprobado, score, tipo_contrato_estimado, razon. "
        "El array debe tener el mismo orden y cantidad de ofertas que se te envían.\n\n"
        + "\n---\n".join(texto_ofertas)
    )

    try:
        return _llamar_deepseek_lote(cliente, user_prompt, len(ofertas))
    except Exception as e:
        return [
            {
                "aprobado": False,
                "score": 0,
                "tipo_contrato_estimado": "",
                "razon": f"Error en la evaluación del lote: {e}",
            }
            for _ in ofertas
        ]


def evaluar_oferta(titulo: str, empresa: str, modalidad: str, descripcion: str, api_key: str) -> dict:
    """Evalúa una oferta con DeepSeek y devuelve un diccionario estructurado."""
    oferta = {
        "titulo": titulo,
        "empresa": empresa,
        "modalidad": modalidad,
        "descripcion": descripcion,
    }
    resultados = evaluar_lote_ofertas([oferta], api_key)
    return resultados[0] if resultados else {
        "aprobado": False,
        "score": 0,
        "tipo_contrato_estimado": "",
        "razon": "Error en la evaluación",
    }
