import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "job_offers.db"
COLUMNS = [
    "url",
    "titulo",
    "empresa",
    "modalidad",
    "ubicacion",
    "descripcion",
    "aprobado",
    "score",
    "tipo_contrato_estimado",
    "razon",
    "fecha_guardado",
]


def _conexion():
    return sqlite3.connect(DB_PATH)


def inicializar_db():
    """Crea la tabla de ofertas si no existe."""
    with _conexion() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                url TEXT PRIMARY KEY,
                titulo TEXT,
                empresa TEXT,
                modalidad TEXT,
                ubicacion TEXT,
                descripcion TEXT,
                aprobado INTEGER,
                score INTEGER,
                tipo_contrato_estimado TEXT,
                razon TEXT,
                fecha_guardado TEXT
            )
            """
        )
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN ubicacion TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def _normalizar_fila(row, columnas):
    """Convierte una fila de DataFrame a tupla compatible con SQLite."""
    valores = []
    for col in columnas:
        if col in row:
            val = row[col]
            if col == "aprobado" and not pd.isna(val):
                val = 1 if val else 0
            elif pd.isna(val):
                val = "" if col != "score" else 0
            valores.append(val)
        else:
            valores.append("" if col != "score" else 0)
    return tuple(valores)


def save_jobs(df: pd.DataFrame):
    """Guarda o actualiza un DataFrame de ofertas en la base de datos."""
    if df.empty:
        return

    df = df.copy()
    if "fecha_guardado" not in df.columns:
        df["fecha_guardado"] = datetime.now().isoformat()

    # Asegurar tipos de datos
    if "aprobado" in df.columns:
        df["aprobado"] = df["aprobado"].apply(lambda x: 1 if x else 0)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

    registros = [_normalizar_fila(row, COLUMNS) for _, row in df.iterrows()]
    with _conexion() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO jobs ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})",
            registros,
        )
        conn.commit()


def get_all_jobs() -> pd.DataFrame:
    """Devuelve todas las ofertas guardadas en la base de datos."""
    with _conexion() as conn:
        df = pd.read_sql_query("SELECT * FROM jobs", conn)
    if not df.empty and "aprobado" in df.columns:
        df["aprobado"] = df["aprobado"].astype(bool)
    return df


def get_evaluated_urls() -> set[str]:
    """Devuelve el conjunto de URLs que ya tienen evaluación."""
    with _conexion() as conn:
        cursor = conn.execute("SELECT url FROM jobs WHERE aprobado IS NOT NULL")
        return {row[0] for row in cursor.fetchall()}


def filter_new_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra el DataFrame para devolver solo ofertas no evaluadas."""
    if df.empty:
        return df
    urls_existentes = get_evaluated_urls()
    return df[~df["url"].isin(urls_existentes)].copy()
