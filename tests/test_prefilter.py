import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prefilter


def test_quitar_acentos():
    assert prefilter._quitar_acentos("Ingeniero de Datos") == "ingeniero de datos"
    assert prefilter._quitar_acentos("Quito, Ecuador") == "quito, ecuador"
    assert prefilter._quitar_acentos(123) == "123"
    assert prefilter._quitar_acentos(None) == ""


def test_extraer_salario_minimo():
    assert prefilter._extraer_salario_minimo("Salario $3,500 USD/mes") == 3500.0
    assert prefilter._extraer_salario_minimo("Paga 2500 USD a 4000 USD") == 2500.0
    assert prefilter._extraer_salario_minimo("No especifica") is None
    assert prefilter._extraer_salario_minimo("1500 USD") == 1500.0


def test_es_ubicacion_valida():
    assert prefilter._es_ubicacion_valida("Remoto", "") is True
    assert prefilter._es_ubicacion_valida("Presencial", "Quito") is True
    assert prefilter._es_ubicacion_valida("Presencial", "Guayaquil") is False
    assert prefilter._es_ubicacion_valida("", "") is True


def test_cumple_prefiltro_data_engineer_senior():
    row = {
        "titulo": "Senior Data Engineer",
        "empresa": "Tech Corp",
        "modalidad": "Remoto",
        "ubicacion": "",
        "salario": "",
        "descripcion": "Buscamos Senior Data Engineer con Python, SQL y AWS.",
    }
    assert prefilter._cumple_prefiltro(row, salario_minimo=2000) is True


def test_cumple_prefiltro_junior_descartado():
    row = {
        "titulo": "Junior Data Analyst",
        "empresa": "Tech Corp",
        "modalidad": "Remoto",
        "ubicacion": "",
        "salario": "",
        "descripcion": "Analista junior con Excel.",
    }
    assert prefilter._cumple_prefiltro(row, salario_minimo=2000) is False


def test_cumple_prefiltro_english_descartado():
    row = {
        "titulo": "Data Engineer",
        "empresa": "Tech Corp",
        "modalidad": "Remoto",
        "ubicacion": "",
        "salario": "",
        "descripcion": "Python, SQL. English required.",
    }
    assert prefilter._cumple_prefiltro(row, salario_minimo=2000) is False


def test_motivo_descarte_data_engineer():
    row = {
        "titulo": "Senior Data Engineer",
        "empresa": "Tech Corp",
        "modalidad": "Presencial",
        "ubicacion": "Lima",
        "salario": "",
        "descripcion": "Python, SQL.",
    }
    motivos = prefilter._motivo_descarte(row, salario_minimo=2000)
    assert "ubicacion" in motivos


if __name__ == "__main__":
    test_quitar_acentos()
    test_extraer_salario_minimo()
    test_es_ubicacion_valida()
    test_cumple_prefiltro_data_engineer_senior()
    test_cumple_prefiltro_junior_descartado()
    test_cumple_prefiltro_english_descartado()
    test_motivo_descarte_data_engineer()
    print("Todas las pruebas pasaron.")
