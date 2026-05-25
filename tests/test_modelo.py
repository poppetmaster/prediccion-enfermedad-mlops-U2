"""
Tests unitarios para el modelo de predicción de enfermedad.
Universidad Icesi — Taller 2 de MLOps

Ejecutar:  pytest tests/test_modelo.py -v
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

# Usar un archivo de predicciones temporal para no contaminar datos reales
os.environ["PREDICTIONS_FILE"] = "predicciones_test.json"

from main import app, predecir_estado, DatosPaciente, obtener_estadisticas, PREDICTIONS_FILE


client = TestClient(app)


# ─────────────────────────────────────────────
# Fixture: limpiar archivo de predicciones antes de cada test
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def limpiar_predicciones():
    """Elimina el archivo de predicciones antes y después de cada test."""
    if os.path.exists(PREDICTIONS_FILE):
        os.remove(PREDICTIONS_FILE)
    yield
    if os.path.exists(PREDICTIONS_FILE):
        os.remove(PREDICTIONS_FILE)


# ─────────────────────────────────────────────
# TEST 1: El modelo retorna las 5 categorías
# ─────────────────────────────────────────────

class TestCincoEstados:
    """Verifica que la función de predicción puede retornar cada uno de
    los 5 estados posibles con los parámetros adecuados."""

    def test_no_enfermo(self):
        datos = DatosPaciente(
            num_sintomas=1, dias_sintomas=1, nivel_dolor=1,
            tiene_fiebre=False, enfermedad_base=False, edad=30,
        )
        resultado = predecir_estado(datos)
        assert resultado["estado"] == "NO ENFERMO"
        assert resultado["puntaje"] < 20

    def test_enfermedad_leve(self):
        datos = DatosPaciente(
            num_sintomas=4, dias_sintomas=5, nivel_dolor=4,
            tiene_fiebre=False, enfermedad_base=False, edad=35,
        )
        resultado = predecir_estado(datos)
        assert resultado["estado"] == "ENFERMEDAD LEVE"
        assert 20 <= resultado["puntaje"] < 45

    def test_enfermedad_aguda(self):
        datos = DatosPaciente(
            num_sintomas=7, dias_sintomas=10, nivel_dolor=6,
            tiene_fiebre=True, enfermedad_base=False, edad=40,
        )
        resultado = predecir_estado(datos)
        assert resultado["estado"] == "ENFERMEDAD AGUDA"
        assert 45 <= resultado["puntaje"] < 65

    def test_enfermedad_cronica(self):
        datos = DatosPaciente(
            num_sintomas=10, dias_sintomas=45, nivel_dolor=8,
            tiene_fiebre=False, enfermedad_base=True, edad=55,
        )
        resultado = predecir_estado(datos)
        assert resultado["estado"] == "ENFERMEDAD CRÓNICA"
        assert 65 <= resultado["puntaje"] < 85

    def test_enfermedad_terminal(self):
        datos = DatosPaciente(
            num_sintomas=18, dias_sintomas=90, nivel_dolor=10,
            tiene_fiebre=True, enfermedad_base=True, edad=75,
        )
        resultado = predecir_estado(datos)
        assert resultado["estado"] == "ENFERMEDAD TERMINAL"
        assert resultado["puntaje"] >= 85


# ─────────────────────────────────────────────
# TEST 2: Estadísticas vacías sin predicciones
# ─────────────────────────────────────────────

class TestEstadisticasVacias:
    """Verifica que el endpoint /stats retorna datos correctos
    cuando no se han realizado predicciones."""

    def test_stats_vacias_funcion(self):
        stats = obtener_estadisticas()
        assert stats["total_predicciones"] == 0
        assert stats["ultima_prediccion"] is None
        assert stats["ultimas_5_predicciones"] == []
        for estado, conteo in stats["predicciones_por_estado"].items():
            assert conteo == 0

    def test_stats_vacias_endpoint(self):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predicciones"] == 0
        assert data["ultima_prediccion"] is None


# ─────────────────────────────────────────────
# TEST 3: Estadísticas se actualizan al predecir
# ─────────────────────────────────────────────

class TestEstadisticasActualizadas:
    """Verifica que las estadísticas se actualizan correctamente
    al registrar predicciones a través de la API."""

    def test_una_prediccion_actualiza_stats(self):
        payload = {
            "num_sintomas": 1, "dias_sintomas": 1, "nivel_dolor": 1,
            "tiene_fiebre": False, "enfermedad_base": False, "edad": 30,
        }
        resp = client.post("/api/predecir", json=payload)
        assert resp.status_code == 200
        assert resp.json()["estado"] == "NO ENFERMO"

        stats = client.get("/stats").json()
        assert stats["total_predicciones"] == 1
        assert stats["predicciones_por_estado"]["NO ENFERMO"] == 1
        assert stats["ultima_prediccion"] is not None

    def test_multiples_predicciones_actualizan_conteo(self):
        casos = [
            {"num_sintomas": 1, "dias_sintomas": 1, "nivel_dolor": 1,
             "tiene_fiebre": False, "enfermedad_base": False, "edad": 30},
            {"num_sintomas": 4, "dias_sintomas": 5, "nivel_dolor": 4,
             "tiene_fiebre": False, "enfermedad_base": False, "edad": 35},
            {"num_sintomas": 18, "dias_sintomas": 90, "nivel_dolor": 10,
             "tiene_fiebre": True, "enfermedad_base": True, "edad": 75},
        ]
        for caso in casos:
            client.post("/api/predecir", json=caso)

        stats = client.get("/stats").json()
        assert stats["total_predicciones"] == 3
        assert stats["predicciones_por_estado"]["NO ENFERMO"] == 1
        assert stats["predicciones_por_estado"]["ENFERMEDAD LEVE"] == 1
        assert stats["predicciones_por_estado"]["ENFERMEDAD TERMINAL"] == 1

    def test_ultimas_5_predicciones(self):
        payload = {
            "num_sintomas": 1, "dias_sintomas": 1, "nivel_dolor": 1,
            "tiene_fiebre": False, "enfermedad_base": False, "edad": 30,
        }
        for _ in range(7):
            client.post("/api/predecir", json=payload)

        stats = client.get("/stats").json()
        assert stats["total_predicciones"] == 7
        assert len(stats["ultimas_5_predicciones"]) == 5


# ─────────────────────────────────────────────
# TEST 4: Validaciones adicionales
# ─────────────────────────────────────────────

class TestValidaciones:
    """Validaciones complementarias del servicio."""

    def test_health_check(self):
        resp = client.get("/api/salud")
        assert resp.status_code == 200
        assert resp.json()["estado"] == "activo"
        assert resp.json()["version"] == "2.0.0"

    def test_pagina_web_carga(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Predicción de Estado de Enfermedad" in resp.text

    def test_prediccion_retorna_campos_esperados(self):
        datos = DatosPaciente(
            num_sintomas=3, dias_sintomas=5, nivel_dolor=4,
            tiene_fiebre=False, enfermedad_base=False, edad=30,
        )
        resultado = predecir_estado(datos)
        assert "estado" in resultado
        assert "puntaje" in resultado
        assert "descripcion" in resultado
        assert "parametros_recibidos" in resultado

    def test_puntaje_dentro_de_rango(self):
        datos = DatosPaciente(
            num_sintomas=20, dias_sintomas=365, nivel_dolor=10,
            tiene_fiebre=True, enfermedad_base=True, edad=80,
        )
        resultado = predecir_estado(datos)
        assert 0 <= resultado["puntaje"] <= 100

    def test_api_valida_parametros_invalidos(self):
        payload = {"num_sintomas": -5, "dias_sintomas": 1, "nivel_dolor": 1}
        resp = client.post("/api/predecir", json=payload)
        assert resp.status_code == 422  # Validation error
