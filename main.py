"""
Pipeline de MLOps — Modelo de Predicción de Estado de Enfermedad
Universidad Icesi · Maestría en Inteligencia Artificial Aplicada

Servicio web construido con FastAPI que simula un modelo de ML capaz de
clasificar el estado de salud de un paciente a partir de sus síntomas.

Estados posibles:
    ● NO ENFERMO
    ● ENFERMEDAD LEVE
    ● ENFERMEDAD AGUDA
    ● ENFERMEDAD CRÓNICA

Autores :   Cristian Camilo Quebrada
            Ruben Dario Sabogal
            Edwin Perez L

Fecha : 13 de mayo de 2026
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────
# 1.  ESQUEMA DE DATOS DE ENTRADA (Pydantic)
# ─────────────────────────────────────────────

class DatosPaciente(BaseModel):
    """Datos clínicos que el médico ingresa para obtener la predicción."""

    num_sintomas: int = Field(
        ..., ge=0, le=20,
        description="Número de síntomas reportados (0-20)",
        json_schema_extra={"example": 5},
    )
    dias_sintomas: int = Field(
        ..., ge=0, le=365,
        description="Días con síntomas activos (0-365)",
        json_schema_extra={"example": 7},
    )
    nivel_dolor: int = Field(
        ..., ge=0, le=10,
        description="Nivel de dolor en escala 0-10",
        json_schema_extra={"example": 6},
    )
    tiene_fiebre: Optional[bool] = Field(
        default=False,
        description="¿Presenta fiebre?",
    )
    enfermedad_base: Optional[bool] = Field(
        default=False,
        description="¿Tiene enfermedad preexistente?",
    )
    edad: Optional[int] = Field(
        default=30, ge=0, le=120,
        description="Edad del paciente en años",
    )


# ─────────────────────────────────────────────
# 2.  FUNCIÓN DE PREDICCIÓN (modelo simulado)
# ─────────────────────────────────────────────

def predecir_estado(datos: DatosPaciente) -> dict:
    """
    Simula la predicción del estado de enfermedad del paciente.

    Aplica un sistema de puntaje basado en reglas clínicas sobre los
    parámetros recibidos (mínimo 3 obligatorios: num_sintomas,
    dias_sintomas y nivel_dolor) y clasifica al paciente en uno de
    los cuatro estados definidos.

    Retorna
    -------
    dict  con  estado · puntaje · descripcion · parametros_recibidos
    """

    puntaje = 0.0

    # ── Factor 1: cantidad de síntomas (peso 25 %) ──────────────
    puntaje += min(datos.num_sintomas / 10, 1.0) * 25

    # ── Factor 2: duración de los síntomas (peso 20 %) ──────────
    if datos.dias_sintomas <= 2:
        puntaje += 4
    elif datos.dias_sintomas <= 7:
        puntaje += 10
    elif datos.dias_sintomas <= 30:
        puntaje += 16
    else:
        puntaje += 20  # cronicidad

    # ── Factor 3: intensidad del dolor (peso 25 %) ──────────────
    puntaje += (datos.nivel_dolor / 10) * 25

    # ── Factor 4: fiebre (peso 10 %) ────────────────────────────
    if datos.tiene_fiebre:
        puntaje += 10

    # ── Factor 5: enfermedad de base (peso 10 %) ────────────────
    if datos.enfermedad_base:
        puntaje += 10

    # ── Factor 6: edad — extremos = mayor riesgo (peso 10 %) ───
    if datos.edad < 5 or datos.edad > 65:
        puntaje += 10
    elif datos.edad < 12 or datos.edad > 50:
        puntaje += 5

    puntaje = max(0.0, min(puntaje, 100.0))

    # ── Clasificación ───────────────────────────────────────────
    if puntaje < 20:
        estado = "NO ENFERMO"
        descripcion = (
            "Los parámetros clínicos del paciente no sugieren presencia "
            "de enfermedad significativa. Se recomienda seguimiento "
            "preventivo de rutina."
        )
    elif puntaje < 45:
        estado = "ENFERMEDAD LEVE"
        descripcion = (
            "Los parámetros clínicos sugieren una condición leve. "
            "Se recomienda monitoreo ambulatorio y tratamiento según "
            "criterio del médico tratante."
        )
    elif puntaje < 70:
        estado = "ENFERMEDAD AGUDA"
        descripcion = (
            "Los parámetros clínicos indican una condición aguda que "
            "requiere atención médica prioritaria. Se recomienda "
            "evaluación clínica completa."
        )
    else:
        estado = "ENFERMEDAD CRÓNICA"
        descripcion = (
            "Los parámetros clínicos sugieren una condición crónica o "
            "de alto riesgo. Se recomienda evaluación especializada y "
            "seguimiento continuo."
        )

    return {
        "estado": estado,
        "puntaje": round(puntaje, 2),
        "descripcion": descripcion,
        "parametros_recibidos": datos.model_dump(),
    }


# ─────────────────────────────────────────────
# 3.  APLICACIÓN FASTAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="Modelo de Predicción de Enfermedad",
    description=(
        "Servicio que simula un modelo de ML para clasificar el estado "
        "de salud de un paciente.  \n"
        "**Universidad Icesi — Pipeline de MLOps**"
    ),
    version="1.0.0",
)


# ── 3-a  Página web (formulario para el médico) ────────────────

PAGINA_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Predicción de Enfermedad — Icesi</title>
<style>
  :root {
    --azul: #003366; --acento: #2E86AB;
    --verde: #27ae60; --amarillo: #f39c12;
    --naranja: #e67e22; --rojo: #c0392b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f0f4f8; color: #333;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 30px 16px;
  }
  .header {
    text-align: center; margin-bottom: 28px;
  }
  .header h1 {
    color: var(--azul); font-size: 1.6rem; margin-bottom: 4px;
  }
  .header p { color: #666; font-size: .9rem; }
  .card {
    background: #fff; border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,.08);
    width: 100%; max-width: 520px; padding: 32px; margin-bottom: 24px;
  }
  .card h2 {
    color: var(--azul); font-size: 1.15rem;
    margin-bottom: 20px; border-bottom: 2px solid var(--acento);
    padding-bottom: 8px;
  }
  label {
    display: block; font-weight: 600; font-size: .85rem;
    color: #444; margin-bottom: 4px; margin-top: 14px;
  }
  input[type=number] {
    width: 100%; padding: 10px 12px; border: 1px solid #ccd;
    border-radius: 8px; font-size: .95rem; transition: border .2s;
  }
  input[type=number]:focus { border-color: var(--acento); outline: none; }
  .checks {
    display: flex; gap: 24px; margin-top: 16px;
  }
  .checks label {
    display: flex; align-items: center; gap: 6px;
    font-weight: 500; cursor: pointer;
  }
  .checks input[type=checkbox] {
    width: 18px; height: 18px; accent-color: var(--acento);
  }
  button {
    margin-top: 24px; width: 100%; padding: 13px;
    background: var(--azul); color: #fff; border: none;
    border-radius: 8px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background .2s;
  }
  button:hover { background: var(--acento); }

  /* ── resultado ── */
  .resultado {
    text-align: center; padding: 24px;
  }
  .resultado .badge {
    display: inline-block; padding: 10px 28px; border-radius: 30px;
    font-size: 1.15rem; font-weight: 700; color: #fff;
    margin-bottom: 12px;
  }
  .badge.no-enfermo   { background: var(--verde);   }
  .badge.leve         { background: var(--amarillo); }
  .badge.aguda        { background: var(--naranja);  }
  .badge.cronica      { background: var(--rojo);     }
  .resultado .score {
    font-size: 2rem; font-weight: 700; color: var(--azul);
  }
  .resultado .desc {
    margin-top: 10px; font-size: .9rem; color: #555; line-height: 1.5;
  }
  .resultado .params {
    margin-top: 16px; font-size: .8rem; color: #888;
    background: #f8f9fa; border-radius: 8px; padding: 12px;
    text-align: left;
  }
  .aviso {
    margin-top: 14px; font-size: .78rem; color: #999;
    text-align: center; font-style: italic;
  }
  footer {
    margin-top: auto; padding-top: 20px;
    font-size: .75rem; color: #aaa; text-align: center;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Predicción de Estado de Enfermedad</h1>
  <p>Universidad Icesi · Pipeline de MLOps · Taller #1</p>
</div>

<!-- FORMULARIO -->
<div class="card">
  <h2>Datos del Paciente</h2>
  <form method="post" action="/predecir">

    <label for="edad">Edad del paciente</label>
    <input type="number" id="edad" name="edad" min="0" max="120"
           value="{{edad}}" required placeholder="Ej: 45"/>

    <label for="num_sintomas">Número de síntomas reportados (0-20)</label>
    <input type="number" id="num_sintomas" name="num_sintomas"
           min="0" max="20" value="{{num_sintomas}}" required placeholder="Ej: 5"/>

    <label for="dias_sintomas">Días con síntomas activos (0-365)</label>
    <input type="number" id="dias_sintomas" name="dias_sintomas"
           min="0" max="365" value="{{dias_sintomas}}" required placeholder="Ej: 7"/>

    <label for="nivel_dolor">Nivel de dolor (0-10)</label>
    <input type="number" id="nivel_dolor" name="nivel_dolor"
           min="0" max="10" value="{{nivel_dolor}}" required placeholder="Ej: 6"/>

    <div class="checks">
      <label><input type="checkbox" name="tiene_fiebre"
             {{"checked" if tiene_fiebre else ""}}/> Tiene fiebre</label>
      <label><input type="checkbox" name="enfermedad_base"
             {{"checked" if enfermedad_base else ""}}/> Enfermedad de base</label>
    </div>

    <button type="submit">Obtener Predicción</button>
  </form>
</div>

<!-- RESULTADO -->
{% if resultado %}
<div class="card resultado">
  <h2>Resultado de la Predicción</h2>

  {% set cls = "no-enfermo" if resultado.estado == "NO ENFERMO"
          else "leve"       if resultado.estado == "ENFERMEDAD LEVE"
          else "aguda"      if resultado.estado == "ENFERMEDAD AGUDA"
          else "cronica" %}

  <span class="badge {{cls}}">{{ resultado.estado }}</span>
  <div class="score">{{ resultado.puntaje }} / 100</div>
  <p class="desc">{{ resultado.descripcion }}</p>

  <div class="params">
    <strong>Parámetros evaluados:</strong><br/>
    Edad: {{ resultado.parametros_recibidos.edad }} años ·
    Síntomas: {{ resultado.parametros_recibidos.num_sintomas }} ·
    Días: {{ resultado.parametros_recibidos.dias_sintomas }} ·
    Dolor: {{ resultado.parametros_recibidos.nivel_dolor }}/10 ·
    Fiebre: {{ "Sí" if resultado.parametros_recibidos.tiene_fiebre else "No" }} ·
    Enf. base: {{ "Sí" if resultado.parametros_recibidos.enfermedad_base else "No" }}
  </div>
</div>
{% endif %}

<footer>Universidad Icesi — Maestría en Inteligencia Artificial Aplicada — 2026</footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def pagina_inicio():
    """Renderiza el formulario vacío para el médico."""
    from jinja2 import Template
    tpl = Template(PAGINA_HTML)
    return HTMLResponse(tpl.render(
        edad="", num_sintomas="", dias_sintomas="",
        nivel_dolor="", tiene_fiebre=False,
        enfermedad_base=False, resultado=None,
    ))


@app.post("/predecir", response_class=HTMLResponse)
async def predecir_formulario(
    edad: int = Form(...),
    num_sintomas: int = Form(...),
    dias_sintomas: int = Form(...),
    nivel_dolor: int = Form(...),
    tiene_fiebre: str = Form(default=None),
    enfermedad_base: str = Form(default=None),
):
    """Recibe datos desde el formulario web y devuelve la página con resultado."""
    datos = DatosPaciente(
        num_sintomas=num_sintomas,
        dias_sintomas=dias_sintomas,
        nivel_dolor=nivel_dolor,
        tiene_fiebre=tiene_fiebre is not None,
        enfermedad_base=enfermedad_base is not None,
        edad=edad,
    )
    resultado = predecir_estado(datos)

    from jinja2 import Template
    tpl = Template(PAGINA_HTML)
    return HTMLResponse(tpl.render(
        edad=edad, num_sintomas=num_sintomas,
        dias_sintomas=dias_sintomas, nivel_dolor=nivel_dolor,
        tiene_fiebre=(tiene_fiebre is not None),
        enfermedad_base=(enfermedad_base is not None),
        resultado=resultado,
    ))


# ── API REST (JSON) ────────────────────────────────────────────

@app.post("/api/predecir", summary="Predicción vía API REST")
async def predecir_api(datos: DatosPaciente):
    """
    Recibe un JSON con los datos del paciente y retorna la predicción.

    Ejemplo con curl (Windows PowerShell):
    ```
    curl -X POST http://localhost:8000/api/predecir `
         -H "Content-Type: application/json" `
         -d '{\"num_sintomas\":5,\"dias_sintomas\":10,\"nivel_dolor\":7,\"tiene_fiebre\":true,\"enfermedad_base\":false,\"edad\":45}'
    ```
    """
    return predecir_estado(datos)


@app.get("/api/salud", summary="Health check")
async def salud():
    """Verifica que el servicio esté activo."""
    return {
        "estado": "activo",
        "servicio": "Modelo de Predicción de Enfermedad",
        "version": "1.0.0",
    }
