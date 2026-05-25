# Modelo de Predicción de Estado de Enfermedad

**Universidad Icesi — Maestría en Inteligencia Artificial Aplicada**
**Pipeline de MLOps — Taller # 2**

## Integrantes del equipo
| Nombre | Correo | Usuario |
|:-------|:--------|:--------|
| [Rubén Darío Sabogal Urbano](https://github.com/rubenesticesi) | 16704992@u.icesi.edu.co| @rubenesticesi|
| [Cristian Quebrada](https://github.com/cris-bytes) | criistianq90@gmail.com| @cris-bytes|
| [Edwin  Perez Lozano](https://github.com/poppetmaster) | edwinandperez@gmail.com| @poppetmaster|

---

## 1. Descripción

Servicio que simula un modelo de ML para clasificar el estado de salud de un
paciente a partir de sus síntomas. Construido con **Python + FastAPI** y
empaquetado en **Docker**.

### Estados que retorna el modelo

| Estado               | Puntaje de riesgo | Descripción                            |
|----------------------|-------------------|----------------------------------------|
| NO ENFERMO           | 0 – 19            | Sin indicios de enfermedad             |
| ENFERMEDAD LEVE      | 20 – 44           | Condición leve, monitoreo ambulatorio  |
| ENFERMEDAD AGUDA     | 45 – 64           | Atención médica prioritaria            |
| ENFERMEDAD CRÓNICA   | 65 – 84           | Evaluación especializada continua      |
| ENFERMEDAD TERMINAL  | 85 – 100          | Múltiples factores de gravedad extrema |

> **Nuevo en Taller 2:** La categoría **ENFERMEDAD TERMINAL** se activa cuando
> coinciden edad avanzada, enfermedad de base, muchos síntomas, larga duración
> y dolor alto (puntaje ≥ 85).

### Parámetros de entrada

| Parámetro          | Tipo  | Rango   | Requerido | Descripción                   |
|--------------------|-------|---------|-----------|-------------------------------|
| `num_sintomas`     | int   | 0-20    | Sí        | Número de síntomas reportados |
| `dias_sintomas`    | int   | 0-365   | Sí        | Días con síntomas activos     |
| `nivel_dolor`      | int   | 0-10    | Sí        | Nivel de dolor (escala 0-10)  |
| `tiene_fiebre`     | bool  | —       | No        | ¿Presenta fiebre?             |
| `enfermedad_base`  | bool  | —       | No        | ¿Enfermedad preexistente?     |
| `edad`             | int   | 0-120   | No        | Edad del paciente en años     |

---

## 2. Requisitos previos

- **Docker Desktop** instalado y en ejecución.
  - Descarga: https://www.docker.com/products/docker-desktop/
  - Verificar instalación: `docker --version`

---

## 3. Estructura del proyecto

```
02_servicio_docker/
├── main.py              ← Aplicación FastAPI + función de predicción + registro
├── requirements.txt     ← Dependencias de Python
├── Dockerfile           ← Instrucciones para construir la imagen
├── .dockerignore        ← Archivos excluidos del contexto de Docker
├── README.md            ← Este archivo
└── tests/
    └── test_modelo.py   ← Pruebas unitarias con pytest
```

---

## 4. Construir la imagen de Docker

```bash
docker build -t modelo-prediccion .
```

Verificar que la imagen se creó:

```bash
docker images
```

---

## 5. Ejecutar el contenedor

```bash
docker run -d -p 8000:8000 --name prediccion modelo-prediccion
```

| Flag               | Significado                                 |
|--------------------|---------------------------------------------|
| `-d`               | Ejecuta en segundo plano (detached)         |
| `-p 8000:8000`     | Mapea el puerto 8000 del contenedor al host |
| `--name prediccion`| Nombre amigable para el contenedor          |

Verificar que está corriendo:

```bash
docker ps
```

---

## 6. Usar el servicio

### 6.1 — Interfaz web (recomendada para el médico)

```
http://localhost:8000
```

### 6.2 — Documentación interactiva (Swagger UI)

```
http://localhost:8000/docs
```

### 6.3 — Endpoint de estadísticas (nuevo en Taller 2)

```
http://localhost:8000/stats
```

Retorna:

```json
{
  "total_predicciones": 10,
  "predicciones_por_estado": {
    "NO ENFERMO": 3,
    "ENFERMEDAD LEVE": 2,
    "ENFERMEDAD AGUDA": 2,
    "ENFERMEDAD CRÓNICA": 2,
    "ENFERMEDAD TERMINAL": 1
  },
  "ultimas_5_predicciones": [ ... ],
  "ultima_prediccion": "2026-05-13T21:30:00+00:00"
}
```

---

## 7. Ejemplos de predicción para cada estado

### NO ENFERMO

```json
{ "num_sintomas": 1, "dias_sintomas": 1, "nivel_dolor": 1, "tiene_fiebre": false, "enfermedad_base": false, "edad": 30 }
```

### ENFERMEDAD LEVE

```json
{ "num_sintomas": 4, "dias_sintomas": 5, "nivel_dolor": 4, "tiene_fiebre": false, "enfermedad_base": false, "edad": 35 }
```

### ENFERMEDAD AGUDA

```json
{ "num_sintomas": 7, "dias_sintomas": 10, "nivel_dolor": 6, "tiene_fiebre": true, "enfermedad_base": false, "edad": 40 }
```

### ENFERMEDAD CRÓNICA

```json
{ "num_sintomas": 10, "dias_sintomas": 45, "nivel_dolor": 8, "tiene_fiebre": false, "enfermedad_base": true, "edad": 55 }
```

### ENFERMEDAD TERMINAL (nuevo)

```json
{ "num_sintomas": 18, "dias_sintomas": 90, "nivel_dolor": 10, "tiene_fiebre": true, "enfermedad_base": true, "edad": 75 }
```

---

## 8. Ejecutar pruebas unitarias

### Opción A — Dentro del contenedor (sin instalar nada local)

```bash
docker exec prediccion pytest tests/test_modelo.py -v
```

### Opción B — Localmente (requiere Python instalado)

```bash
pip install -r requirements.txt
pytest tests/test_modelo.py -v
```

### Resultado esperado

```
tests/test_modelo.py::TestCincoEstados::test_no_enfermo          PASSED
tests/test_modelo.py::TestCincoEstados::test_enfermedad_leve     PASSED
tests/test_modelo.py::TestCincoEstados::test_enfermedad_aguda    PASSED
tests/test_modelo.py::TestCincoEstados::test_enfermedad_cronica  PASSED
tests/test_modelo.py::TestCincoEstados::test_enfermedad_terminal PASSED
tests/test_modelo.py::TestEstadisticasVacias::...                PASSED
tests/test_modelo.py::TestEstadisticasActualizadas::...          PASSED
...
```

---

## 9. Detener y eliminar el contenedor

```bash
docker stop prediccion
docker rm prediccion
docker rmi modelo-prediccion
```

---

## 10. Cambios respecto al Taller 1

| Cambio                        | Detalle                                                    |
|-------------------------------|------------------------------------------------------------|
| Nueva categoría               | ENFERMEDAD TERMINAL (puntaje ≥ 85)                         |
| Registro de predicciones      | Cada predicción se guarda en `predicciones.json`           |
| Endpoint `/stats`             | Estadísticas: totales, conteo por estado, últimas 5        |
| Pruebas unitarias             | `tests/test_modelo.py` con pytest (15 tests)               |
| Dependencias nuevas           | `pytest` y `httpx` agregados a requirements.txt            |
| Ajuste de umbrales            | AGUDA: 45-64, CRÓNICA: 65-84, TERMINAL: 85-100            |

---

## 11. Notas importantes

- La función de predicción utiliza un sistema de puntaje basado en reglas
  clínicas. En un entorno real, sería reemplazada por un modelo de ML entrenado
  con datos clínicos reales.
- El servicio corre en `http://localhost:8000` y no requiere conexión a internet
  una vez construida la imagen.
