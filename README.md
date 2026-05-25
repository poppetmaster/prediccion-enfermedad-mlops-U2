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

Este servicio simula un modelo de Machine Learning que clasifica el estado de
salud de un paciente a partir de sus síntomas. Está construido con **Python** y
**FastAPI**, empaquetado en una imagen **Docker**.

### Estados que retorna el modelo

| Estado               | Puntaje de riesgo |
|----------------------|-------------------|
| NO ENFERMO           | 0 – 19            |
| ENFERMEDAD LEVE      | 20 – 44           |
| ENFERMEDAD AGUDA     | 45 – 69           |
| ENFERMEDAD CRÓNICA   | 70 – 100          |

### Parámetros de entrada

El modelo recibe **6 parámetros** (los 3 primeros son obligatorios):

| Parámetro          | Tipo  | Rango   | Descripción                           |
|--------------------|-------|---------|---------------------------------------|
| `num_sintomas`     | int   | 0-20    | Número de síntomas reportados         |
| `dias_sintomas`    | int   | 0-365   | Días con síntomas activos             |
| `nivel_dolor`      | int   | 0-10    | Nivel de dolor (escala 0-10)          |
| `tiene_fiebre`     | bool  | —       | ¿Presenta fiebre? (opcional)          |
| `enfermedad_base`  | bool  | —       | ¿Tiene enfermedad preexistente? (opc) |
| `edad`             | int   | 0-120   | Edad del paciente en años (opcional)  |

---

## 2. Requisitos previos

- **Docker Desktop** instalado y en ejecución.
  - Descarga: https://www.docker.com/products/docker-desktop/
  - Verificar instalación:
    ```
    docker --version
    ```

---

## 3. Estructura del proyecto

```
02_servicio_docker/
├── main.py              ← Aplicación FastAPI + función de predicción
├── requirements.txt     ← Dependencias de Python
├── Dockerfile           ← Instrucciones para construir la imagen
├── .dockerignore        ← Archivos excluidos del contexto de Docker
└── README.md            ← Este archivo
```

---

## 4. Construir la imagen de Docker

Abrir una terminal (**PowerShell** o **CMD**) en la carpeta del proyecto y
ejecutar:

```bash
docker build -t modelo-prediccion .
```

> El punto `.` al final indica que el `Dockerfile` está en el directorio actual.
> La primera vez descargará la imagen base de Python (~150 MB).

Verificar que la imagen se creó correctamente:

```bash
docker images
```

Debe aparecer una fila con el nombre `modelo-prediccion`.

---

## 5. Ejecutar el contenedor

```bash
docker run -d -p 8000:8000 --name prediccion modelo-prediccion
```

| Flag               | Significado                                    |
|--------------------|------------------------------------------------|
| `-d`               | Ejecuta en segundo plano (detached)            |
| `-p 8000:8000`     | Mapea el puerto 8000 del contenedor al host    |
| `--name prediccion`| Nombre amigable para el contenedor             |

Verificar que el contenedor está corriendo:

```bash
docker ps
```

---

## 6. Usar el servicio

### 6.1 — Interfaz web (recomendada para el médico)

Abrir un navegador y visitar:

```
http://localhost:8000
```

Se muestra un formulario donde el médico ingresa los datos del paciente y
obtiene la predicción con un indicador visual de color.

### 6.2 — Documentación interactiva (Swagger UI)

FastAPI genera documentación automática con un botón **"Try it out"** para
probar directamente:

```
http://localhost:8000/docs
```
---

## 7. Ejemplos de predicción para cada estado

A continuación se presentan combinaciones de parámetros que generan **cada uno
de los 4 estados**, demostrando que la función cubre todas las clasificaciones:

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
{ "num_sintomas": 8, "dias_sintomas": 10, "nivel_dolor": 7, "tiene_fiebre": true, "enfermedad_base": false, "edad": 45 }
```

### ENFERMEDAD CRÓNICA

```json
{ "num_sintomas": 12, "dias_sintomas": 60, "nivel_dolor": 9, "tiene_fiebre": true, "enfermedad_base": true, "edad": 70 }
```

---

## 8. Detener y eliminar el contenedor

```bash
# Detener
docker stop prediccion

# Eliminar contenedor
docker rm prediccion

# Eliminar imagen (opcional)
docker rmi modelo-prediccion
```

---

## 9. Notas importantes

- La función de predicción utiliza un sistema de puntaje basado en reglas
  clínicas. En un entorno real, sería reemplazada por un modelo de ML entrenado
  con datos clínicos reales.
- El servicio corre en `http://localhost:8000` y no requiere conexión a internet
  una vez construida la imagen.
