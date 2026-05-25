# ============================================================
#  Dockerfile — Modelo de Predicción de Estado de Enfermedad
#  Universidad Icesi · Maestría en Inteligencia Artificial Aplicada
# ============================================================

# 1. Imagen base liviana de Python
FROM python:3.12-slim

# 2. Metadatos
LABEL maintainer="MLOps"
LABEL description="Servicio de predicción de enfermedad — Pipeline de MLOps"
LABEL version="2.0.0"

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiar dependencias e instalarlas (capa cacheada por Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# 5. Copiar código fuente de la aplicación y tests
COPY main.py .
COPY tests/ tests/

# 6. Exponer el puerto del servicio
EXPOSE 8000

# 7. Comando de inicio: uvicorn sirve la app FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
