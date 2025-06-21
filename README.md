# Recommendation System API

Este proyecto es un sistema de recomendación de productos desarrollado con **FastAPI**, que utiliza embeddings, autenticación JWT y una arquitectura modular para gestionar usuarios, productos y recomendaciones inteligentes.

## Características principales

- **API RESTful** construida con FastAPI.
- **Autenticación JWT** para rutas protegidas.
- **Embeddings** de productos usando OpenAI y FAISS para recomendaciones.
- **Gestión de usuarios** con registro, login y consulta de usuario actual.
- **Persistencia** en PostgreSQL usando SQLAlchemy y Databases.
- **Migraciones** con Alembic.
- **Logging** centralizado y configurable.
- **Configuración** por entorno usando ".env" y Pydantic Settings.

## Estructura del proyecto

"""
Recommendation-System/
├── alembic/                  # Migraciones de base de datos
├── api/
│   ├── main.py               # Punto de entrada FastAPI
│   ├── database/             # Configuración y conexión a bases de datos
│   ├── models/               # Modelos Pydantic
│   ├── routers/              # Routers de FastAPI (auth, embeddings)
│   ├── schemas/              # Esquemas SQLAlchemy
│   ├── services/             # Lógica de negocio (auth, embeddings, llm, etc.)
│   └── requirements.txt      # Dependencias Python
├── config.py                 # Configuración global y de entornos
├── logging_config.py         # Configuración de logging
├── .env                      # Variables de entorno
├── README.md                 # Este archivo
└── ...
"""

## Instalación y configuración

1. **Clona el repositorio y entra al directorio:**
   """bash
   git clone <repo_url>
   cd Recommendation-System
   """
2. **Crea y activa un entorno virtual:**
   """bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   """
3. **Instala las dependencias:**
   """bash
   pip install -r api/requirements.txt
   """
4. **Configura el archivo ".env":**
   - Ajusta las URLs de base de datos, claves y parámetros según tu entorno.

5. **Aplica las migraciones:**
   """bash
   alembic upgrade head
   """

## Ejecución

"""bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 4000
"""

La API estará disponible en: [http://127.0.0.1:4000/docs](http://127.0.0.1:4000/docs)

## Endpoints principales

### Autenticación
- "POST /api/auth/register" — Registro de usuario
- "POST /api/auth/login" — Login y obtención de JWT
- "GET /api/auth/me" — Usuario autenticado actual (requiere JWT)

### Recomendaciones
- "POST /api/embeddings/recommendation" — Obtener recomendación de productos (requiere JWT)

## Seguridad
- Usa JWT para proteger rutas sensibles.
- El token debe enviarse en el header: "Authorization: Bearer <token>"

## Variables de entorno principales (".env")
- "ENV_STATE" — Entorno actual ("dev", "test", "prod")
- "DEV_POSTGRES_URL_PRIMARY" — URL de la base de datos de embeddings y usuarios
- "DEV_POSTGRES_URL_SECONDARY" — URL de la base de datos de productos
- "JWT_SECRET_KEY", "ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES" — Seguridad JWT
- "OPENAI_MODEL", "API_KEY", "BASE_URL" — Configuración de embeddings

## Migraciones y bases de datos
- Las tablas "users_table" y "embeddings_table" se crean en la base de datos de embeddings.
- La tabla "products_table" se crea en la base de datos de productos.
- Revisa y ajusta los strings de conexión en ".env" según tu entorno.

## Licencia
MIT

---

**Desarrollado para fines académicos y de investigación.**
