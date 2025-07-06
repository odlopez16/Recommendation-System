from config import config

# Configuración de sesiones
class Settings:
    # Límite de sesiones activas por usuario
    MAX_SESSIONS_PER_USER = config.MAX_SESSIONS_PER_USER

# Instancia de configuración para usar en la aplicación
settings = Settings()