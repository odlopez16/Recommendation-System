from typing import Optional
from pydantic import BaseModel, UUID4, Field, HttpUrl, constr
from datetime import datetime


class Product(BaseModel):
    """
    Modelo Pydantic para representar productos en el sistema.
    
    Este modelo define la estructura y validación de datos para los productos
    en el sistema de recomendaciones. Incluye validaciones para asegurar
    la integridad y consistencia de los datos.
    
    Atributos:
        id (UUID4): Identificador único del producto
        name (str): Nombre del producto (1-100 caracteres)
        price (float): Precio del producto (debe ser positivo)
        description (str): Descripción detallada del producto (1-1000 caracteres)
        image (Optional[HttpUrl]): URL opcional de la imagen del producto
        created_at (datetime): Fecha y hora de creación del producto
        category (str): Categoría del producto (1-50 caracteres)
    
    Ejemplo:
        >>> producto = Product(
        ...     id="123e4567-e89b-12d3-a456-426614174000",
        ...     name="Laptop Gaming",
        ...     price=999.99,
        ...     description="Laptop gaming de alta potencia",
        ...     category="electronics"
        ... )
    """
    id: UUID4
    name: str = Field(..., min_length=1, max_length=100, description="Nombre del producto")
    price: float = Field(..., ge=0, description="Precio del producto")
    description: str = Field(..., min_length=1, max_length=1000, description="Descripción del producto")
    image: Optional[str]
    created_at: datetime = Field(description="Fecha de creación")
    category: str = Field(..., min_length=1, max_length=50, description="Categoría del producto")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "name": "Ejemplo de Producto",
                "price": 99.99,
                "description": "Descripción detallada del producto",
                "image": "https://ejemplo.com/imagen.jpg",
                "category": "electronics"
            }
        }