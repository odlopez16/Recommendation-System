from typing import Optional, Literal
from fastapi import APIRouter, status, Depends, Query, HTTPException, Path
from api.services.auth.auth_service import get_current_user
from api.models.auth_model import UserInDB
from api.models.products_model import Product
from api.services.recommendations.product_service import product_processor
from api.exceptions.exceptions import not_found_exception
from typing import Any, Optional, Literal
from fastapi import APIRouter, status, Depends, Query, HTTPException, Path
from api.services.auth.auth_service import get_current_user
from api.models.auth_model import UserInDB
from api.models.products_model import Product
from api.services.recommendations.product_service import product_processor
from pydantic import UUID4, constr
import logging

logger = logging.getLogger("api.routers.products_router")

"""
Router para el manejo de productos en el sistema de recomendaciones.
Incluye endpoints para listar, buscar y filtrar productos.
"""

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/", response_model=list[Product])
async def get_products(
    current_user: UserInDB = Depends(get_current_user),
    skip: int = Query(
        default=0, 
        ge=0, 
        description="Número de productos a saltar para la paginación"
    ),
    limit: int = Query(
        default=10, 
        ge=1,
        le=100, 
        description="Número máximo de productos a retornar"
    ),
    category: Optional[str] = Query(
        default=None, 
        min_length=1,
        max_length=50,
        description="Categoría de productos a filtrar"
    ),
    sort_by: Optional[Literal["name", "price"]] = Query(
        default=None, 
        description="Campo por el cual ordenar: 'name' para nombre, 'price' para precio"
    ),
    order: Optional[Literal["asc", "desc"]] = Query(
        default="asc", 
        description="Orden: 'asc' para ascendente, 'desc' para descendente"
    ),
    search: Optional[str] = Query(
        default=None,
        min_length=1,
        max_length=100, 
        description="Término de búsqueda para filtrar por nombre de producto"
    )
) -> list[Product]:
    """
    Obtiene una lista paginada y ordenada de productos, con opciones de filtrado.
    
    Este endpoint permite obtener productos con varias opciones de filtrado y ordenamiento:
    - Paginación mediante skip y limit
    - Filtrado por categoría
    - Ordenamiento por nombre o precio
    - Búsqueda por término en el nombre del producto
    
    Args:
        current_user (UserInDB): Usuario autenticado que realiza la petición
        skip (int): Número de productos a saltar para la paginación
        limit (int): Número máximo de productos a retornar por página
        category (str, opcional): Categoría específica para filtrar productos
        sort_by (str, opcional): Campo por el cual ordenar ('name' o 'price')
        order (str, opcional): Dirección del ordenamiento ('asc' o 'desc')
        search (str, opcional): Término de búsqueda para filtrar por nombre de producto
    
    Returns:
        List[Product]: Lista de productos que coinciden con los criterios especificados
    
    Raises:
        HTTPException(404): No se encontraron productos con los criterios especificados
        HTTPException(400): Error en los parámetros de la solicitud
        HTTPException(500): Error interno del servidor
    
    Ejemplos:
        >>> GET /products?limit=10&skip=0
        >>> GET /products?category=electronics&sort_by=price&order=desc
        >>> GET /products?search=laptop&sort_by=name&order=asc
    """
    try:
        # Verificar si la base de datos primaria está vacía
        if await product_processor.get_products_from_primary_db() == []:
            await product_processor.migrate_products()
            
        # Obtener productos con filtros
        products = await product_processor.get_products_from_primary_db(
            skip=skip, 
            limit=limit, 
            category=category,
            sort_by=sort_by,
            order=order,
            search=search
        )
        
        # Verificar si se encontraron productos
        if not products:
            raise not_found_exception(detail="No se encontraron productos con los criterios especificados")
            
        return products
        
    except ValueError as ve:
        # Error de validación de datos
        logger.warning(f"Error de validación: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Error interno del servidor
        logger.error(f"Error al obtener productos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al obtener los productos"
        )

@router.get("/{product_id}", response_model=Product, status_code=status.HTTP_200_OK)
async def get_product_by_id(
    product_id: UUID4 = Path(..., description="ID del producto a buscar"),
    current_user: UserInDB = Depends(get_current_user)
) -> Product:
    """
    Obtiene un producto específico por su ID único.
    
    Este endpoint permite obtener los detalles completos de un producto específico
    utilizando su identificador único (UUID). Requiere autenticación.
    
    Args:
        product_id (UUID4): Identificador único del producto a buscar
        current_user (UserInDB): Usuario autenticado que realiza la petición
    
    Returns:
        Product: Objeto con todos los detalles del producto encontrado
        
    Raises:
        HTTPException(404): El producto no existe en la base de datos
        HTTPException(400): El formato del ID proporcionado no es válido
        HTTPException(500): Error interno del servidor durante la búsqueda
        
    Ejemplo:
        >>> GET /products/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        return await product_processor.get_product_by_id(product_id=product_id)
    except HTTPException:
        raise not_found_exception(detail="No se encontró el producto")
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ID de producto inválido: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error al obtener producto")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al obtener el producto"
        )