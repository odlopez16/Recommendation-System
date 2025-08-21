import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, RateLimitError, APIError
from tenacity import retry, wait_random_exponential, stop_after_attempt
from api.models.products_model import Product
from config import config
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
    timeout=20.0,  # Recommended timeout (seconds)
    max_retries=2  # Automatic internal retries of the SDK
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def build_answer(user_query: str, recommended_products: list[Product]) -> str | None:
    """
    Generates a product recommendation response using an OpenAI language model.

    Args:
        user_query (str): User's search query.
        recommended_products (list): List of recommended Product objects.

    Returns:
        str | None: Natural language generated response or None if an error occurs.
    """
    products_list: str = "\n".join(
        f"- {p.name}: {p.description} {p.price}"
        for p in recommended_products
    )
    sys_role = "You are an AI assistant specialized in product recommendations."

    prompt = f"""
        A user has made the following query: "{user_query}"

        Here is a list of available products:
        {products_list}

        Use only the products from the list to answer the user's query. Do not invent products or additional information. Respond in English if the products are in English, otherwise in Spanish, in a clear and helpful manner, explaining why the recommended products are suitable for the query according to their respective descriptions.
    """
    try:
        response = client.chat.completions.create(
            model="radiance",
            messages=[
                {"role": "system", "content": sys_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300  # Limits the response to avoid excessive token usage
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        logger.warning("Rate limit reached")
        raise  # Will be retried by tenacity
    except APIError as e:
        logger.error("OpenAI API error")
    except Exception as e:
        logger.error("Unexpected error")
    return None