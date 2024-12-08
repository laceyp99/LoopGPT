# decorators.py

import functools
import logging
from typing import Callable
from pydantic import ValidationError
import openai
from openai import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    Timeout,
    APIError,
    OpenAIError,
)
from code.exceptions import (
    MusicGenerationError,
    APICallError,
    ResponseValidationError,
    NetworkError,
    AuthenticationError as CustomAuthenticationError,
    RateLimitError as CustomRateLimitError,
)

def handle_errors(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except ValidationError as e:
            logging.error(f"Validation error in {func.__name__}: {e}")
            raise ResponseValidationError(f"Validation error: {e}") from e

        except AuthenticationError as e:
            logging.error(f"Authentication error in {func.__name__}: {e}")
            raise CustomAuthenticationError(f"Authentication error: {e}") from e

        except RateLimitError as e:
            logging.error(f"Rate limit error in {func.__name__}: {e}")
            raise CustomRateLimitError(f"Rate limit error: {e}") from e

        except APIConnectionError as e:
            logging.error(f"API connection error in {func.__name__}: {e}")
            raise NetworkError(f"API connection error: {e}") from e

        except Timeout as e:
            logging.error(f"API timeout error in {func.__name__}: {e}")
            raise NetworkError(f"API timeout error: {e}") from e

        except APIError as e:
            logging.error(f"API error in {func.__name__}: {e}")
            raise APICallError(f"API error: {e}") from e

        except OpenAIError as e:
            logging.error(f"OpenAI error in {func.__name__}: {e}")
            raise MusicGenerationError(f"OpenAI error: {e}") from e

        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {e}")
            raise MusicGenerationError(f"An unexpected error occurred: {e}") from e

    return wrapper