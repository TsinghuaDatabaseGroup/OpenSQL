import functools
import random
import time

import openai
from openai import OpenAI

# --- LLM Stuff ---

SYSTEM_MSG = "You are an SQL expert, very skilled at understanding natural language, database schemas, and generating SQL queries."
CLIENT = OpenAI(api_key="<YOUR_API_KEY>", base_url="https://api.deepseek.com")
OPENAI_POTENTIAL_ERRORS = (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError)


def retry_with_exponential_backoff(errors: tuple, initial_delay: float = 10, exponential_base: float = 2, jitter: bool = True, max_retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    num_retries += 1
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.") from None
                    delay *= exponential_base * (1 + jitter * random.random())
                    time.sleep(delay)
                except Exception as e:
                    raise e

        return wrapper

    return decorator


def deepseek_completion_json(
    prompt: str,
    model: str = "deepseek-reasoner",
    system_msg: str = SYSTEM_MSG,
    temperature: float = 0,
    top_p: float = 0.95,
    max_tokens: int = 20480,
) -> str:
    assert model.startswith("deepseek")

    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


@retry_with_exponential_backoff(OPENAI_POTENTIAL_ERRORS)
def deepseek_completion_json_with_backoff(*args, **kwargs) -> str:
    return deepseek_completion_json(*args, **kwargs)
