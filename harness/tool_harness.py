"""
harness/tool_harness.py
Decorator wrap tool với retry, structured output, logging.
"""

import time
import traceback
import logging
from functools import wraps
from typing import Callable, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tool_harness")


def tool_harness(name: str, max_retries: int = 3, timeout_ms: int = 10000):
    """
    Wrap tool với:
    - Structured output: {status, data, error, latency_ms, tool}
    - Auto retry + exponential backoff
    - Logging tự động
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            attempt = 0
            last_error = None

            while attempt < max_retries:
                start = time.time()
                attempt += 1
                try:
                    result = fn(*args, **kwargs)
                    latency = round((time.time() - start) * 1000)
                    if latency > timeout_ms:
                        logger.warning(f"[{name}] Slow: {latency}ms")
                    logger.info(f"[{name}] OK | {latency}ms")
                    return {
                        "status": "ok",
                        "data": result,
                        "error": None,
                        "latency_ms": latency,
                        "tool": name,
                    }
                except Exception as e:
                    last_error = e
                    logger.warning(f"[{name}] FAIL {attempt}/{max_retries} | {e}")
                    if attempt < max_retries:
                        time.sleep(0.5 * attempt)

            return {
                "status": "error",
                "data": None,
                "error": str(last_error),
                "trace": traceback.format_exc(),
                "latency_ms": -1,
                "tool": name,
            }

        wrapper._is_harnessed = True
        wrapper._tool_name = name
        return wrapper

    return decorator
