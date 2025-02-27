from functools import wraps
import time
from rich.console import Console
console = Console()
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        console.print(f"{func.__name__} 执行时间: {(end - start)*1000:.4f} 毫秒")
        return result
    return wrapper