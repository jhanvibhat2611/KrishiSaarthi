"""Runtime validation that remains active under optimized Python."""
def require(condition, message="Validation failed"):
    if not condition:
        raise ValueError(message)
