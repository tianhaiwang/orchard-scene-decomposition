def copy_diagnostics(value):
    if isinstance(value, list):
        return [copy_diagnostics(item) for item in value]
    if isinstance(value, dict):
        return {key: copy_diagnostics(item) for key, item in value.items()}
    return value
