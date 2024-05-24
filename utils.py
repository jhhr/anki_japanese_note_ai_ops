def get_field_config(config: dict, field_name: str, model: dict) -> str:
    model_name = model["name"]
    try:
        field_config = config[field_name]
    except KeyError:
        raise Exception(f'Missing config for "{field_name}"')
    try:
        field = field_config[model_name]
    except KeyError:
        raise Exception(f'Missing config for "{field_name}" with model {model_name}')
    return field
