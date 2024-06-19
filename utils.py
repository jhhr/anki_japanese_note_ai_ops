def get_field_config(config: dict, field_name: str, model: dict) -> str:
    model_name = model["name"]
    try:
        model_config = config[model_name]
    except KeyError:
        raise Exception(f'Note type "{model_name}" has not been configured in the settings.')
    try:
        field = model_config[field_name]
    except KeyError:
        raise Exception(f'Missing config for "{field_name}" with model {model_name}')
    return field
