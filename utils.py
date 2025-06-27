def get_field_config(config: dict, field_name: str, note_type: dict) -> str:
    note_type_name = note_type["name"]
    try:
        model_config = config[note_type_name]
    except KeyError:
        raise Exception(f'Note type "{note_type_name}" has not been configured in the settings.')
    try:
        field = model_config[field_name]
    except KeyError:
        raise Exception(f'Missing config for "{field_name}" with model {note_type_name}')
    return field
