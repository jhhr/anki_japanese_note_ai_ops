from anki import notes_pb2
from anki.notes import Note



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


def copy_into_new_note(note: Note) -> Note:
    """
    Duplicate a note by creating a new instance and copying the fields from the original note.
    """
    new_copy = Note(col=note.col, model=note.note_type())

    new_copy.fields = list(note.fields)  
    new_copy.tags = list(note.tags)
    note_type = note.note_type()
    if not note_type:
        raise ValueError("Note type is not set for the note being copied.")
    new_copy._fmap = new_copy.col.models.field_map(note_type) 
    return new_copy

