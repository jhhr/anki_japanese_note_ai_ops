import traceback
import logging
from typing import Optional

from anki.notes import Note
from anki.models import NotetypeDict


def get_field_config(config: dict, field_name: str, note_type: NotetypeDict) -> str:
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


def print_error_traceback(e: Exception, logger_instance: Optional[logging.Logger] = None):
    """Print the traceback of an exception without triggering Anki to display an error dialog."""
    # format_exception returns the complete exception info including traceback
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    if logger_instance:
        logger_instance.error("Traceback:")
        for line in tb_lines:
            logger_instance.error(line.rstrip())
    else:
        print("Traceback:")
        for line in tb_lines:
            print(line.rstrip())
