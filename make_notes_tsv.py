import os
from anki.notes import Note
from aqt import mw
from aqt.qt import QFileDialog
from aqt.import_export.importing import import_file

DEBUG = False


def prompt_os_to_save_file(
    filename: str,
    text: str,
) -> None:
    """
    Prompt the OS to save a file with the given filename and text.

    Args:
        filename (str): The name of the file to save.
        text (str): The content to write into the file.
    """
    if not filename:
        raise ValueError("Filename must not be empty")

    file_dialog = QFileDialog(mw)
    file_dialog.setWindowTitle("Save File")
    file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    file_dialog.setDefaultSuffix("txt")
    file_dialog.setNameFilter("All Files (*);;Text Files (*.txt);;TSV Files (*.tsv)")
    file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
    file_dialog.setDirectory(mw.pm.profileFolder())  # Default to Anki's profile folder
    file_dialog.setViewMode(QFileDialog.ViewMode.List)
    file_dialog.setOptions(
        QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.DontConfirmOverwrite
    )
    file_dialog.selectFile(filename)  # Set the default filename
    # Show the dialog and get the selected file path
    if file_dialog.exec():
        file_path = file_dialog.selectedFiles()[0]
        if file_path:
            # Write the text to the file, overwriting, if it already exists
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)


def import_tsv_file(filename: str, text: str) -> None:
    """
    Write text to a file in the media folder
    """
    if not filename:
        raise ValueError("Filename must not be empty")

    # Write the text to a temporary file in the media folder
    media_folder = os.path.join(mw.pm.profileFolder(), "collection.media")
    file_path = os.path.join(media_folder, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

    # Import the file using Anki's import functionality
    import_file(
        mw,
        file_path,
    )


def make_tsv_from_notes(
    notes: list[Note],
    config: dict,
):
    """
    Create a TSV string from the list of notes. Uses the

    Args:
        notes (list[Note]): List of notes to create CSV from.
        config (dict): Addon config to get deck to write to per note_type

    Returns:
        str: TSV string with note fields.
    """
    if not notes:
        return ""

    if not config:
        print("Error: Missing addon configuration")
        return ""

    tsv_lines = ["#separator:tab", "#html:true", "#notetype column:1"]

    # If the config specifies a deck for any of the notes, add the deck column, otherwise
    # deck will be unspecified
    deck_column = ""
    for note in notes:
        note_type = note.note_type()
        if not note_type:
            print(f"Error: Note {note.id} has no note type")
            continue
        note_type_name = note_type["name"]
        try:
            deck = config[note_type_name]["insert_deck"]
            if deck:
                deck_column = "#deck column:2"
                break
        except KeyError:
            # Note type is not configured, skip it
            continue
    if deck_column:
        tsv_lines.append(deck_column)

    # Process the notes, putting the fields in the right order
    for note in notes:
        note_type = note.note_type()
        if not note_type:
            print(f"Error: Note {note.id} has no note type")
            continue
        note_type_name = note_type["name"]

        # 1st column = note type name
        note_data = [note_type_name]
        # 2nd column = deck name, if specified in the config
        # Use Default deck if not specified
        if deck_column:
            try:
                deck = config[note_type_name]["insert_deck"]
                if deck:
                    note_data.append(deck)
                else:
                    note_data.append("Default")
            except KeyError:
                note_data.append("Default")

        # Next all fields in the order defined in the note type
        fields = note_type["flds"]
        for field in fields:
            field_name = field["name"]
            # Get the field value, if it exists, else use empty string
            field_value = note[field_name] if field_name in note else ""
            # Escape tabs and newlines in the field value
            field_value = str(field_value).replace("\t", " ").replace("\n", " ")
            note_data.append(field_value)

        tsv_lines.append("\t".join(note_data))

    # Create the TSV string
    return "\n".join(tsv_lines)
