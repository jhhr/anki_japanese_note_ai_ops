from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = True


def get_translated_field_from_model(sentence):
    return_field = "english_sentence"
    # HTML-keeping prompt
    # keep_html_prompt = f"sentence_to_translate_into_english: {sentence}\n\nTranslate the sentence into English. Copy the HTML structure into the English translation. Return the translation in a JSON string as the value of the key \"{return_field}\". Convert \" characters into ' withing the value to keep the JSON valid."
    no_html_prompt = f'sentence_to_translate_into_english: {sentence}\n\nIgnore any HTML in the sentence.\nReturn an HTML-free English translation of the sentence in a JSON string as the value of the key "{return_field}".'
    model = mw.addonManager.getConfig(__name__).get("translate_sentence_model", "")
    result = get_response(model, no_html_prompt)
    if result is None:
        # If translation failed, return nothing
        return None
    try:
        return result[return_field]
    except KeyError:
        return None


def translate_sentence_in_note(
    note: Note, config: dict, show_warning: bool = True
) -> bool:
    model = note.note_type()
    try:
        sentence_field = get_field_config(config, "sentence_field", model)
        translated_sentence_field = get_field_config(
            config, "translated_sentence_field", model
        )
    except Exception as e:
        print(e)
        return False

    if DEBUG:
        print("sentence_field in note", sentence_field in note)
        print("translated_sentence_field in note", translated_sentence_field in note)
    # Check if the note has the required fields
    if sentence_field in note and translated_sentence_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        sentence = note[sentence_field]
        if DEBUG:
            print("sentence", sentence)
        # Check if the value is non-empty
        if sentence:
            # Call API to get translation
            translated_sentence = get_translated_field_from_model(sentence)
            if DEBUG:
                print("translated_sentence", translated_sentence)
            if translated_sentence is not None:
                # Update the note with the new value
                note[translated_sentence_field] = translated_sentence
                return True
            return False
        return False
    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_translate_notes_op(col, notes: Sequence[Note], edited_nids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Translating sentences"
    op = translate_sentence_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids)


def translate_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated translation"
    bulk_op = bulk_translate_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
