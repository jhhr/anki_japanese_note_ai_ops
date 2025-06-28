from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence
from typing import Dict

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = False


def get_single_meaning_from_model(
        config: Dict[str, str],
        vocab: str,
        sentence: str,
        dict_entry: str,
    ):
    return_field = "cleaned_meaning"
    prompt = f'\
    Below, the dictionary entry for the word or phrase may contain multiple meanings.\
    Extract the one meaning matching the usage of the word in the sentence.\
    If there are only two meanings for this word or phrase, one literal and one figurative, pick both and shorten their respective descriptions.\
    Omit any example sentences the matching meaning included (often include within 「」 brackets).\
    Shorten and simplify the meaning as much possible, ideally into 1-3 sentences, with more complex meanings being allowed more sentences.\
    Descriptions of animals and plants are often scientific. From these omit descriptions on their ecology and only describe their appearance and type of plant/animal with simple language.\
    In case there is only a single meaning, return that.\
    Clean off meaning numberings and other notation leaving only a plain text description.\
    \
    Return the extracted meaning, in Japanese, in a JSON string as the value of the key "{return_field}".\
    \
    word_or_phrase: {vocab}\
    sentence: {sentence}\
    dictionary_entry_for_word: {dict_entry}\
    '
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    try:
        return result[return_field]
    except KeyError:
        return dict_entry


def get_new_meaning_from_model(
        config: Dict[str, str],
        vocab: str,
        sentence: str,
    ) -> str:
    return_field = "new_meaning"
    prompt = f'\
    Below is a sentence containing a word or phrase.\
    Generate a short monolingual dictionary style definition of the general meaning used in the sentence by the word or phrase.\
    If there are two usage patterns for this word or phrase - for example, one literal and one figurative - describe both shortly.\
    If there are more than two usage patterns for this word or phrase, describe the one used in the sentence.\
    The word itself should not be used in the definition..\
    Generally aim to for the definition to be a single sentence.\
    If it is necessary to explain more, the maximum length should be 3 sentences.\
    The definition should be in the same language as the sentence.\
    \
    Return the meaning in a JSON string as the value of the key "{return_field}".\
    \
    word_or_phrase: {vocab}\
    sentence: {sentence}\
    \
    '
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return nothing if the generating failed
        return ""
    try:
        return result[return_field]
    except KeyError:
        return ""


def clean_meaning_in_note(
    config: Dict[str, str],
    note: Note,
    notes_to_add_dict: Dict[str, list[Note]],
    ) -> bool:
    note_type = note.note_type()
    if not note_type:
        print("Error: note_type() call failed for note", note.id)
        return False

    try:
        meaning_field = get_field_config(config, "meaning_field", note_type)
        word_field = get_field_config(config, "word_field", note_type)
        sentence_field = get_field_config(config, "sentence_field", note_type)
    except KeyError as e:
        print(e)
        return False

    if DEBUG:
        print("cleaning meaning in note", note.id)
        print("meaning_field in note", meaning_field in note)
        print("word_field in note", word_field in note)
        print("sentence_field in note", sentence_field in note)
    # Check if the note has the required fields
    if meaning_field in note and word_field in note and sentence_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        dict_entry = note[meaning_field]
        word = note[word_field]
        sentence = note[sentence_field]
        # Check if the value is non-empty
        if dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            modified_meaning_jp = get_single_meaning_from_model(
                config, word, sentence, dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the we changed something
            if modified_meaning_jp != dict_entry:
                return True
            return False
        else:
            # If there's no dict_entry, we'll use chatGPT to generate one from scratch
            new_meaning = get_new_meaning_from_model(config, word, sentence)
            note[meaning_field] = new_meaning
            if new_meaning != "":
                return True
            return False

    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_clean_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list,
    notes_to_add_dict: Dict[str, list[Note]] = {},
    ):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("word_meaning_model", "")
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids, notes_to_add_dict, model)


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
