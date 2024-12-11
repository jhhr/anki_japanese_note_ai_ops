from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence
from typing import Dict

from .base_ops import (
    get_response_from_chat_gpt,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = True


def get_single_meaning_from_chat_gpt(vocab, sentence, dict_entry):
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
    result = get_response_from_chat_gpt(prompt)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    try:
        return result[return_field]
    except KeyError:
        return dict_entry


def generate_meaning_from_chatGPT(vocab, sentence):
    return_field = "new_meaning"
    prompt = f'\
    Below is a sentence containing a word or phrase.\
    Generate a short monolingual dictionary style general definition of the word or phrase.\
    If there are two usage patterns for this word or phrase, one literal and one figurative, describe both shortly.\
    If there are more than two usage patterns for this word or phrase, describe the one used in the sentence.\
    The word itself should not be used in the definition. If any synonyms exist, mention at most two.\
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
    result = get_response_from_chat_gpt(prompt)
    if result is None:
        # Return nothing if the generating failed
        return ""
    try:
        return result[return_field]
    except KeyError:
        return ""


def clean_meaning_in_note(
    note: Note, config: Dict[str, str], show_warning: bool = True
):
    model = note.note_type()

    try:
        meaning_field = get_field_config(config, "meaning_field", model)
        word_field = get_field_config(config, "word_field", model)
        sentence_field = get_field_config(config, "sentence_field", model)
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
            modified_meaning_jp = get_single_meaning_from_chat_gpt(
                word, sentence, dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the we changed something
            if modified_meaning_jp != dict_entry:
                return True
            return False
        else:
            # If there's no dict_entry, we'll use chatGPT to generate one from scratch
            new_meaning = generate_meaning_from_chatGPT(word, sentence)
            note[meaning_field] = new_meaning
            if new_meaning != "":
                return True
            return False

    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_clean_notes_op(col, notes: Sequence[Note], edited_nids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids)


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
