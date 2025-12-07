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
    AsyncTaskProgressUpdater,
)
from ..sync_local_ops.mdx_dictionary import AnkiMDXHelper
from ..utils import get_field_config

DEBUG = False

mdx_helper = AnkiMDXHelper()


def get_single_meaning_from_model(
    config: Dict[str, str],
    vocab: str,
    sentences: list[str],
    jp_dict_entry: str,
    en_dict_entry: str = "",
):
    jp_meaning_return_field = "cleaned_meaning"
    en_meaning_return_field = "english_meaning"
    sentences_formatted = ""
    if len(sentences) > 1:
        for sen in sentences:
            sentences_formatted += f"- {sen}\n"
    else:
        sentences_formatted = sentences[0]
    prompt = f"""Below, the dictionary entry for the word or phrase may contain multiple meanings. Your task is to either 1) extract the one meaning 2) or combine and rephrase meanings matching the usage of the word in the sentence{'s' if len(sentences) > 1 else ''}.

Selection criteria:
- If there are only two meanings for this word or phrase, one literal and one figurative, pick both and shorten their respective descriptions. Do this even if the sentence{'s' if len(sentences) > 1 else ''} only uses one of the meanings.
- In case there is only a single meaning, return that.
- DO NOT overfit the definition to the sentence{'s' if len(sentences) > 1 else ''}, but rather aim for a short general definition that fits the usage in {'each sentence' if len(sentences) > 1 else 'the sentence'}.

Omission rules:
- Omit any example sentences the matching meaning(s) included (often included within 「」 brackets).
- Descriptions of animals and plants are often scientific. From these omit descriptions on their ecology and only describe their appearance and type of plant/animal with simple language.

Simplification rules:
- YOU MUST Shorten and simplify the meaning as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.

Formatting rules:
- Clean off meaning numberings and other notation leaving only a plain text description.

Additionally, but only if it seems necessary, reword the English dictionary definition to fit the Japanese one. The English definition should ideally simply list equivalent words, if there are some, and only explain in sentences when it's necessary.

Return a JSON object with two fields:
Return the extracted and possibly modified Japanese meaning as the value of the key "{jp_meaning_return_field}".
Return the possibly modified English meaning as the value of the key "{en_meaning_return_field}".

Word or phrase: {vocab}
---
Sentence{'s' if len(sentences) > 1 else ''}: {sentences_formatted}
---
Japanese dictionary entry: {jp_dict_entry}
"""
    if en_dict_entry:
        prompt += f"---\nEnglish dictionary entry: {en_dict_entry}\n"
    if DEBUG:
        print("Prompt for cleaning meaning:", prompt)
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return jp_dict_entry, en_dict_entry
    try:
        return result[jp_meaning_return_field], result[en_meaning_return_field]
    except KeyError:
        return jp_dict_entry, en_dict_entry


def get_new_meaning_from_model(
    config: Dict[str, str],
    vocab: str,
    sentences: list[str],
) -> tuple[str, str]:
    jp_meaning_return_field = "new_meaning"
    en_meaning_return_field = "english_meaning"
    sentences_formatted = ""
    if len(sentences) > 1:
        for sen in sentences:
            sentences_formatted += f"- {sen}\n"
    else:
        sentences_formatted = sentences[0]
    prompt = f"""Below {'is a sentence' if len(sentences) == 1 else 'are sentences each'} containing a certain word or phrase. Your task is to generate a short monolingual dictionary style definition of the general meaning used in the sentence by the word or phrase.

- Generally aim to for the definition to be a single sentence. If it is necessary to explain more, the maximum length should be 3 sentences.
- Do not overfit the definition to the sentence{'s' if len(sentences) > 1 else ''}, but rather aim for a general definition that fits the usage in {'each sentence' if len(sentences) > 1 else 'the sentence'}.
- If there are two usage patterns for this word or phrase - for example, one literal and one figurative - describe both shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentence.
- The word itself should not be used in the definition.

The definition should be in the same language as the sentence. Also, generate a very short English translation of the meaning, ideally a list of equivalent words or phrases but explaining further, if necessary.

Return a JSON object with two fields:
Return the meaning ias the value of the key "{jp_meaning_return_field}".
Return the English translation as the value of the key "{en_meaning_return_field}".

Word or phrase: {vocab}
Sentence{'s' if len(sentences) > 1 else ''}: {sentences_formatted}
"""
    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return nothing if the generating failed
        return "", ""

    new_meaning = ""
    en_meaning = ""
    try:
        new_meaning = result[jp_meaning_return_field]
    except KeyError:
        print(f"Error: '{jp_meaning_return_field}' not found in the result")

    try:
        en_meaning = result[en_meaning_return_field]
    except KeyError:
        print(f"Error: '{en_meaning_return_field}' not found in the result")
    return new_meaning, en_meaning


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
        english_meaning_field = get_field_config(config, "english_meaning_field", note_type)
        word_field = get_field_config(config, "word_field", note_type)
        word_reading_field = get_field_config(config, "word_reading_field", note_type)
        sentence_field = get_field_config(config, "sentence_field", note_type)
        word_list_field = get_field_config(config, "word_list_field", note_type)
    except KeyError as e:
        print(e)
        return False

    if DEBUG:
        print("cleaning meaning in note", note.id)
        print("meaning_field in note", meaning_field in note)
        print("english_meaning_field in note", english_meaning_field in note)
        print("word_field in note", word_field in note)
        print("word_reading_field in note", word_reading_field in note)
        print("sentence_field in note", sentence_field in note)

    # Find other notes with the same word
    other_note_ids = mw.col.find_notes(f'"{word_list_field}:*{note.id}*" -nid:{note.id}')
    other_sentences = []
    for onid in other_note_ids:
        onote = mw.col.get_note(onid)
        if sentence_field in onote and onote[sentence_field] not in other_sentences:
            other_sentences.append(onote[sentence_field])
    # Check if the note has the required fields
    if (
        meaning_field in note
        and english_meaning_field in note
        and word_field in note
        and word_reading_field in note
        and sentence_field in note
    ):
        if DEBUG:
            print("note has fields")

        mdx_helper.load_mdx_dictionaries_if_needed(config)

        pick_dictionary = config.get("mdx_pick_dictionary", "all")

        # Get dictionary entry from mdx helper
        jp_dict_entry = mdx_helper.get_definition_text(
            word=note[word_field],
            reading=note[word_reading_field],
            pick_dictionary=pick_dictionary,
        )
        prev_en_meaning = note[english_meaning_field]
        word = note[word_field]
        sentences = [note[sentence_field]] + other_sentences
        # Check if the value is non-empty
        if jp_dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            new_jp_meaning, new_en_meaning = get_single_meaning_from_model(
                config, word, sentences, jp_dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = new_jp_meaning
            note[english_meaning_field] = new_en_meaning
            # Return success, if the we changed something
            if new_jp_meaning != jp_dict_entry or new_en_meaning != prev_en_meaning:
                return True
            return False
        else:
            # If there's no dict_entry, we'll use chatGPT to generate one from scratch
            new_meaning, en_meaning = get_new_meaning_from_model(config, word, sentences)
            note[meaning_field] = new_meaning
            note[english_meaning_field] = en_meaning
            if new_meaning != "" or en_meaning != "":
                return True
            return False

    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_clean_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list,
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: Dict[str, list[Note]] = {},
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("word_meaning_model", "")
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(
        message, config, op, col, notes, edited_nids, progress_updater, notes_to_add_dict, model
    )


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Cleaning meanings")
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
