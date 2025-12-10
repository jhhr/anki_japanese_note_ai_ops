import logging
from typing import TypedDict
from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..sync_local_ops.mdx_dictionary import AnkiMDXHelper
from ..utils import get_field_config

logger = logging.getLogger(__name__)

mdx_helper = AnkiMDXHelper()


def get_sentences_for_note(
    config: dict[str, str],
    note: Note,
    exclude_self: bool = False,
) -> list[str]:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return []
    word_list_field = get_field_config(config, "word_list_field", note_type)
    sentence_field = get_field_config(config, "sentence_field", note_type)
    other_sentence_note_ids = mw.col.find_notes(f'"{word_list_field}:*{note.id}*" -nid:{note.id}')
    other_sentences = [] if exclude_self else [note[sentence_field]]
    for onid in other_sentence_note_ids:
        onote = mw.col.get_note(onid)
        if sentence_field in onote and onote[sentence_field] not in other_sentences:
            other_sentences.append(onote[sentence_field])
    return other_sentences


WordAndSentences = TypedDict(
    "WordAndSentences",
    {
        "jp_meaning": str,
        "en_meaning": str,
        "sentences": list[str],
    },
)


def update_all_meanings_for_word(
    config: dict[str, str], word: str, reading: str, meanings_dict: dict[NoteId, WordAndSentences]
) -> dict[NoteId, tuple[str, str]]:
    """
    Receive a list of current meanings and sentences for a word, and have an AI model rework the
    meanings to better fit the sentences, using the dictionary definitions as reference.

    param config: Addon configuration dictionary.
    param word: The word or phrase being defined.
    param reading: The reading of the word or phrase.
    param meanings_dict: A dictionary mapping note IDs to their meanings and example sentences.
    return: A dictionary mapping note IDs to tuples of (new_japanese_meaning, new_english_meaning).
    """
    if not meanings_dict:
        return {}
    mdx_helper.load_mdx_dictionaries_if_needed(config, show_progress=True, finish_progress=False)
    pick_dictionary = config.get("mdx_pick_dictionary", "all")

    # We won't necessarily have a dictionary entry for the word so the prompt will differ slightly
    # depending on whether we have one or not
    dict_meaning_for_word = mdx_helper.get_definition_text(
        word=word,
        reading=reading,
        pick_dictionary=pick_dictionary,
    )
    # Format current meanings and sentences for the prompt
    meanings_and_sentences = ""
    # Sort by note id, smallest to largest, to have a consistent order
    meanings_dict_items = list(meanings_dict.items())
    meanings_dict_items.sort(key=lambda x: x[0])
    meaning_index_to_note_id = {}
    for i, (note_id, ws) in enumerate(meanings_dict_items):
        meaning_index_to_note_id[i] = note_id
        sentences_formatted = ""
        if len(ws["sentences"]) > 0:
            for sen in ws["sentences"]:
                sentences_formatted += f"- {sen}\n"
        else:
            sentences_formatted = ""
        meanings_and_sentences += f"""Meaning {i + 1}:
Japanese meaning: {ws['jp_meaning']}
English meaning: {ws['en_meaning']}
---
Sentences:
{sentences_formatted}
"""
    meaning_index_field = "meaning_index"
    jp_meaning_return_field = "jp_meaning"
    en_meaning_return_field = "en_meaning"

    prompt = f"""{f'''Below is the dictionary entry for a word or phrase, along with currently used meanings for groups of sentences containing that word or phrase. Your task is to rework the meanings to better fit the usage in the sentences, using the dictionary entry as reference.
For each meaning, either extract the relevant part from the dictionary entry, or rephrase it to better fit the sentences. Follow these rules:
- Do not overfit the definitions to the sentences, but rather aim for general definitions that fit the usage in each sentence.
- If the dictionary entry describes two usage patterns for this word or phrase - for example, one literal and one figurative - those should become one meaning where each is described shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentences.
- Omit any example sentences included in the dictionary entry (often included within 「」 brackets).
- Shorten and simplify the meanings as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.
''' if dict_meaning_for_word else f'''Below are currently used meanings for groups of sentences containing a certain word or phrase. Your task is to rework the meanings to better fit the usage in the sentences.
Follow these rules:
- Do not overfit the definitions to the sentences, but rather aim for general definitions that fit the usage in each sentence.
- If there are two usage patterns for this word or phrase - for example, one literal and one figurative - those should become one meaning where each is described shortly.
- If there are more than two usage patterns for this word or phrase, describe the one used in the sentences.
- Shorten and simplify the meanings as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.
'''}

Return a JSON array array of objects. Each object corresponds to a meaning and has the following fields:
- "{meaning_index_field}": The index of the meaning (starting from 1).
- "{jp_meaning_return_field}": The reworked Japanese meaning.
- "{en_meaning_return_field}": The reworked English meaning.

You do not need to rework all meanings, only those that seem not to fit well with the sentences; the array can be of any length, including empty if all meanings are fine.

Example output format:
[
    {{"{meaning_index_field}": 1, "{jp_meaning_return_field}": "最初の意味の短い説明。", "{en_meaning_return_field}": "A short English translation of the first meaning."}},
    {{"{meaning_index_field}": 3, "{jp_meaning_return_field}": "三番目の意味の短い説明。", "{en_meaning_return_field}": "A short English translation of the third meaning."}}
]

Word or phrase (and its reading):
{word} ({reading})
---{'''
Dictionary entry:
{dict_meaning_for_word}
---
''' if dict_meaning_for_word else ''}
Current meanings and sentences:
{meanings_and_sentences}"""
    logger.debug(f"Prompt for updating meanings: {prompt}")

    model = config.get("word_meaning_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return nothing if the call failed
        return {}
    updated_meanings: dict[NoteId, tuple[str, str]] = {}
    if isinstance(result, list):
        for meaning_obj in result:
            try:
                assert (
                    isinstance(meaning_obj, dict)
                    and meaning_index_field in meaning_obj
                    and isinstance(meaning_obj[meaning_index_field], int)
                    and jp_meaning_return_field in meaning_obj
                    and isinstance(meaning_obj[jp_meaning_return_field], str)
                    and meaning_obj[jp_meaning_return_field].strip() != ""
                    and en_meaning_return_field in meaning_obj
                    and isinstance(meaning_obj[en_meaning_return_field], str)
                    and meaning_obj[en_meaning_return_field].strip() != ""
                )
            except AssertionError:
                logger.warning(f"Invalid meaning object in result: {meaning_obj}")
                continue
            meaning_index = meaning_obj[meaning_index_field] - 1
            note_id = meaning_index_to_note_id.get(meaning_index)
            if note_id is not None:
                updated_meanings[note_id] = (
                    meaning_obj[jp_meaning_return_field],
                    meaning_obj[en_meaning_return_field],
                )
            else:
                logger.warning(f"Meaning index {meaning_index + 1} has no corresponding note ID")
    logger.debug(f"Updated meanings: {updated_meanings}")
    return updated_meanings


def get_single_meaning_from_model(
    config: dict[str, str],
    word: str,
    reading: str,
    sentences: list[str],
    jp_dict_entry: str,
    en_dict_entry: str = "",
):
    jp_meaning_return_field = "cleaned_meaning"
    en_meaning_return_field = "english_meaning"
    logger.debug(f"Getting single meaning with {len(sentences)} sentences")
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

Word or phrase (and its reading):
{word} ({reading})
---
Sentence{'s' if len(sentences) > 1 else ''}:
{sentences_formatted}
---
Japanese dictionary entry:
{jp_dict_entry}
"""
    if en_dict_entry:
        prompt += f"---\nEnglish dictionary entry:\n{en_dict_entry}\n"
    logger.debug(f"Prompt for cleaning meaning: {prompt}")
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
    config: dict[str, str],
    word: str,
    reading: str,
    sentences: list[str],
) -> tuple[str, str]:
    logger.debug(f"Getting new meaning with {len(sentences)} sentences")
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
Return the meaning as the value of the key "{jp_meaning_return_field}".
Return the English translation as the value of the key "{en_meaning_return_field}".

Word or phrase (and its reading): {word} ({reading})
Sentence{'s' if len(sentences) > 1 else ''}: {sentences_formatted}
"""
    logger.debug(f"Prompt for new meaning: {prompt}")
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
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return False

    try:
        meaning_field = get_field_config(config, "meaning_field", note_type)
        english_meaning_field = get_field_config(config, "english_meaning_field", note_type)
        word_field = get_field_config(config, "word_field", note_type)
        word_sort_field = get_field_config(config, "word_sort_field", note_type)
        word_normal_field = get_field_config(config, "word_normal_field", note_type)
        word_reading_field = get_field_config(config, "word_reading_field", note_type)
        sentence_field = get_field_config(config, "sentence_field", note_type)
    except KeyError as e:
        logger.error(str(e))
        return False

    logger.debug(f"cleaning meaning in note {note.id}")

    # Check if the note has the required fields
    if (
        meaning_field in note
        and english_meaning_field in note
        and word_field in note
        and word_reading_field in note
        and sentence_field in note
    ):
        logger.debug(f"notes_to_update_dict: {notes_to_update_dict is not None}")
        if notes_to_update_dict is not None:
            # Need to watch out with this, if this op is called in bulk from other ops, some other
            # update may already prevent us from changing this note
            if note.id in notes_to_update_dict:
                logger.debug(
                    f"Skipping note {note.id} as it's already marked as updated by a previous op"
                )
                return False
            meaning_notes_query = (
                f"{word_sort_field}:re:m\\d+ -{word_sort_field}:re:x\\d+"
                f' -nid:{note.id} "{word_reading_field}:{note[word_reading_field]}"'
                f' ("{word_normal_field}:{note[word_normal_field]}" OR'
                f' "{word_field}:{note[word_field]}")'
            )
            other_meaning_note_ids = mw.col.find_notes(meaning_notes_query)
            all_meaning_notes = [
                (
                    mw.col.get_note(onid)
                    if onid not in notes_to_update_dict
                    else notes_to_update_dict[onid]
                )
                for onid in other_meaning_note_ids
            ]
            all_meaning_notes.append(note)

            logger.debug(
                f"All meaning notes count: {len(all_meaning_notes)}, query: {meaning_notes_query}"
            )

            if len(all_meaning_notes) > 1:
                meaning_sentences_dict = {
                    n.id: WordAndSentences(
                        jp_meaning=n[meaning_field],
                        en_meaning=n[english_meaning_field],
                        sentences=get_sentences_for_note(config, n),
                    )
                    for n in all_meaning_notes
                }
                updated_meanings_dict = update_all_meanings_for_word(
                    config,
                    note[word_field],
                    note[word_reading_field],
                    meaning_sentences_dict,
                )
                any_changed = False
                for n in all_meaning_notes:
                    if n.id in updated_meanings_dict:
                        new_jp_meaning, new_en_meaning = updated_meanings_dict[n.id]
                        prev_en_meaning = n[english_meaning_field]
                        prev_jp_meaning = n[meaning_field]
                        n[meaning_field] = new_jp_meaning
                        n[english_meaning_field] = new_en_meaning
                        n.add_tag("updated_jp_meaning")
                        if new_jp_meaning != prev_jp_meaning or new_en_meaning != prev_en_meaning:
                            any_changed = True
                            if n.id != 0 and n.id not in notes_to_update_dict:
                                notes_to_update_dict[n.id] = n
                return any_changed

        mdx_helper.load_mdx_dictionaries_if_needed(
            config, show_progress=True, finish_progress=False
        )
        pick_dictionary = config.get("mdx_pick_dictionary", "all")
        # Get dictionary entry from mdx helper
        jp_dict_entry = mdx_helper.get_definition_text(
            word=note[word_field],
            reading=note[word_reading_field],
            pick_dictionary=pick_dictionary,
        )
        prev_en_meaning = note[english_meaning_field]
        word = note[word_field]
        reading = note[word_reading_field]
        sentences = get_sentences_for_note(config, note)
        # Check if the value is non-empty
        if jp_dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            new_jp_meaning, new_en_meaning = get_single_meaning_from_model(
                config, word, reading, sentences, jp_dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = new_jp_meaning
            note[english_meaning_field] = new_en_meaning
            # Return success, if the we changed something
            if new_jp_meaning != jp_dict_entry or new_en_meaning != prev_en_meaning:
                if note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False
        else:
            # If there's no dict_entry, we'll let a model generate one from scratch
            new_meaning, en_meaning = get_new_meaning_from_model(config, word, reading, sentences)
            note[meaning_field] = new_meaning
            note[english_meaning_field] = en_meaning
            if new_meaning != "" or en_meaning != "":
                if note.id not in notes_to_update_dict:
                    notes_to_update_dict[note.id] = note
                return True
            return False

    else:
        logger.error("note is missing fields")
    return False


def bulk_clean_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list,
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("word_meaning_model", "")
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(
        message,
        config,
        op,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict=notes_to_add_dict,
        notes_to_update_dict=notes_to_update_dict,
        model=model,
    )


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Cleaning meanings")
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
