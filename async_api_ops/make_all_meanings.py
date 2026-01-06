import logging
import json
from enum import Enum
from pathlib import Path

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
from ..sync_local_ops.mdx_dictionary import mdx_helper
from ..utils import get_field_config

logger = logging.getLogger(__name__)

MEANINGS_DICT_FILE = "_all_meanings_dict.json"


class MakeMeaningsResult(Enum):
    SUCCESS = 1
    NO_DICTIONARY_ENTRY = 2
    ERROR = 3


def make_all_meanings_for_word(
    config: dict[str, str], word: str, reading: str, all_meanings_dict: dict[str, list[dict]]
) -> MakeMeaningsResult:
    """
    Receive a word and its reading, and get an LLM to split all the meanings for that word found
    in the dictionary entries and write those into a JSON file.

    param config: Addon configuration dictionary.
    param word: The word or phrase being defined.
    param reading: The reading of the word or phrase.
    param all_meanings_dict: The dictionary to store all meanings.
    return: True when meanings were successfully made, False otherwise.
    """
    mdx_helper.load_mdx_dictionaries_if_needed(config, show_progress=True, finish_progress=False)

    # We won't necessarily have a dictionary entry for the word so the prompt will differ slightly
    # depending on whether we have one or not
    dict_meaning_for_word = mdx_helper.get_definition_text(
        word=word,
        reading=reading,
        # use all dictionaries to get the most comprehensive entry possible
        pick_dictionary="all",
    )
    if not dict_meaning_for_word:
        logger.debug(f"No dictionary entry found for word '{word}' ({reading})")
        return MakeMeaningsResult.NO_DICTIONARY_ENTRY

    jp_meaning_field = "jp_meaning"
    en_meaning_field = "en_meaning"
    prompt = f"""Below are dictionary entries from multiple different dictionaries for a word or phrase. Your task is to create a single comprehensive list of all distinct meanings expressed in these. Follow these rules:

- Create a Japanese meaning and English meaning.
- The English meaning doesn't have to translate the Japanese meaning literally but can explain the meaning by simply listing equivalent words or phrases, if possible. 
- Each meaning should be concise, ideally a single sentence but more if absolutely necessary.
- If the dictionary entries describes two usage patterns for this word or phrase - for example, one literal and one figurative - those should become one meaning where each is described shortly.
- If the dictionary entries includes multiple meanings that are similar, combine them into one. Avoid grouping too many meanings together; prioritize short and clear definitions.
- Make sure to analyze the different dictionary entries carefully and identify the same meanings expressed in different ways. The aim is to compress all the information into a minimal set of distinct meanings.
- Shorten and simplify the meanings as much as possible, ideally into 1 sentence and at most 2 (if describing both a literal and figurative usage), with more complex meanings being allowed more explanation.
- Exclude any example sentences, usage notes, or extraneous information.

Return a JSON object with one `meanings` field containing an array of objects, each with `{jp_meaning_field}` and `{en_meaning_field}` keys for each distinct meaning.

Word or phrase (and its reading):
{word} ({reading})
---
Dictionary entry:
{dict_meaning_for_word}
---
"""
    logger.debug(f"Prompt for updating meanings: {prompt}")

    response_schema = {
        "type": "object",
        "properties": {
            "meanings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        jp_meaning_field: {"type": "string"},
                        en_meaning_field: {"type": "string"},
                    },
                    "required": [jp_meaning_field, en_meaning_field],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["meanings"],
        "additionalProperties": False,
    }

    model = config.get("make_meanings_model", "")
    result = get_response(model, prompt, response_schema=response_schema)
    if result is None:
        logger.error("No response from model when making all meanings")
        return False
    if not isinstance(result, dict):
        logger.error(f"Response from model was not a dictionary: {result}")
        return False
    if "meanings" not in result:
        logger.error(f"Response from model did not contain 'meanings' field: {result}")
        return False
    if not isinstance(result["meanings"], list):
        logger.error(f"Response from model 'meanings' field was not a list: {result}")
        return False
    if not all(isinstance(m, dict) for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field did not contain all dictionaries: {result}"
        )
        return False
    if not all(jp_meaning_field in m and en_meaning_field in m for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field missing required keys in some meanings: {result}"
        )
        return MakeMeaningsResult.ERROR

    all_meanings = result["meanings"]
    all_meanings_dict[f"{word}_{reading}"] = all_meanings

    return MakeMeaningsResult.SUCCESS


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
- DO NOT overfit the definition to the sentence{'s' if len(sentences) > 1 else ''}, but rather pick as many meanings as possible that can broadly fit the theme of the sentence{'s' if len(sentences) > 1 else ''}.
- Figurative uses of literal meanings should be one definition: When there is a pair of meanings for this word or phrase, one literal and one figurative, always pick both and shorten their respective descriptions. Do this even if the sentence{'s' if len(sentences) > 1 else ''} only uses one of the meanings.
- In case there is only a single meaning, return that.

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


def make_meanings_in_note(
    config: dict[str, str],
    note: Note,
    processed_words_set: set[str],
    all_meanings_dict: dict[str, list[dict]],
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    note_type = note.note_type()
    if not note_type:
        logger.error(f"note_type() call failed for note {note.id}")
        return False

    try:
        word_sort_field = get_field_config(config, "word_sort_field", note_type)
        meaning_field = get_field_config(config, "meaning_field", note_type)
        word_field = get_field_config(config, "word_field", note_type)
        word_reading_field = get_field_config(config, "word_reading_field", note_type)

        word_normal_field = get_field_config(config, "word_normal_field", note_type)
    except KeyError as e:
        logger.error(str(e))
        return False

    logger.debug(f"cleaning meaning in note {note.id}")

    # Check if the note has the required fields
    if (
        meaning_field in note
        and word_field in note
        and word_reading_field in note
        and word_sort_field in note
        and word_normal_field in note
    ):
        word_key = f"{note[word_field]}_{note[word_reading_field]}"
        if (
            word_key in processed_words_set
            or note.has_tag("2-meanings-generated-to-json")
            or note.has_tag("2-no-dictionary-entry")
        ):
            logger.debug(f"word '{word_key}' already processed, skipping")
            return True

        meaning_notes_query = (
            f"{word_sort_field}:re:m\\d+ -{word_sort_field}:re:x\\d+"
            f' -nid:{note.id} "{word_reading_field}:{note[word_reading_field]}"'
            f' ("{word_normal_field}:{note[word_normal_field]}" OR'
            f' "{word_field}:{note[word_field]}")'
        )
        other_meaning_note_ids = mw.col.find_notes(meaning_notes_query)
        other_meaning_notes = [mw.col.get_note(onid) for onid in other_meaning_note_ids]
        logger.debug(
            f"Other meaning notes count: {len(other_meaning_notes)}, query: {meaning_notes_query}"
        )

        all_meaning_notes = other_meaning_notes + [note]

        result = make_all_meanings_for_word(
            config,
            note[word_field],
            note[word_reading_field],
            all_meanings_dict,
        )
        if result == MakeMeaningsResult.SUCCESS:
            processed_words_set.add(word_key)
            # Tag all notes for this word to know which notes have had meanings generated
            for meaning_note in all_meaning_notes:
                meaning_note.add_tag("2-meanings-generated-to-json")
                notes_to_update_dict[meaning_note.id] = meaning_note
            return True
        elif result == MakeMeaningsResult.NO_DICTIONARY_ENTRY:
            logger.debug(
                f"No dictionary entry found for word '{note[word_field]}'"
                f" ({note[word_reading_field]}), skipping meaning generation"
            )
            processed_words_set.add(word_key)
            # Tag all notes for this word to know which notes have no dictionary entry
            for meaning_note in all_meaning_notes:
                meaning_note.add_tag("2-no-dictionary-entry")
                notes_to_update_dict[meaning_note.id] = meaning_note
            return False
        return False
    else:
        logger.error("note is missing fields")
    return False


def write_meanings_dict_to_file(all_meanings_dict: dict[str, list[dict]]):
    media_path = Path(mw.pm.profileFolder(), "collection.media")
    all_meanings_dict_path = Path(media_path, MEANINGS_DICT_FILE)
    with open(all_meanings_dict_path, "w", encoding="utf-8") as f:
        json.dump(all_meanings_dict, f, ensure_ascii=False, indent=2)


def bulk_make_meanings_op(
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
    model = config.get("make_meanings_model", "")
    message = "Making meanings"
    processed_words_set: set[str] = set()

    media_path = Path(mw.pm.profileFolder(), "collection.media")
    all_meanings_dict_path = Path(media_path, MEANINGS_DICT_FILE)

    # Load existing meanings dictionary if it exists, we'll mutate the dict while processing and
    # then write it back at the end
    all_meanings_dict = {}
    if all_meanings_dict_path.exists():
        with open(all_meanings_dict_path, "r", encoding="utf-8") as f:
            all_meanings_dict = json.load(f)

    def op(config, note, notes_to_add_dict, notes_to_update_dict):
        nonlocal processed_words_set, all_meanings_dict
        return make_meanings_in_note(
            config,
            note,
            processed_words_set,
            all_meanings_dict,
            notes_to_add_dict,
            notes_to_update_dict,
        )

    def on_end():
        nonlocal all_meanings_dict
        # Write updated meanings dictionary to file after successful operation
        write_meanings_dict_to_file(all_meanings_dict)

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
        on_end=on_end,
    )


def make_meanings_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Making meanings")
    done_text = "Made meanings"
    bulk_op = bulk_make_meanings_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
