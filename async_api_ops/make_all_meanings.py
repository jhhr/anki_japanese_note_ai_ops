import logging
import json
from pathlib import Path

from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from ..configuration import (
    MEANINGS_DICT_FILE,
    NO_DICTIONARY_ENTRY_TAG,
    MEANINGS_GENERATED_TAG,
    GeneratedMeaningsDictType,
    WordAndSentences,
    MakeMeaningsResult,
)

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..sync_local_ops.mdx_dictionary import mdx_helper
from ..utils import get_field_config

logger = logging.getLogger(__name__)


def make_meaning_dict_key(word: str, reading: str) -> str:
    return f"{word}_{reading}"


def make_all_meanings_for_word(
    config: dict[str, str],
    word: str,
    reading: str,
    all_meanings_dict: GeneratedMeaningsDictType,
) -> MakeMeaningsResult:
    """
    Receive a word and its reading, and get an LLM to split all the meanings for that word found
    in the dictionary entries and write those into a JSON file.

    :param config: Addon configuration dictionary.
    :param word: The word or phrase being defined.
    :param reading: The reading of the word or phrase.
    :param all_generated_meanings_dict: Dict of all generated meanings
            for reuse across multiple calls. Provided to avoid doing file operations during
            async operations and to avoid doing file reading in every op.
    :return: True when meanings were successfully made, False otherwise.
    """
    mdx_helper.load_mdx_dictionaries_if_needed(config, show_progress=True, finish_progress=False)

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
    prompt = f"""Below are dictionary entries from multiple different dictionaries for a word or phrase. Your task is to compress this information into a comprehensive list of distinct meanings expressed in these. The objective is to create a partitioning of all possible usages that is useful for an English-speaking Japanese learner. Follow these rules:

Definition rules:
- Create a Japanese meaning and English meaning.
- MINIMIZE DIFFERENT MEANINGS: Make as a few separate meanings as possible by combining meanings that are related. Extremely common words like 見る, 行く, 来る, 有る, etc. have a large number of meanings which should be especially aggressively combined into as few meanings as possible.
- BE CONCISE: Each meaning should be concise, ideally a single sentence but more if absolutely necessary to combine multiple related meanings.
- The English meaning should almost always be a short list of equivalent words or phrases separated by semicolons. Only explain in sentences when equivalents do not exist; e.g. the word is "untranslatable".

Combination rules:
- Most of all, if the dictionary entries includes multiple meanings that are similar, combine them into one.
- However also combine dissimilar meanings that that are of the type one literal and one metaphorical - those should become one meaning where each is described shortly.

Information extraction rules:
- Make sure to analyze the different dictionary entries carefully and identify the same meanings expressed in different ways. The aim is to compress all the information into a minimal set of distinct meanings.
- Note that the dictionary entries may include information about additinal phrases that use the word. These would usually come after the list of main meanings per dictionary. Ignore these additional phrases and only focus on the main meanings of the word or phrase itself.
- Exclude any example sentences, usage notes, or extraneous information.

Return a JSON object with one `meanings` field containing an array of objects, each with `{jp_meaning_field}` and `{en_meaning_field}` keys for each distinct meaning.

Word or phrase (and its reading):
{word} ({reading})
---
Dictionary entry:
{dict_meaning_for_word}
---
"""
    logger.debug(f"Prompt for generating possible meanings: {prompt}")

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
        return MakeMeaningsResult.ERROR
    if not isinstance(result, dict):
        logger.error(f"Response from model was not a dictionary: {result}")
        return MakeMeaningsResult.ERROR
    if "meanings" not in result:
        logger.error(f"Response from model did not contain 'meanings' field: {result}")
        return MakeMeaningsResult.ERROR
    if not isinstance(result["meanings"], list):
        logger.error(f"Response from model 'meanings' field was not a list: {result}")
        return MakeMeaningsResult.ERROR
    if not all(isinstance(m, dict) for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field did not contain all dictionaries: {result}"
        )
        return MakeMeaningsResult.ERROR
    if not all(jp_meaning_field in m and en_meaning_field in m for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field missing required keys in some meanings: {result}"
        )
        return MakeMeaningsResult.ERROR

    all_meanings = result["meanings"]
    all_meanings_dict[f"{word}_{reading}"] = all_meanings

    return MakeMeaningsResult.SUCCESS


def revise_meanings_for_word(
    config: dict[str, str],
    word: str,
    reading: str,
    bad_note_meanings_dict: dict[NoteId, WordAndSentences],
    all_meanings_dict: GeneratedMeaningsDictType,
) -> MakeMeaningsResult:
    """
    Receive a previously generated meanings, a list of words and sentences that could not matched
    to any of those meanings for a word and its reading, and get an LLM to revise those meanings
    so that the meanings better cover the word usages.

    :param config: Addon configuration dictionary.
    :param word: The word or phrase being defined.
    :param reading: The reading of the word or phrase.
    :param bad_note_meanings_dict: Dict of note IDs to WordAndSentences for notes that could not
            be matched to any of the previously generated meanings.
    :param all_generated_meanings_dict: Dict of all generated meanings
            for reuse across multiple calls. Provided to avoid doing file operations during
            async operations and to avoid doing file reading in every op.
    :return: The revised list of meanings when successful, None otherwise. Will also mutate
            all_meanings_dict to include the revised meanings.
    """
    mdx_helper.load_mdx_dictionaries_if_needed(config, show_progress=True, finish_progress=False)

    meanings_and_sentences = ""
    for i, word_and_sentences in enumerate(list(bad_note_meanings_dict.values())):
        sentences_formatted = ""
        for sen in word_and_sentences["sentences"]:
            sentences_formatted += f"  - JP: {sen['jp_sentence']} -- EN: {sen['en_sentence']}\n"
        meanings_and_sentences += f"""
UNMATCHED USAGE {i + 1}:
- Description of usage in Japanese: {word_and_sentences['jp_meaning'] or '(empty)'}
- English description: {word_and_sentences['en_meaning'] or '(empty)'}
- Sentences:
{sentences_formatted}
"""
    word_key = make_meaning_dict_key(word, reading)
    existing_meanings = all_meanings_dict.get(word_key, [])

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
    prompt = f"""Below are dictionary entries from multiple different dictionaries for a word or phrase. From this, a single list of all distinct meanings was previously created. The list was meant to be comprehensive enough to match all possible usages of the word but some usages of the word were encountered that did not match any of the meanings. Your task is to revise the previous list of meanings to better cover all the usages expressed in the usages and the dictionary entry. The objective is to create a partitioning of all possible usages that is useful for an English-speaking Japanese learner. Follow these rules:

Definition rules:
- Create a Japanese meaning and English meaning.
- MINIMIZE DIFFERENT MEANINGS: Make as a few separate meanings as possible by combining meanings that are related. Extremely common words like 見る, 行く, 来る, 有る, etc. have a large number of meanings which should be especially aggressively combined into as few meanings as possible.
- BE CONCISE: Each meaning should be concise, ideally a single sentence but more if absolutely necessary to combine multiple related meanings.
- The English meaning should almost always be a short list of equivalent words or phrases separated by semicolons. Only explain in sentences when equivalents do not exist; e.g. the word is "untranslatable".
- If an unmatched usage is something that no dictionary entry covers, add a new meaning to cover that usage. Create the meaning based on the description and sentences provided following the same rules as above.

Combination rules:
- Most of all, if the dictionary entries includes multiple meanings that are similar, combine them into one.
- However also combine dissimilar meanings that that are of the type one literal and one metaphorical - those should become one meaning where each is described shortly.
- Especially, if an unmatched usage is a combination of two meanings then follow its example, and make just one meaning that covers both.

Information extraction rules:
- Make sure to analyze the different dictionary entries carefully and identify the same meanings expressed in different ways. The aim is to compress all the information into a minimal set of distinct meanings.
- Note that the dictionary entries may include information about additinal phrases that use the word. These would usually come after the list of main meanings per dictionary. Ignore these additional phrases and only focus on the main meanings of the word or phrase itself.
- Exclude any example sentences, usage notes, or extraneous information.
- When creating new meanings based on unmatched usages, focus on the core meaning being expressed and avoid referencing anything about the example sentences directly.

Return a JSON object with one `meanings` field containing an array of objects, each with `{jp_meaning_field}` and `{en_meaning_field}` keys for each distinct meaning.

WORD OR PHRASE (READING):
{word} ({reading})

---
PREVIOUSLY GENERATED MEANINGS:
{{{json.dumps(existing_meanings, ensure_ascii=False, indent=2)}}}
---
WORD USAGES NOT COVERED BY PREVIOUS MEANINGS:
{meanings_and_sentences}
---
DICTIONARY ENTRIES:
{dict_meaning_for_word}
---
"""
    logger.debug(f"Prompt for revising possible meanings: {prompt}")

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
        return MakeMeaningsResult.ERROR
    if not isinstance(result, dict):
        logger.error(f"Response from model was not a dictionary: {result}")
        return MakeMeaningsResult.ERROR
    if "meanings" not in result:
        logger.error(f"Response from model did not contain 'meanings' field: {result}")
        return MakeMeaningsResult.ERROR
    if not isinstance(result["meanings"], list):
        logger.error(f"Response from model 'meanings' field was not a list: {result}")
        return MakeMeaningsResult.ERROR
    if not all(isinstance(m, dict) for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field did not contain all dictionaries: {result}"
        )
        return MakeMeaningsResult.ERROR
    if not all(jp_meaning_field in m and en_meaning_field in m for m in result["meanings"]):
        logger.error(
            f"Response from model 'meanings' field missing required keys in some meanings: {result}"
        )
        return MakeMeaningsResult.ERROR

    revised_meanings = result["meanings"]
    logger.debug(f"Revised meanings: {json.dumps(revised_meanings, ensure_ascii=False, indent=2)}")
    all_meanings_dict[f"{word}_{reading}"] = revised_meanings

    return MakeMeaningsResult.SUCCESS


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
        word_key = make_meaning_dict_key(note[word_field], note[word_reading_field])
        if (
            word_key in processed_words_set
            or note.has_tag(MEANINGS_GENERATED_TAG)
            or note.has_tag(NO_DICTIONARY_ENTRY_TAG)
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
                meaning_note.add_tag(MEANINGS_GENERATED_TAG)
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
                meaning_note.add_tag(NO_DICTIONARY_ENTRY_TAG)
                notes_to_update_dict[meaning_note.id] = meaning_note
            return False
        return False
    else:
        logger.error("note is missing fields")
    return False


def write_meanings_dict_to_file(all_meanings_dict: dict[str, list[dict]]):
    media_path = Path(mw.pm.profileFolder(), "collection.media")
    all_meanings_dict_path = Path(media_path, MEANINGS_DICT_FILE)
    # Sort the dictionary by keys before writing to file for consistency
    all_meanings_dict = dict(sorted(all_meanings_dict.items()))
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
    all_generated_meanings_dict: GeneratedMeaningsDictType = {}
    if all_meanings_dict_path.exists():
        with open(all_meanings_dict_path, "r", encoding="utf-8") as f:
            all_generated_meanings_dict = json.load(f)

    def op(config, note, notes_to_add_dict, notes_to_update_dict):
        nonlocal processed_words_set, all_generated_meanings_dict
        return make_meanings_in_note(
            config,
            note,
            processed_words_set,
            all_generated_meanings_dict,
            notes_to_add_dict,
            notes_to_update_dict,
        )

    def on_end():
        nonlocal all_generated_meanings_dict
        # Write updated meanings dictionary to file after successful operation
        write_meanings_dict_to_file(all_generated_meanings_dict)

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
