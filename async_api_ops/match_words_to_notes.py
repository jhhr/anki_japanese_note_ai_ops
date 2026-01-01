import random
import json
import re
import asyncio
import threading
import logging
from typing import Union, Sequence, Callable, Any, Coroutine, cast
from aqt import mw

from anki.notes import Note, NoteId
from anki.models import NotetypeDict
from anki.collection import Collection

from .base_ops import (
    get_response,
    make_inner_bulk_op,
    bulk_nested_notes_op,
    selected_notes_op,
    CancelState,
    AsyncTaskProgressUpdater,
)
from .clean_meaning import clean_meaning_in_note
from .extract_words import word_lists_str_format
from ..kana_conv import to_hiragana
from ..utils import copy_into_new_note, get_field_config, print_error_traceback
from ..configuration import (
    raw_one_meaning_word_type,
    raw_multi_meaning_word_type,
    matched_word_type,
)

logger = logging.getLogger(__name__)

WORD_LIST_TO_PART_OF_SPEECH: dict[str, str] = {
    "nouns": "Noun",
    "proper_nouns": "Proper Noun",
    "numbers": "Number",
    "counters": "Counter",
    "verbs": "Verb",
    "compound_verbs": "Verb",
    "adjectives": "Adjective",
    "adverbs": "Adverb",
    "adjectivals": "Adjectival",
    "particles": "Particle",
    "conjunctions": "Conjunction",
    "pronouns": "Pronoun",
    "suffixes": "Suffix",
    "prefixes": "Prefix",
    "expressions": "Expression",
    "yojijukugo": "Idiom",
}
WORD_LISTS = list(WORD_LIST_TO_PART_OF_SPEECH.keys())


def make_new_note_id(note: Note) -> int:
    """
    Generate a new note ID based on the note's fields.
    This is used to ensure that new notes have unique IDs.

    Args:
        note (Note): The note to generate the ID for.

    Returns:
        int: A unique ID for the note represented as a negative integer.
    """
    if not note:
        return 0
    # Use a simple random negative number for fake IDs
    return -random.randint(1000000, 9999999)


ProcessedWordTuple = Union[
    raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type, None
]
FinalWordTuple = Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]


def update_fake_note_ids(
    new_notes: Sequence[Note],
    config: dict,
    progress_updater: AsyncTaskProgressUpdater,
) -> dict[NoteId, Note]:
    """
    Update the fake note IDs in the notes to be the actual note IDs.
    Args:
        notes (Sequence[Note]): The notes to update.
        config (dict): The addon configuration.
    Returns:
        dict[NoteId, Note]: A dictionary mapping the original note IDs to the updated notes.
    """
    notes_to_update_dict: dict[NoteId, Note] = {}
    if not new_notes:
        return notes_to_update_dict
    if not config:
        logger.error("Error: Missing addon configuration")
        return notes_to_update_dict
    total_notes = len(new_notes)
    progress_updater.update_new_note_processing_progress(
        total_notes=total_notes,
    )
    for index, new_note in enumerate(new_notes):
        note_type = new_note.note_type()
        if not note_type:
            logger.error(f"Error: Note {new_note.id} has no note type")
            continue
        new_note_id_field = get_field_config(config, "new_note_id_field", note_type)
        word_list_field = get_field_config(config, "word_list_field", note_type)
        if not new_note_id_field or not word_list_field:
            logger.error("Error: Missing required fields in config")
            return notes_to_update_dict
        if new_note_id_field in new_note and word_list_field in new_note:
            # Find other notes whose word_list_field contains the fake note ID
            fake_note_id = new_note[new_note_id_field]
            if not fake_note_id:
                continue
            referencing_note_ids = mw.col.find_notes(f'"{word_list_field}:*{fake_note_id}*"')
            if not referencing_note_ids:
                continue
            referencing_notes = []
            # First get any notes already added to update_notes_dict matching any referencing IDs
            previous_nids = []
            for nid in referencing_note_ids:
                if nid in notes_to_update_dict:
                    referencing_notes.append(notes_to_update_dict[nid])
                    previous_nids.append(nid)
            referencing_note_ids = [nid for nid in referencing_note_ids if nid not in previous_nids]
            # Fetch the rest from the collection
            referencing_notes.extend([mw.col.get_note(nid) for nid in referencing_note_ids])
            for referencing_note in referencing_notes:
                if new_note_id_field in referencing_note:
                    # Update the word_list_field to point to the actual new note ID
                    referencing_note[word_list_field] = referencing_note[word_list_field].replace(
                        str(fake_note_id), str(new_note.id)
                    )
                    if referencing_note.id not in notes_to_update_dict:
                        # Note was updated, add it to the updated notes dict, if not already there
                        notes_to_update_dict[referencing_note.id] = referencing_note
            # Empty fake note ID to indicate we've finished updating the references for this note
            new_note[new_note_id_field] = ""
            progress_updater.update_new_note_processing_progress(
                total_notes=total_notes,
                new_notes_processed=index + 1,
            )

    # For every new note that emptied its fake note ID, we can now add it to the updated notes dict
    for new_note in new_notes:
        if (
            new_note_id_field in new_note
            and not new_note[new_note_id_field]
            and new_note.id != 0
            and new_note.id not in notes_to_update_dict
        ):
            notes_to_update_dict[new_note.id] = new_note

    return notes_to_update_dict


def json_result_corrector(json_result: str) -> str:
    # Sometimes the AI omits the closing ] and } causing json decoding to fail
    return json_result + "]}"


def match_words_to_notes(
    config: dict,
    current_note: Note,
    word_tuples: list[
        Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]
    ],
    word_list_key: str,
    sentence: str,
    tasks: list[asyncio.Task],
    note_tasks: list[asyncio.Task],
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    progress_updater: AsyncTaskProgressUpdater,
    cancel_state: CancelState,
    update_word_list_in_dict: Callable[[list[ProcessedWordTuple]], None],
    note_type: NotetypeDict,
    replace_existing: bool = False,
):
    """
    Match words to notes based on the kanjified sentence.

    Args:
        config (dict): Addon config
        current_note (Note): The current note being processed
        word_tuples (list): List of tuples containing words and their readings, possibly some values
            already processed by this function previously
        sentence (str): The sentence that provides context for the words' meaning
        tasks (list): List of asyncio tasks to append to. Will be mutated by this function.
        notes_to_add_dict (dict): Dict to append new notes to be added. Will be mutated by this
            function. Used to also check if the operation has already created something it should
            reuse.
        notes_to_update_dict (dict): Dict to append notes to be updated with new meanings and also
            to get an already updated note for additional changes if needed. Will be mutated by this
            function.
        progress_updater (AsyncTaskProgressUpdater): An updater to report progress of the operation.
        update_word_list_in_dict (Callable): Function to update the word list in a note. Should
            be called async when the task actually finishes.
        replace_existing (bool): If True, replace existing matched words with new matches.
            Otherwise, words that already have a note match will be skipped during processing and
            returned as is.
    """
    if not word_tuples:
        logger.debug("No words to match against notes")
        return word_tuples
    if not sentence:
        logger.error("Error: No sentence provided for matching words")
        return word_tuples
    if not config:
        logger.error("Error: Missing addon configuration")
        return word_tuples
    # Get the model name from the config
    model = config.get("match_words_model", "")
    if not model:
        logger.error("Error: Missing match words model in config")
        return word_tuples

    config["rate_limits"] = config.get("rate_limits", {})
    rate_limit = config["rate_limits"].get(model, None)

    # Get the field names from the config
    word_list_field = get_field_config(config, "word_list_field", note_type)
    word_kanjified_field = get_field_config(config, "word_kanjified_field", note_type)
    word_normal_field = get_field_config(config, "word_normal_field", note_type)
    word_reading_field = get_field_config(config, "word_reading_field", note_type)
    word_sort_field = get_field_config(config, "word_sort_field", note_type)
    meaning_field = get_field_config(config, "meaning_field", note_type)
    meaning_audio_field = get_field_config(config, "meaning_audio_field", note_type)
    furigana_sentence_field = get_field_config(config, "furigana_sentence_field", note_type)
    part_of_speech_field = get_field_config(config, "part_of_speech_field", note_type)
    english_meaning_field = get_field_config(config, "english_meaning_field", note_type)
    new_note_id_field = get_field_config(config, "new_note_id_field", note_type)

    missing_fields = []
    for field_name in [
        word_list_field,
        word_kanjified_field,
        word_normal_field,
        word_reading_field,
        word_sort_field,
        meaning_field,
        furigana_sentence_field,
        part_of_speech_field,
        english_meaning_field,
        new_note_id_field,
    ]:
        if not field_name:
            missing_fields.append(field_name)
    if missing_fields:
        logger.error(f"Error: Missing fields in config: {', '.join(missing_fields)}")
        return word_tuples

    processed_word_tuples = cast(list[ProcessedWordTuple], word_tuples.copy())

    logger.debug(f"match_words_to_notes, notes_to_add_dict before: {notes_to_add_dict}")

    # Dictionary to track locks per word to prevent race conditions
    word_locks: dict[str, asyncio.Lock] = {}
    word_locks_lock = asyncio.Lock()  # Lock to safely create new word locks

    async def match_op(
        _,
        notes_to_add_dict: dict[str, list[Note]],
        notes_to_update_dict: dict[NoteId, Note],
        # the **op_args passed by process_op in make_inner_bulk_op and process_word_tuple
        # below
        word_index: int,
        word: str,
        reading: str,
    ) -> bool:
        """Process a single word tuple to match it with notes in the collection.
        Args:
            config (dict): Addon configuration
            word_index (int): Index of the word tuple in the original list.
            word (str): The word to match.
            reading (str): The reading of the word to match.
            word_list_key (str): The word list the word came from used for part of speech
        Returns:
            bool: True if the word tuple was processed successfully, False otherwise.
        """
        logger.debug(f"match_op, notes_to_add_dict before: {notes_to_add_dict}")
        logger.debug(f"Processing word tuple at index {word_index}: {word}, reading: {reading}")
        nonlocal processed_word_tuples, word_locks, word_locks_lock
        # If the word contains only non-japanese characters, skip it
        if not re.search(r"[ぁ-んァ-ン一-龯]", word):
            logger.debug(
                f"Skipping word '{word}' at index {word_index} as it contains no Japanese"
                " characters"
            )
            processed_word_tuples[word_index] = None
        # Check for existing suru verbs words including する in either field, remove する in the word
        if (
            word.endswith("する")
            and reading.endswith("する")
            and not re.match(r"(?:に|が)する$", word)
        ):
            word = word[:-2]
            reading = reading[:-2]

        # Get or create a lock for this specific word to prevent race conditions
        async with word_locks_lock:
            if word not in word_locks:
                word_locks[word] = asyncio.Lock()

        # Acquire the lock for this word before checking/modifying notes_to_add_dict
        async with word_locks[word]:
            # Entries for words starting with the honorific prefix may use the kanji or hiragana so
            # query for both
            go_word_query = ""
            if word.startswith("御"):
                o_word = "お" + word[1:]
                go_word = "ご" + word[1:]
                go_word_query = (
                    f' OR "{word_kanjified_field}:{o_word}" OR "{word_normal_field}:{o_word}" OR'
                    f' "{word_kanjified_field}:{go_word}" OR "{word_normal_field}:{go_word}"'
                )

            word_query = (
                f'("{word_kanjified_field}:{word}" OR "{word_normal_field}:{word}" OR'
                f' "{word_normal_field}:{reading}"{go_word_query})'
            )
            word_query_suru = (
                f'("{word_kanjified_field}:{word}する" OR "{word_normal_field}:{word}する")'
            )
            no_x_in_sort_field = f'-"{word_sort_field}:re:\(x\d\)"'
            query = f"({word_query} OR {word_query_suru}) {no_x_in_sort_field}"
            logger.debug(f"Searching for notes with query: {query}")
            note_ids: Sequence[NoteId] = mw.col.find_notes(query)
            # Filter by reading matches, we don't do this in the query since it's not easy to check
            # for a reading where some parts are in katakana

            hiragana_reading = to_hiragana(reading)
            hiragana_reading_suru = to_hiragana(reading + "する")
            matching_notes = []
            for note_id in note_ids:
                note = (
                    mw.col.get_note(note_id)
                    if note_id not in notes_to_update_dict
                    else notes_to_update_dict[note_id]
                )
                if note and word_reading_field in note:
                    note_reading = to_hiragana(note[word_reading_field])
                    logger.debug(
                        f"Comparing note reading: {note_reading} with {hiragana_reading} and"
                        f" {hiragana_reading_suru}"
                    )
                    if note_reading in [hiragana_reading, hiragana_reading_suru]:
                        matching_notes.append(note)

            logger.debug(
                f"Found matching notes: {[note[word_sort_field] for note in matching_notes]}"
            )

            # This part is why the locks are needed, async tasks should await for the lock before
            # checking notes_to_add_dict
            matching_new_notes = notes_to_add_dict.get(word, [])

            if not matching_notes:
                new_note = Note(col=mw.col, model=note_type)
                new_note[word_kanjified_field] = word
                new_note[word_normal_field] = word
                new_note[word_reading_field] = reading
                new_note[word_sort_field] = word
                new_note.add_tag("new_matched_jp_word")
                # Query for a note with a (kun)/(on) or (rX) marker in the sort field, we'll want to
                # set the marker in this note appropriately based on that:
                # - if there is (kun) --> this should be (on), or other way around
                # - if there is (rX) --> this should be (rY) where Y is the next number in the sequence
                # - if there is (kun)(rX) --> this should be (kun)(rY) where Y is the next number in
                #   the sequence, same for (on)(rX)
                # - if there is no marker, this should have none
                marker_regex = f"^{word} ?(?:\((?:kun|on)\))?(?:\(r\d+\))?(?:\(m\d+\))?$"
                marker_note_ids = mw.col.find_notes(f'"{word_sort_field}:re:{marker_regex}"')
                # Additionally, search the notes_to_add_dict to see if any of them have a marker like
                # this, as we'll need to increment the rX number higher than the largest one found
                marker_notes = [mw.col.get_note(note_id) for note_id in marker_note_ids]
                if word in notes_to_add_dict:
                    for added_note in notes_to_add_dict[word]:
                        if word_sort_field in added_note:
                            marker_sort_field = added_note[word_sort_field]
                            if re.match(rf"{marker_regex}", marker_sort_field):
                                # If the marker matches, we can add this note ID to the marker notes
                                marker_notes.append(added_note)

                # When checking the marker notes, theres' two cases
                # Case 1: The existing notes had some (rX) number, the new should be (rY) where Y is
                #         the next number in the sequence. No edits needed to the existing notes
                # Case 2: If the existing notes didn't have (rX) the new note will have (r2) and the
                #         existing notes will be edited to have (r1)
                def update_marker_note(a_marker_note: Note):
                    # Case 2: No (rX) present
                    # If there's other markers, place (r1) between (kun/on) and the rest

                    if a_marker_note.id and a_marker_note.id in notes_to_update_dict:
                        # Replace note with the one from the dict so all edits to the note are in it
                        a_marker_note = notes_to_update_dict[a_marker_note.id]
                    other_markers_match = re.search(
                        r"(\((?:kun|on)\))?(\(\w\d+\))?",
                        a_marker_note[word_sort_field],
                    )
                    other_markers = ""
                    kun_on_marker = ""
                    if other_markers_match:
                        kun_on_marker = other_markers_match.group(1) or ""
                        other_markers = other_markers_match.group(2) or ""
                    a_marker_note[word_sort_field] = f"{word} {kun_on_marker}(r1){other_markers}"

                    # This is needed for an existing note that hasn't been edited yet
                    if a_marker_note.id and a_marker_note.id not in notes_to_update_dict:
                        notes_to_update_dict[a_marker_note.id] = a_marker_note

                if len(marker_notes) == 1:
                    marker_note = marker_notes[0]
                    if marker_note and word_sort_field in marker_note:
                        if marker_note.id and marker_note.id in notes_to_update_dict:
                            marker_note = notes_to_update_dict[marker_note.id]
                        marker_sort_field = marker_note[word_sort_field]
                        # Check if the sort field has a (kun) or (on) marker
                        if "(kun)" in marker_sort_field:
                            new_note[word_sort_field] = f"{word} (on)"
                        elif "(on)" in marker_sort_field:
                            new_note[word_sort_field] = f"{word} (kun)"
                        else:
                            # If no (kun)/(on) marker, just use the word
                            new_note[word_sort_field] = word
                        # Check for (rX) markers
                        r_match = re.search(r"\(r(\d+)\)", marker_sort_field)
                        if r_match:
                            r_number = int(r_match.group(1)) + 1
                            new_note[word_sort_field] += f" (r{r_number})"
                        else:
                            new_note[word_sort_field] += " (r2)"
                            update_marker_note(marker_note)

                elif len(marker_notes) > 1:
                    # This ought to be case where there's either zero or multiple rX markers for the
                    # same word, so get the largest rX number and otherwise use the same (kun)/(on)
                    # logic as above
                    largest_r_number = 0
                    # Hopefully there are only either (kun)(rX) or (on)(rX) markers, and not both
                    # as we can't know which kind of reading this word is using without doing some
                    # reading lookups for the word...
                    found_kun = False
                    found_on = False
                    for marker_note in marker_notes:
                        if marker_note and word_sort_field in marker_note:
                            marker_sort_field = marker_note[word_sort_field]
                            if "(kun)" in marker_sort_field:
                                found_kun = True
                            elif "(on)" in marker_sort_field:
                                found_on = True
                            r_match = re.search(r"\(r(\d+)\)", marker_sort_field)
                            if r_match:
                                r_number = int(r_match.group(1))
                                if r_number > largest_r_number:
                                    largest_r_number = r_number
                    if found_kun and not found_on:
                        new_note[word_sort_field] = f"{word} (kun)(r{largest_r_number + 1})"
                    elif found_on and not found_kun:
                        new_note[word_sort_field] = f"{word} (on)(r{largest_r_number + 1})"
                    elif not found_kun and not found_on:
                        new_note[word_sort_field] = f"{word} (r{largest_r_number + 1})"
                    else:
                        # If we found both (kun) and (on) markers, we can't decide which one to use,
                        # so just use the word without any markers and add a tag to the note for this
                        # to be manually checked
                        new_note[word_sort_field] = word
                        new_note.add_tag("check_reading_marker")
                    if largest_r_number == 0:
                        # Case 2: need to update the marker notes
                        for marker_note in marker_notes:
                            update_marker_note(marker_note)

                new_note[furigana_sentence_field] = sentence
                new_note[meaning_field] = ""
                new_note[part_of_speech_field] = WORD_LIST_TO_PART_OF_SPEECH.get(word_list_key, "")
                new_note[english_meaning_field] = ""
                new_note_id = make_new_note_id(new_note)
                new_note[new_note_id_field] = str(new_note_id)
                # Add the new note to the notes to add dict
                # With the threading lock in place other tasks processing the same word can find it
                notes_to_add_dict.setdefault(word, []).append(new_note)
                create_meaning_result = clean_meaning_in_note(
                    config, new_note, notes_to_add_dict, notes_to_update_dict
                )
                new_note[word_sort_field] = (
                    new_note[word_sort_field].replace(") (", ")(").replace("  ", " ")
                )
                processed_word_tuples[word_index] = (
                    word,
                    reading,
                    new_note[word_sort_field],
                    new_note_id,
                )
                return create_meaning_result

            if matching_new_notes:
                # If we have notes to add that match the word, we should add them to the list of notes
                # to check against, so we are comparing against both existing notes and new notes
                matching_notes.extend(matching_new_notes)

            # Get all the meanings from the notes to check against the sentence
            meanings: list[tuple[str, int, NoteId, str, str, str]] = []
            # meaning is a tuple of (jp_meaning, meaning_number, note_id, example_sentence, en_meaning)
            largest_meaning_index = 0
            note_to_copy = None
            for note in matching_notes:
                logger.debug(
                    f"Processing note {note.id} for word {word} with reading {reading}, sort field"
                    f" {note[word_sort_field]}"
                )
                if meaning_field in note:
                    meaning = note[meaning_field]
                    other_sentence = (
                        note[furigana_sentence_field] if furigana_sentence_field in note else ""
                    )
                    english_meaning = (
                        note[english_meaning_field] if english_meaning_field in note else ""
                    )
                    match_word = note[word_kanjified_field] if word_kanjified_field in note else ""
                    if meaning:
                        sort_field = note[word_sort_field]
                        # Get the meaning number, if any from sort field, in the form (m1), (m2), etc.
                        match = re.match(r"\(m(\d+)\)", sort_field)
                        matched_meaning_number = 0
                        if not note_to_copy:
                            # Ensure we have a note to copy that has a meaning
                            note_to_copy = note
                        if match:
                            matched_meaning_number = int(match.group(1))
                            if matched_meaning_number > largest_meaning_index:
                                largest_meaning_index = matched_meaning_number
                                note_to_copy = note
                        meanings.append((
                            meaning,
                            matched_meaning_number,
                            note.id,
                            other_sentence,
                            english_meaning,
                            match_word,
                        ))
                    else:
                        logger.debug(f"Note {note.id} has empty meaning field")
                else:
                    logger.debug(f"Note {note.id} is missing meaning field")
            if note_to_copy:
                # use the updated note if available
                if note_to_copy.id in notes_to_update_dict:
                    note_to_copy = notes_to_update_dict[note_to_copy.id]
            if not meanings:
                logger.debug(f"No meanings found for word {word} with reading {reading}")
                # We found notes but all were missing meanings, have to skip this word as it can't
                # processed properly
                return False
            # Sort meanings by the meaning number
            meanings.sort(key=lambda x: x[1])
            meanings_str = ""
            for i, (
                jp_meaning,
                _,
                _,
                example_sentence,
                en_meaning,
                match_word,
            ) in enumerate(meanings):
                meanings_str += f"""Meaning number {i + 1}:
- *match_word*: {match_word}
- *jp_meaning*: {jp_meaning}
- *example_sentence*: {example_sentence}
- *en_meaning*: {en_meaning}
"""

            instructions = """You are an expert Japanese lexicographer. Your task is to analyze how a Japanese word is used in a _current sentence_ and compare it to a list of existing dictionary meanings. You are designed to output JSON.

**Primary Goal: Minimize creation of new meanings**
Your main goal is to try to contain all closely related nuances in a single meaning. Always modify one of the existing meanings, if it somewhat matches the current context and could be adjusted to fit the current and previous contexts. However, do not refer to specific phrases in a meaning, to avoid overfitting; to fit more contexts, try to explain the meaning in general terms using easy-to-understand language. Keep the meaning short, 3 sentences is already too long. Only if such modifications no longer make sense because the existings meanings are not close enough or further modification would make the meaning too long and/or complicated, you may consider the **CREATE NEW** action.

**Your Actions**
You will generate a JSON object. This array will describe your actions. You must provide one of the two actions.

1.  **MATCH**
    -   Choose this if one existing meaning either accurately represents the word's usage in the sentence or can be modified to fit its previous meaning and the current context.
    -   In the JSON, create a object with `"is_matched_meaning": true`.
    -   Set `"meaning_number"` to the 1-based index of the matching meaning.
    -   You can optionally provide improved `"jp_meaning"` or `"en_meaning"` in this same object if the original has minor flaws.

2.  **CREATE NEW**
    -   Choose this **only if every existing meaning is unsuitable**.
    -   In the JSON, create a object with `"is_matched_meaning": false"` and `"meaning_number": null`.
    -   You MUST provide a new `"jp_meaning"` and `"en_meaning"`.

**JSON OUTPUT RULES:**
- The output is a single JSON object with 3-4 properties:
- "is_matched_meaning": A boolean indicating whether you are matching an existing meaning (true) or creating a new one (false).
- "meaning_number": An integer (1-based index) indicating which existing meaning you are matching, or null if creating a new meaning.
- "jp_meaning": (optional) A string with the improved Japanese meaning if matching, or the new Japanese meaning if creating a new one.
- "en_meaning": (optional) A string with the improved English meaning if matching, or the new English meaning if creating a new one.
- **CRITICAL**: `meaning_number` must be a valid 1-based index from the provided list. Do not invent numbers.

---
**Example 1: MATCH with an improvement to the matched meaning**
The first meaning is a good match, but its English and Japanese are modified to match the current usage.
```json
{
    "is_matched_meaning": true,
    "meaning_number": 1,
    "jp_meaning": "改善された日本語の定義。",
    "en_meaning": "A new, improved English definition for the matched meaning."
}
```

**Example 2: CREATE NEW**
None of the meanings fit, so you create a new one.
```json
{
    "is_matched_meaning": false,
    "meaning_number": null,
    "jp_meaning": "新しい日本語の定義。",
    "en_meaning": "A new English definition for the new usage."
}
```"""

            prompt = f"""MEANINGS AND EXAMPLE SENTENCES
{meanings_str}

_Targeted word_: {word}
_Current sentence_: {sentence}"""

            # response_schema = {
            #     "type": "object",
            #     "properties": {
            #         "meanings": {
            #             "type": "array",
            #             "description": "Array of meaning objects for this word",
            #             "items": {
            #                 "type": "object",
            #                 "properties": {
            #                     "is_matched_meaning": {
            #                         "type": "boolean",
            #                         "description": "Whether this meaning matches an existing note",
            #                     },
            #                     "meaning_number": {
            #                         "type": "integer",
            #                         "description": (
            #                             "The number of the listed meaning if it matches, or null if it"
            #                             " doesn't match"
            #                         ),
            #                     },
            #                     "en_meaning": {
            #                         "type": "string",
            #                         "description": "New English meaning definition for the word",
            #                     },
            #                     "jp_meaning": {
            #                         "type": "string",
            #                         "description": "New Japanese meaning definition for the word",
            #                     },
            #                 },
            #                 "required": ["is_matched_meaning"],
            #             },
            #         }
            #     },
            #     "required": ["meanings"],
            # }

            # The response is not expected to be very long, 1k tokens would about 30 meanings and
            # generally a word might have 1-3 meanings with some rare words having 5+
            # Reaching this limit very likely means the model got stuck in a loop repeating the same
            # text over and over
            # However the thinking tokens are also counted towards the limit so the total token count
            # of thinking + response must be considered. Thinking can take a several thousand tokens
            max_output_tokens = 8000
            logger.debug(f"\n\nmeanings_str: {meanings_str}")

            raw_result = get_response(
                model,
                prompt,
                cancel_state=cancel_state,
                instructions=instructions,
                # The response schema seems to actually make the model mess up the result more as the
                # schema doesn't match the complex rules described in the instructions
                # response_schema=response_schema,
                max_output_tokens=max_output_tokens,
                json_result_corrector=json_result_corrector,
            )
            logger.debug(f"Raw result: {raw_result}")
            if raw_result is None:
                logger.debug("Failed to get a response from the API.")
                # If the prompt failed, return nothing
                return False
            meaning_action = None
            # First get the list of meanings from the raw result
            if isinstance(raw_result, dict):
                meaning_action = raw_result
            elif isinstance(raw_result, list) and len(raw_result) > 0:
                # If the result is a list, get the first item
                meaning_action = raw_result[0]
            else:
                logger.debug(
                    f"Error: Expected a list or dict, got {type(raw_result)} instead. Result:"
                    f" {raw_result}"
                )
                return False
            # Check the meaning_action structure
            if "meaning_number" in meaning_action and not isinstance(
                meaning_action["meaning_number"], (int, type(None))
            ):
                logger.debug(
                    "Error: invalid meaning action, 'meaning_number' is not an int or"
                    f" None. Result: {meaning_action}"
                )
                return False
            if "is_matched_meaning" in meaning_action and not isinstance(
                meaning_action["is_matched_meaning"], bool
            ):
                logger.debug(
                    "Error: invalid meaning action, 'is_matched_meaning' is not a bool."
                    f" Result: {meaning_action}"
                )
                return False
            if "jp_meaning" in meaning_action and not isinstance(
                meaning_action["jp_meaning"], (str, type(None))
            ):
                logger.debug(
                    "Error: invalid meaning action, 'jp_meaning' is not a str or None."
                    f" Result: {meaning_action}"
                )
                return False
            if "en_meaning" in meaning_action and not isinstance(
                meaning_action["en_meaning"], (str, type(None))
            ):
                logger.debug(
                    "Error: invalid meaning action, 'en_meaning' is not a str or None."
                    f" Result: {meaning_action}"
                )
                return False
            if "is_matched_meaning" in meaning_action and "meaning_number" not in meaning_action:
                logger.debug(
                    "Error: invalid meaning action, 'is_matched_meaning' is set but"
                    f" 'meaning_number' is not. Result: {meaning_action}"
                )
                return False
            if "meaning_number" in meaning_action and meaning_action["meaning_number"] is not None:
                if (
                    meaning_action["meaning_number"] < 1
                    or meaning_action["meaning_number"] > len(meanings) + 1
                ):
                    logger.debug(
                        "Error: invalid meaning action, 'meaning_number' is out of range."
                        f" Result: {meaning_action}"
                    )
                    return False
            is_matched_meaning = meaning_action.get("is_matched_meaning", False)
            meaning_number = meaning_action.get("meaning_number", None)
            jp_meaning = meaning_action.get("jp_meaning", None)
            en_meaning = meaning_action.get("en_meaning", None)
            # If meaning_number is too big, the AI got confused, skip this
            if meaning_number is not None and meaning_number > len(meanings):
                logger.debug(
                    "Error: invalid 'meaning_number' in meaning action, too large. Result:"
                    f" {meaning_action}"
                )
                return False
            if is_matched_meaning and meaning_number is not None:
                meaning_number = meaning_number - 1  # Convert to 0-based index
                try:
                    meanings[meaning_number]
                except IndexError:
                    logger.debug(
                        f"Error: Matched meaning number {meaning_number} is out of range for"
                        f" word {word} with reading {reading}"
                    )
                    return False

            if not is_matched_meaning and not jp_meaning and not en_meaning:
                logger.debug(
                    "Error: Meaning action is not a match and is not modifying"
                    f" either meaning. Result: {meaning_action}"
                )
                return False
            if not is_matched_meaning and meaning_number is None and (jp_meaning or en_meaning):
                # If either jp_meaning or en_meaning is set, we should treat this as a new meaning
                # though technically the requirement was for both, we'll accept this as still useful
                # (semi)valid action CREATE NEW
                logger.debug(
                    f"Interpreting odd meaning action as CREATE NEW for word {word} with reading"
                    f" {reading}"
                )
                meaning_action = {
                    "meaning_number": None,
                    "is_matched_meaning": False,
                    "jp_meaning": jp_meaning,
                    "en_meaning": en_meaning,
                }
            logger.debug(
                f"Valid meaning action for word {word} with reading {reading}: {meaning_action}"
            )
            if not is_matched_meaning and meaning_number is None and (jp_meaning or en_meaning):
                # Action CREATE NEW. duplicate the note with the biggest meaning number, incrementing it by 1
                if meanings:
                    largest_meaning_index += 1
                if note_to_copy:
                    if note_to_copy.id in notes_to_update_dict:
                        # Replace note object from dict if its there
                        note_to_copy = notes_to_update_dict[note_to_copy.id]
                    new_note = copy_into_new_note(note_to_copy)
                    # Don't copy tags and the word list field to the new note
                    new_note.set_tags_from_str("")
                    new_note[word_list_field] = ""
                    new_note[meaning_audio_field] = ""
                    new_note.add_tag("new_matched_jp_word")
                    new_note[meaning_field] = jp_meaning.strip() if jp_meaning else ""
                    new_note[english_meaning_field] = en_meaning.strip() if en_meaning else ""

                    # If we're copying a note, we need to ensure the meaning number is at least 2
                    # as the first meaning should be (m1)
                    largest_meaning_index = max(largest_meaning_index, 2)
                    # Either replace the (mX) in the sort field, or if there was none, add it
                    prev_sort_field = new_note[word_sort_field]
                    mxRec = re.compile(r"\(m(\d+)\)")
                    # Additionally, check notes_to_add_dict for any notes we've added for this word
                    # during this run, as we'll need to use the largest meaning index out of them
                    if word in notes_to_add_dict:
                        for added_note in notes_to_add_dict[word]:
                            if word_sort_field in added_note:
                                added_sort_field = added_note[word_sort_field]
                                mx_match = mxRec.search(added_sort_field)
                                if mx_match:
                                    # If we found a (mX) in the sort field,
                                    # update the largest meaning index
                                    largest_meaning_index = max(
                                        largest_meaning_index, int(mx_match.group(1)) + 1
                                    )
                    new_note_id = make_new_note_id(new_note)
                    new_note[new_note_id_field] = str(new_note_id)
                    if prev_sort_field and mxRec.search(prev_sort_field):
                        # Replace the existing (mX) with the new meaning number
                        new_note[word_sort_field] = mxRec.sub(
                            f"(m{largest_meaning_index})", prev_sort_field
                        )
                    elif prev_sort_field:
                        if re.search(r"\(\w\d+\)", prev_sort_field):
                            # If there is no (mX) but some other number, add meaning number to end
                            # And update the note_to_copy, since it should a meaning number now too
                            new_note[word_sort_field] += f"(m{largest_meaning_index})"
                            note_to_copy[word_sort_field] += f"(m{largest_meaning_index - 1})"
                        else:
                            # Else, same but add a space
                            new_note[word_sort_field] += f" (m{largest_meaning_index})"
                            note_to_copy[word_sort_field] += f" (m{largest_meaning_index - 1})"
                            note_to_copy[word_sort_field] = (
                                note_to_copy[word_sort_field]
                                .replace(") (", ")(")
                                .replace("  ", " ")
                            )
                        if note_to_copy.id not in notes_to_update_dict:
                            notes_to_update_dict[note_to_copy.id] = note_to_copy
                    # Note to copy was missing sort field somehow? Add it now + the meaning numbers
                    else:
                        new_note[word_sort_field] = f"{word} (m{largest_meaning_index})"
                        note_to_copy[word_sort_field] = f"{word} (m{largest_meaning_index - 1})"
                        if note_to_copy.id not in notes_to_update_dict:
                            notes_to_update_dict[note_to_copy.id] = note_to_copy
                    notes_to_add_dict.setdefault(word, []).append(new_note)
                    new_note[word_sort_field] = (
                        new_note[word_sort_field].replace(") (", ")(").replace("  ", " ")
                    )
                    # Note, new_note.id will be 0 here, we'll instead use the sort field value to
                    # find it after insertion and then update the processed_word_tuples
                    processed_word_tuples[word_index] = (
                        word,
                        reading,
                        new_note[word_sort_field],
                        new_note_id,
                    )
                    return True
                else:
                    logger.debug(f"Error: No note to copy for word {word} with reading {reading}")
                    return False
            elif is_matched_meaning and meaning_number is not None:
                # Action MATCH. update the meaning in the note with the matched meaning
                logger.debug(
                    f"Matched meaning JP:'{jp_meaning}'/EN:'{en_meaning}' for word {word} with"
                    f" reading {reading}, meaning number {meaning_number}, meanings: {meanings}"
                )
                # We have a match, so we can update the note with the matched meaning
                matched_note = None

                matched_meaning, _, matched_note_id, _, _, _ = meanings[meaning_number]
                for note in matching_notes:
                    if note.id == matched_note_id and matched_meaning == note[meaning_field]:
                        # Ensure the matched meaning is the same as in the note to account for id=0
                        matched_note = note
                        if matched_note.id in notes_to_update_dict:
                            matched_note = notes_to_update_dict[matched_note.id]
                        break
                if not matched_note:
                    # This shouldn't happen as the meaning came from one of the notes in the list
                    # and indicates the matched_meaning somehow isn't in the matching_notes
                    logger.debug(
                        f"Error: Matched note with ID {matched_note_id} not found for word"
                        f" {word} with reading {reading}"
                    )
                    return False
                # Add the matched note to processed_word_tuples
                new_word_tuple = (word, reading, matched_note[word_sort_field], matched_note.id)
                logger.debug(
                    f"Setting processed word tuple at index {word_index}, with tuple"
                    f" {new_word_tuple}"
                )
                processed_word_tuples[word_index] = new_word_tuple
                # Now we need to update the meaning in the note
                if jp_meaning and matched_note[meaning_field] != jp_meaning.strip():
                    matched_note[meaning_field] = jp_meaning.strip()
                    matched_note.add_tag("updated_jp_meaning")
                    if matched_note.id not in notes_to_update_dict:
                        notes_to_update_dict[matched_note.id] = matched_note
                if en_meaning and matched_note[english_meaning_field] != en_meaning.strip():
                    matched_note[english_meaning_field] = en_meaning.strip()
                    if matched_note.id not in notes_to_update_dict:
                        notes_to_update_dict[matched_note.id] = matched_note
                logger.debug(
                    f"Updated note {matched_note.id} with new meaning '{jp_meaning}' and"
                    f" english meaning '{en_meaning}'"
                )
                return True
            else:
                logger.debug(
                    f"Error: Unexpected invalid meaning object for word {word} with reading"
                    f" {reading}: {meaning_action}"
                )
                return False

    new_tasks_count = 0

    def handle_return_word_tuples():
        nonlocal processed_word_tuples
        logger.debug(f"Returning processed word tuples: {processed_word_tuples}")
        # filter out any None values from processed_word_tuples
        update_word_list_in_dict(processed_word_tuples)

    need_update_note = False

    def create_result_handler(word_index, word):
        def handle_result(_: bool):
            logger.debug(f"Task completed for word {word} (index {word_index})")
            # At this point, processed_word_tuples[word_index] should have the final result for
            # this word. Only update the word list after all tasks complete
            # Don't call update_word_list_in_dict here

        return handle_result

    word_list_task_count = 0

    for i, word_tuple in enumerate(word_tuples):
        if mw.progress.want_cancel():
            break
        if not isinstance(word_tuple, (tuple, list)):
            logger.debug(f"Error: Invalid word tuple at index {i}: {word_tuple}")
            continue
        word = ""
        reading = ""
        if len(word_tuple) == 4:
            # If this the id is a negative number, it means the word was added as a new note
            # and should try to set the id to an actual note.id, if the note was created
            try:
                fake_note_id = int(word_tuple[3])
            except Exception as e:
                logger.debug(f"Error: Invalid third value in word tuple: {word_tuple}, {e}")
                continue
            if fake_note_id < 0:
                logger.debug(
                    f"Trying to set new note ID found in word tuple at index {i}:"
                    f" {word_tuple} to actual note.id"
                )
                # Is the current note the holder of this fake Id?
                unfake_note = None
                if current_note[new_note_id_field] == str(fake_note_id):
                    # Yes, so we can use the current note's ID
                    unfake_note = current_note
                    if unfake_note.id in notes_to_update_dict:
                        unfake_note = notes_to_update_dict[unfake_note.id]
                    logger.debug(
                        f"Current note ID {current_note.id} matches fake note ID"
                        f" {fake_note_id}, updating word tuple at index {i}"
                    )

                else:
                    # Try to find a note with this in its 'new_note_id_field' field
                    nids = mw.col.find_notes(
                        f'''"note:{note_type['name']}" "{new_note_id_field}:{fake_note_id}"'''
                    )
                    if len(nids) == 1:
                        note_id = nids[0]
                        # Found a note, set the note_id into the word_tuple
                        logger.debug(
                            f"Found note with ID {note_id} for new note ID {fake_note_id},"
                            f" updating word tuple at index {i}"
                        )
                        unfake_note = mw.col.get_note(note_id)
                    elif len(nids) > 1:
                        logger.debug(
                            f"Error: Found multiple notes with new note ID {fake_note_id},"
                            " can't determine which one to use"
                        )
                        continue
                    else:
                        logger.debug(
                            f"Error: No note found with new note ID {fake_note_id}, this note"
                            " was never added doesn't exist"
                        )
                        continue
                if unfake_note is not None:
                    # Set the new note ID to the actual note ID
                    word_tuple = (word_tuple[0], word_tuple[1], word_tuple[2], unfake_note.id)
                    # Update the processed word tuples with the new note ID
                    processed_word_tuples[i] = word_tuple
                    logger.debug(
                        f"Setting new note ID to {unfake_note.id} for word tuple at index {i}:"
                        f" {word_tuple}"
                    )
                    # Also remove the fake id from the note now, so we don't try doing this again
                    if unfake_note.id in notes_to_update_dict:
                        # If the note was already updated, update the note in the dict
                        notes_to_update_dict[unfake_note.id][new_note_id_field] = ""
                    else:
                        unfake_note[new_note_id_field] = ""
                        notes_to_update_dict[unfake_note.id] = unfake_note
                    need_update_note = True
                else:
                    logger.debug(
                        f"Error: No note found to un-fake the new note ID {fake_note_id} for"
                        f" word tuple at index {i}"
                    )
                    continue
            if not replace_existing:
                # Not replacing existing word links
                continue
            else:
                word, reading, _, _ = word_tuple
        elif len(word_tuple) == 2:
            word, reading = word_tuple
        else:
            logger.debug(f"Error: Invalid word tuple length at index {i}: {word_tuple}")
            continue
        if not word or not reading:
            logger.debug(f"Error: Empty word or reading at index {i}: {word_tuple}")
            continue

        logger.debug(f"Processing word tuple {word_tuple} at index {i}")

        new_tasks_count += 1

        def handle_op_error(e: Exception):
            logger.error(f"Error processing word tuple {word_tuple} at index {i}: {e}")
            print_error_traceback(e, logger)

        handle_op_result = create_result_handler(i, word)

        process_word_tuple: Callable[..., Coroutine[Any, Any, bool]] = make_inner_bulk_op(
            config=config,
            op=match_op,
            rate_limit=rate_limit,
            progress_updater=progress_updater,
            handle_op_error=handle_op_error,
            handle_op_result=handle_op_result,
            cancel_state=cancel_state,
        )
        if mw.progress.want_cancel():
            break
        task: asyncio.Task = asyncio.create_task(
            process_word_tuple(
                # task_index is consumed by process_op in make_inner_bulk_op and not passed back!
                task_index=len(tasks),
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
                # the below kwargs are passed back to the match_op function (and any more if we were
                # to add them)
                word=word,
                reading=reading,
                word_index=i,
            )
        )
        progress_updater.increment_counts(
            total_tasks=1,
        )
        # Show progress as tasks are being gathered, this too can take a bit
        if len(tasks) % 5 == 0:
            progress_updater.update_progress()
        tasks.append(task)
        note_tasks.append(task)
        word_list_task_count += 1

    # After all tasks are created, add one final task
    if word_list_task_count > 0 or need_update_note:
        logger.debug(f"Adding final update task after processing {len(note_tasks)} word tasks")

        async def final_update_task():
            # Wait for all word-specific tasks to complete
            await asyncio.gather(*note_tasks)
            logger.debug(
                "All word tasks completed, updating word list with final"
                f" processed_word_tuples: {processed_word_tuples}"
            )
            update_word_list_in_dict(processed_word_tuples)

        # Add this task, but don't add it to note_tasks to avoid circular waiting
        tasks.append(asyncio.create_task(final_update_task()))

    logger.debug(f"Final processed word tuples: {processed_word_tuples}")
    if not note_tasks and need_update_note:
        logger.debug(
            "No word tasks were created, but we need to update the note with final_word_tuples"
        )

        # If we ended up skipping all word tuples, we still may need to update the note
        # Create a dummy task that'll trigger calling handle_return_word_tuples
        async def run_dummy_task():
            await asyncio.sleep(0)
            progress_updater.increment_counts(
                notes_done=1,
            )
            handle_return_word_tuples()

        note_tasks.append(asyncio.create_task(run_dummy_task()))


def match_words_to_notes_for_note(
    config: dict,
    note: Note,
    tasks: list[asyncio.Task],
    edited_nids: list[NoteId],
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    progress_updater: AsyncTaskProgressUpdater,
    cancel_state: CancelState,
) -> None:
    """
    Match words to notes for a single note.

    Args:
        config (dict): Addon config
        note (Note): The note to match words for.
        tasks (list): List of asyncio tasks to append to. Will be mutated by this function.
        notes_to_add_dict (dict): Dict of new notes for unmatched words. Will be mutated by this
            function. Used to also check if the operation has already created something it should
            reuse.
        notes_to_update_dict (dict): Dict to append notes to be updated with new meanings and also
            to get an already updated note for additional changes if needed. Will be mutated by this
            function.
        increment_done_tasks (Callable[..., None]): Used for progress dialog updating
        increment_in_progress_tasks (Callable[..., None]): Used for progress dialog updating
        get_progress (Callable[..., str]): Used for progress dialog updating
    """
    if not note:
        logger.error("Error: No note provided for matching words")
        return

    if not config:
        logger.error("Error: Missing addon configuration")
        return

    replace_existing = config.get("replace_existing_matched_words", False)

    note_type = note.note_type()
    if not note_type:
        logger.error(f"Error: Note {note.id} is missing note type")
        return

    furigana_sentence_field = get_field_config(config, "furigana_sentence_field", note_type)
    if not furigana_sentence_field:
        logger.error("Error: Missing sentence field in config")
        return

    if furigana_sentence_field not in note:
        logger.error(f"Error: Note is missing the sentence field '{furigana_sentence_field}'")
        return
    sentence = note[furigana_sentence_field]
    if not sentence:
        logger.error(f"Error: Note's sentence field '{furigana_sentence_field}' is empty")
        return

    word_lists_to_process = config.get("word_lists_to_process", {})
    if not word_lists_to_process:
        logger.error("Error: No word lists to process in the config")
    if not isinstance(word_lists_to_process, dict):
        logger.error("Error: Invalid word lists format in the config, expected a dictionary")
        return
    # Filter the WORD_LISTS based on the config
    word_list_keys = [wl for wl in WORD_LISTS if word_lists_to_process.get(wl, False)]

    word_tuples = []
    note_tasks: list[asyncio.Task] = []
    # Get the word tuples from the note
    word_list_field = get_field_config(config, "word_list_field", note_type)
    if word_list_field in note:
        try:
            word_list_dict = json.loads(note[word_list_field])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from word list field: {e}")
            # tag note
            note.add_tag("invalid_word_list_json")
            if note.id not in notes_to_update_dict:
                notes_to_update_dict[note.id] = note
            word_list_dict = {}
        if not isinstance(word_list_dict, dict):
            logger.error("Error: Invalid word list format in the note, expected a dictionary")
            return

        # Make a task for waiting for until all tasks for a single note are done before
        # updating the note
        async def wait_for_tasks(
            all_note_tasks: list[asyncio.Task],
            current_note: Note,
            updated_word_list_dict: dict[str, list[FinalWordTuple]],
        ):
            await asyncio.gather(*all_note_tasks)
            if current_note.id in notes_to_update_dict:
                logger.debug(f"Updating note {note.id} with new word list")
                current_note = notes_to_update_dict[current_note.id]
                edited_nids.append(current_note.id)
            logger.debug(
                f"Updating note {note.id} with word list field '{word_list_field}'"
                f"with dict: {updated_word_list_dict}"
            )
            current_note = notes_to_update_dict.get(current_note.id, current_note)
            new_word_list = word_lists_str_format(updated_word_list_dict)
            if new_word_list is not None:
                current_note[word_list_field] = new_word_list
                if current_note.id not in notes_to_update_dict:
                    notes_to_update_dict[current_note.id] = current_note
            progress_updater.increment_counts(
                notes_done=1,
            )

        def make_word_list_updater(current_key):
            def update_function(updated_tuples: list[ProcessedWordTuple]):
                logger.debug(
                    f"Updating word list for key '{current_key}' with tuples: {updated_tuples}"
                )
                word_list_dict[current_key] = [wt for wt in updated_tuples if wt is not None]

            return update_function

        encountered_words = set()
        for word_list_key in word_list_keys:
            # Go through each list and replace the key in the dict with the result
            word_tuples = word_list_dict.get(word_list_key, [])
            # Check if any words have already been encountered
            for wt in word_tuples:
                word = wt[0]
                if word in encountered_words:
                    # remove word from word_tuples
                    word_tuples.remove(wt)
                else:
                    encountered_words.add(word)
            if not isinstance(word_tuples, list):
                logger.error(
                    f"Error: Invalid word list format for key '{word_list_key}' in the note"
                )
                continue
            update_word_list_in_dict = make_word_list_updater(word_list_key)
            match_words_to_notes(
                config=config,
                current_note=note,
                word_tuples=word_tuples,
                word_list_key=word_list_key,
                sentence=sentence,
                tasks=tasks,
                note_tasks=note_tasks,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
                progress_updater=progress_updater,
                cancel_state=cancel_state,
                update_word_list_in_dict=update_word_list_in_dict,
                note_type=note_type,
                replace_existing=replace_existing,
            )
        if note_tasks:
            # Create a task to wait for all note tasks to finish
            tasks.append(
                asyncio.create_task(
                    wait_for_tasks(
                        all_note_tasks=note_tasks,
                        current_note=note,
                        updated_word_list_dict=word_list_dict,
                    )
                )
            )
        return
    else:
        logger.error(f"Error: Note is missing the word list field '{word_list_field}'")
        return


def bulk_match_words_to_notes(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
):
    """
    Bulk match words to notes for selected notes.

    Args:
        col (Collection): The Anki collection to operate on.
        notes (Sequence[Note]): List of notes to process.
        notes_to_add_dict (list): List to append new notes to be created for unmatched words. Will
            be mutated by this function.

    Returns:
        Generator[asyncio.Task]: A generator yielding asyncio tasks for matching words to notes.
    """
    config = mw.addonManager.getConfig(__name__)
    if not config:
        logger.error("Error: Missing addon configuration")
        return
    model = config.get("match_words_model", "")
    message = "Matching words"
    inner_op = match_words_to_notes_for_note
    return bulk_nested_notes_op(
        message=message,
        config=config,
        bulk_inner_op=inner_op,
        col=col,
        notes=notes,
        edited_nids=edited_nids,
        progress_updater=progress_updater,
        notes_to_add_dict=notes_to_add_dict,
        notes_to_update_dict=notes_to_update_dict,
        model=model,
    )


def match_words_to_notes_from_selected(
    nids: Sequence[NoteId],
    parent: Any,
):
    """
    Match words to notes for selected notes.

    Args:
        nids (Sequence[NoteId]): List of note IDs to process.
        parent (Any): Parent widget for the operation.

    Returns:
        Generator[asyncio.Task]: A generator yielding asyncio tasks for matching words to notes.
    """
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Matching words to notes")
    done_text = "Matched words to notes"
    bulk_op = bulk_match_words_to_notes
    new_notes_op = update_fake_note_ids
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater, new_notes_op)
