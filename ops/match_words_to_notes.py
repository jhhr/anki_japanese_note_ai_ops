import random
import json
import re
import asyncio
import time
from typing import Union, Sequence, Callable, Any, Coroutine, Optional
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
from ..utils import copy_into_new_note, get_field_config
from ..types import raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type

DEBUG = False
    
WORD_LIST_TO_PART_OF_SPEECH: dict[str, str] = {
  "nouns": "Noun",
  "proper_nouns": "Proper Noun",
  "number": "Number",
  "counter": "Counter",
  "verbs": "Verb",
  "compound_verbs": "Verb",
  "adjectives": "Adjective",
  "adverbs": "Adverb",
  "adjectivals": "Adjectival",
  "particles": "Particle",
  "conjuctions": "Conjunction",
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
        str: A unique ID for the note represinted as a negative integer.
    """
    if not note:
        return 0
    # Use the current epoch time in nanoseconds as a temporary unique ID
    return -time.time_ns() + random.randint(0, 100)

def match_words_to_notes(
    config: dict,
    current_note: Note,
    word_tuples: list[Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]],
    word_list_key: str,
    sentence: str,
    tasks: list[asyncio.Task],
    note_tasks: list[asyncio.Task],
    notes_to_add_dict: dict[str, list[Note]],
    updated_notes_dict: dict[NoteId, Note],
    progress_updater: AsyncTaskProgressUpdater,
    cancel_state: CancelState,
    update_word_list_in_dict: Callable[[list[Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]]], None],
    note_type: NotetypeDict,
    replace_existing: bool = False
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
        updated_notes_dict (dict): Dict to append notes to be updated with new meanings and also
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
        if DEBUG:
            print("No words to match against notes")
        return word_tuples
    if not sentence:
        print("Error: No sentence provided for matching words")
        return word_tuples
    if not config:
        print("Error: Missing addon configuration")
        return word_tuples
    # Get the model name from the config
    model = config.get("match_words_model", "")
    if not model:
        print("Error: Missing match words model in config")
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
        print(f"Error: Missing fields in config: {', '.join(missing_fields)}")
        return word_tuples
    
    processed_word_tuples: list[Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]] = word_tuples.copy()
    
    if DEBUG:
        print(f"match_words_to_notes, notes_to_add_dict before: {notes_to_add_dict}")
    
    def match_op(
            _,
            notes_to_add_dict: dict[str, list[Note]],
            # the **op_args passed by process_op in make_inner_bulk_op and process_word_tuple 
            # below
            word_index: int,
            word: str,
            reading: str,
        ) -> bool:
        """ Process a single word tuple to match it with notes in the collection.
        Args:
            config (dict): Addon configuration
            word_index (int): Index of the word tuple in the original list.
            word (str): The word to match.
            reading (str): The reading of the word to match.
            word_list_key (str): The word list the word came from used for part of speech
        Returns:
            bool: True if the word tuple was processed successfully, False otherwise.
        """
        if DEBUG:
            print((f"match_op, notes_to_add_dict before: {notes_to_add_dict}"))
            print(f"Processing word tuple at index {word_index}: {word}, reading: {reading}")
        nonlocal processed_word_tuples, updated_notes_dict
        # Check for existing suru verbs words including する in either field
        reading_query = f'"{word_reading_field}:{to_hiragana(reading)}"'
        reading_query_suru = f'"{word_reading_field}:{to_hiragana(reading)}する"'
        word_query = f'("{word_kanjified_field}:{word}" OR "{word_normal_field}:{word}"'
        word_query_suru = f'"{word_kanjified_field}:{word}する" OR "{word_normal_field}:{word}する")'
        no_x_in_sort_field = f'-"{word_sort_field}:re:\(x\d\)"'
        query = f'(({word_query} {reading_query}) OR ({word_query_suru} {reading_query_suru})) {no_x_in_sort_field}'
        if DEBUG:
            print(f"Searching for notes with query: {query}")
        note_ids: Sequence[NoteId] = mw.col.find_notes(query)
        matching_new_notes = notes_to_add_dict.get(word, [])
            
        if not note_ids and not matching_new_notes:
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
            marker_regex = f'^{word} ((?:\((?:kun|on)\))?(?:\(r\d+\))?)$'
            marker_note_ids = mw.col.find_notes(f'"{word_sort_field}:re:{marker_regex}"')
            marker_notes = [mw.col.get_note(nid) for nid in marker_note_ids]
            # Additionally, search the notes_to_add_dict to see if any of them have a marker like
            # this, as we'll need to increment the rX number higher than the largest one found
            if word in notes_to_add_dict:
                for added_note in notes_to_add_dict[word]:
                    if word_sort_field in added_note:
                        marker_sort_field = added_note[word_sort_field]
                        if re.match(rf"{marker_regex}", marker_sort_field):
                            # If the marker matches, we can add this note ID to the list of marker notes
                            marker_notes.append(added_note)

            if len(marker_notes) == 1:
                marker_note = marker_notes[0]
                if marker_note and word_sort_field in marker_note:
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
                        new_note[word_sort_field] += f"(r{r_number})"
            elif len(marker_notes) > 1:
                # This ought to be case where there's multiple rX markers for the same word,
                # so get the largest rX number and otherwise use the same (kun)/(on) logic as
                # above
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
                else:
                    # If we found both (kun) and (on) markers, we can't decide which one to use,
                    # so just use the word without any markers and add a tag to the note for this
                    # to be manually checked
                    new_note[word_sort_field] = word
                    new_note.add_tag("check_reading_marker")
                
            new_note[furigana_sentence_field] = sentence
            new_note[meaning_field] = ""
            new_note[part_of_speech_field] = WORD_LIST_TO_PART_OF_SPEECH.get(word_list_key, "")
            new_note[english_meaning_field] = ""
            new_note_id = make_new_note_id(new_note)
            new_note[new_note_id_field] = str(new_note_id)
            if DEBUG:
                print(f"1 notes_to_add_dict before adding new note: {notes_to_add_dict}")
            notes_to_add_dict.setdefault(word, []).append(new_note)
            if DEBUG:
                print(f"1 notes_to_add_dict after adding new note: {notes_to_add_dict}")
            create_meaning_result = clean_meaning_in_note(config, new_note, notes_to_add_dict)
            processed_word_tuples[word_index] = (word, reading, word, new_note_id)
            return create_meaning_result
        
        matching_notes = []
        if note_ids:
            # If we have existing notes that match the word, we should use them
            matching_notes = [updated_notes_dict.get(nid) or mw.col.get_note(nid) for nid in note_ids]
        if matching_new_notes:
            # If we have notes to add that match the word, we should add them to the list of notes
            # to check against, so we are comparing against both existing notes and new notes
            matching_notes.extend(matching_new_notes)
        
        # Get all the meanings from the notes to check against the sentence
        meanings: list[tuple[str, int, NoteId, str, str]] = []
        # meaning is a tuple of (jp_meaning, meaning_number, note_id, example_sentence, en_meaning)
        largest_meaning_index = 0
        note_to_copy = None
        for note in matching_notes:
            if DEBUG:
                print(f"Processing note {note.id} for word {word} with reading {reading}, sort field {note[word_sort_field]}")
            if meaning_field in note:
                meaning = note[meaning_field]
                other_sentence = note[furigana_sentence_field] if furigana_sentence_field in note else ""
                english_meaning = note[english_meaning_field] if english_meaning_field in note else ""
                if meaning:
                    sort_field = note[word_sort_field]
                    # Get the meaning number, if any from sort field, in the form (m1), (m2), etc.
                    match = re.match(r"\(m(\d+)\)\.", sort_field)
                    meaning_number = 0
                    if not note_to_copy:
                        # Ensure we have a note to copy that has a meaning
                        note_to_copy = note
                    if match:
                        meaning_number = int(match.group(1))
                        if meaning_number > largest_meaning_index:
                            largest_meaning_index = meaning_number
                            note_to_copy = note
                    meanings.append((meaning, meaning_number, note.id, other_sentence, english_meaning))
                else:
                    if DEBUG:
                        print(f"Note {note.id} has empty meaning field")
            else:
                if DEBUG:
                    print(f"Note {note.id} is missing meaning field")
        if not meanings:
            if DEBUG:
                print(f"No meanings found for word {word} with reading {reading}")
            # We found notes but all were missing meanings, have to skip this word as it can't
            # processed properly
            return False
        # Sort meanings by the meaning number
        meanings.sort(key=lambda x: x[1])
        meanings_str = "\n".join(f"\n -{i+1}. *jp_meaning*: {meaning[0]}\n -{i+1}. *example_sentence* {sentence}\n -{i+1}. *en_meaning*: {meaning[4]}" for i, meaning in enumerate(meanings))       
        
        prompt = f"""Below are listed some dictionary entry-like _meanings_ for a word along with _examples sentences_ for each meaning, and a _current sentence_ being used for determining which meaning to assign to _the word_. Your task is to determine whether any of the meanings match the usage in the sentence and choose one or more actions perform:
 1. Either, select one of the meanings as matching the how the word is used in the current sentence. You may or may not modify the meaning.
 2. Or, determine that none of the meanings below match how the word is used in the current sentence and a new high quality dictionary-like definition should be created.
 3. And in addition to performing actions 1 or 2, one or more of the meanings should be improved so that its new formulation will better fit both its example sentence and the current sentence.

More details on choosing a meaning:
- First and foremost, the ideal standard for multi-meaning words' definitions should be to slice the space of possible meanings into the smallest possible set of definitions that avoids any possible ambiguity when assigning a word's usage in a sentence to some meaning.
- Secondarily, the meanings should be easy to understand for a language-learner; explanations should use simple speech as much the topic of the word allows. Technical jargon or subtle nuances must be adequately explained though - the actual information on what the meaning *is* mustn't be lost in simplification.
- The inclication of these definitions should then be toward explaining closely related nuances in a single definition and splitting to different definitions when a clear topic border is found.
- Thus, avoid increasing the specificity of existing meanings, except when their current state is low-quality ambiguousness.

THE MEANINGS AND EXAMPLE SENTENCES
{meanings_str}

_The word_: {word}
_Current sentence_: {sentence}

JSON RETURN FORMAT

Return a JSON string of an array of objects, at most one more than there are meanings and at least one. Each object has the following four fields:
{{
  "meaning_number": <integer|null> - The ordinal number of the meaning listed above to signify it is being selected and/or modified. Must be a number for actions 1. and 3. Must be null for action 2.
  "is_matched_meaning": <boolean> - true, to indicate action 1. Otherwise false for actions 2 or 3. Either zero or only one object has this set to true.
  "jp_meaning": <string|null> - If non-null, will be used to replace or create the current meaning in action 1. 2. or 3. Is only null in action 1. when no modification is chose to be done. Must be non-null for action 2.
  "en_meaning": <string|null> - If non-null, will be used to replace or create the current english meaning in action 1. 2. or 3. Must non-null for action 2. In actions 1. and 3 "jp_meaning" and "en_meaning" do not have to simultaenously non-null; only one or other may be modified.
}}

Maximum number of objects is reached when performing action 2. to create a new meaning and action 3. for all other meanings. The minimum number of objects is one: to perform either action 1. or 2. only

For action 1. "meaning_number" and "is_matched_meaning" are required, the other are optional
For action 2. the reverse of 1. "meaning_number" and "is_matched_meaning" must be null, and the others are required
For action 3. "meaning_number" is required, "is_matched_meaning" must be null, at least one of "jp_meaning" and "en_meaning" is required
"""
        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "meaning_number": {"type": "integer"},
                    "is_matched_meaning": {"type": "boolean"},
                    "jp_meaning": {"type": "string"},
                    "en_meaning": {"type": "string"},
                },
                "required": [ "is_matched_meaning",],
            },
        }
        if DEBUG:
            print(f"\n\nmeanings_str: {meanings_str}")
        raw_result = get_response(model, prompt, cancel_state=cancel_state, response_schema=response_schema)
        result = None
        if raw_result is None:
            if DEBUG:
                print("Failed to get a response from the API.")
            # If the prompt failed, return nothing
            return False
        if isinstance(raw_result, dict):
            # this may be a case where the AI decided to return a single object instead of a list
            # we'll try to handle it like that then
            if DEBUG:
                print(f"Warning: Expected a list, got a dict instead. Result: {raw_result}")
            result = [raw_result]  # Wrap it in a list to handle it uniformly, if it's garbage, we'll
            # catch it in the processing below
        elif not isinstance(raw_result, list):
            if DEBUG:
                print(f"Error: Expected a list, got {type(raw_result)} instead. Result: {raw_result}")
            return False
        elif not raw_result:
            if DEBUG:
                print("Error: Result is an empty list, should have at least one object.")
            return False
        else:
            result = raw_result
        valid_meaning_objects = []
        valid_matched_meaning_found = False
        for i, res in enumerate(result):
            if not isinstance(res, dict):
                if DEBUG:
                    print(f"Error: Expected a dict, got {type(res)} instead at index {i}. Result: {res}")
                continue
            if "meaning_number" in res and not isinstance(res["meaning_number"], (int, type(None))):
                if DEBUG:
                    print(f"Error: invalid object at index {i}, 'meaning_number' is not an int or None. Result: {res}")
                continue
            if "is_matched_meaning" in res and not isinstance(res["is_matched_meaning"], bool):
                if DEBUG:
                    print(f"Error: invalid object at index {i}, 'is_matched_meaning' is not a bool. Result: {res}")
                continue
            if "jp_meaning" in res and not isinstance(res["jp_meaning"], (str, type(None))):
                if DEBUG:
                    print(f"Error: invalid object at index {i}, 'jp_meaning' is not a str or None. Result: {res}")
                continue
            if "en_meaning" in res and not isinstance(res["en_meaning"], (str, type(None))):
                if DEBUG:
                    print(f"Error: invalid object at index {i}, 'en_meaning' is not a str or None. Result: {res}")
                continue
            if "is_matched_meaning" in res and "meaning_number" not in res:
                if DEBUG:
                    print(f"Error: invalid object at index {i}, 'is_matched_meaning' is set but 'meaning_number' is not. Result: {res}")
                continue
            if "meaning_number" in res and res["meaning_number"] is not None:
                if res["meaning_number"] < 1 or res["meaning_number"] > len(meanings) + 1:
                    if DEBUG:
                        print(f"Error: invalid object at index {i}, 'meaning_number' is out of range. Result: {res}")
                    continue
            is_matched_meaning = res.get("is_matched_meaning", False)
            meaning_number = res.get("meaning_number", None)
            jp_meaning = res.get("jp_meaning", None)
            en_meaning = res.get("en_meaning", None)
            if is_matched_meaning and meaning_number is not None:
                if meaning_number < 1 or meaning_number > len(meanings) + 1:
                    if DEBUG:
                        print(f"Error: Matched meaning number {meaning_number} is out of range for word {word} with reading {reading}")
                    continue
                if valid_matched_meaning_found:
                    if DEBUG:
                        print(f"Error: More than one matched meaning found for word {word} with reading {reading}")
                    continue
                valid_matched_meaning_found = True
            # Valid action 1
            valid_meaning_objects.append({
                "meaning_number": meaning_number,
                "is_matched_meaning": is_matched_meaning,
                "jp_meaning": jp_meaning,
                "en_meaning": en_meaning
            })
            if not is_matched_meaning and not jp_meaning and not en_meaning:
                if DEBUG:
                    print(f"Error: Meaning object at index {i} is hot a match and is not modifying either meaning. Result: {res}")
                continue
            if not is_matched_meaning and meaning_number is None and (jp_meaning or en_meaning):
                # If either jp_meaning or en_meaning is set, we should treat this as a new meaning
                # though technically the requirement was for both, we'll accept this as it is still useful
                # (semi)valid action 2
                valid_meaning_objects.append({
                    "meaning_number": None,
                    "is_matched_meaning": False,
                    "jp_meaning": jp_meaning,
                    "en_meaning": en_meaning
                })
            if not is_matched_meaning and meaning_number is not None and (jp_meaning or en_meaning):
                # If either jp_meaning or en_meaning is set
                # Valid action 3
                valid_meaning_objects.append({
                    "meaning_number": meaning_number,
                    "is_matched_meaning": False,
                    "jp_meaning": jp_meaning,
                    "en_meaning": en_meaning
                })
        if not valid_meaning_objects:
            if DEBUG:
                print(f"Error: No valid meaning objects found for word {word} with reading {reading}")
            return False
        if not valid_matched_meaning_found and len(valid_meaning_objects) == 1:
            # If we have only one meaning object and it is not a matched meaning, not sure what
            # the AI was thinking, so treat is it as invalid
            if DEBUG:
                print(f"Error: Only one meaning object found for word {word} with reading {reading}, but it is not a matched meaning")
            return False
        if DEBUG:
            print(f"Valid meaning objects for word {word} with reading {reading}: {valid_meaning_objects}")
        # Now we have a list of valid meaning objects, we can process them
        # Process actions 1 and 2 first, since there should be only one or the other
        for meaning_object in valid_meaning_objects:
            meaning_number = meaning_object.get("meaning_number", None)
            is_matched_meaning = meaning_object.get("is_matched_meaning", False)
            jp_meaning = meaning_object.get("jp_meaning", None)
            en_meaning = meaning_object.get("en_meaning", None)
            if DEBUG:
                print(f"Processing meaning object: {meaning_object}")
            if not is_matched_meaning and meaning_number is None and (jp_meaning or en_meaning):
                # Action 2. duplicate the note with the biggest meaning number, incrementing it by 1
                if meanings:
                    largest_meaning_index += 1
                if note_to_copy:
                    if note_to_copy.id in updated_notes_dict:
                        # Replace note object from dict if its there
                        note_to_copy = updated_notes_dict[note_to_copy.id]
                    new_note = copy_into_new_note(note_to_copy)
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
                                    # If we found a (mX) in the sort field, update the largest meaning index
                                    largest_meaning_index = max(largest_meaning_index, int(mx_match.group(1)) + 1)
                    if prev_sort_field and mxRec.search(prev_sort_field):
                        # Replace the existing (mX) with the new meaning number
                        new_note[word_sort_field] = mxRec.sub(f"(m{largest_meaning_index})", prev_sort_field)
                    elif prev_sort_field:
                        if re.search(r"\(\w\d+\)", prev_sort_field):
                            # If there is no (mX) but some other number add the meaning number at the end
                            # And update the note_to_copy, since it should a meaning number now too
                            new_note[word_sort_field] += f"(m{largest_meaning_index})"
                            note_to_copy[word_sort_field] += f"(m{largest_meaning_index -1})"
                        else:
                            # Else, same but add a space
                            new_note[word_sort_field] += f" (m{largest_meaning_index})"
                            note_to_copy[word_sort_field] += f" (m{largest_meaning_index -1})"
                            new_note_id = make_new_note_id(new_note)
                            new_note[new_note_id_field] = new_note_id
                        updated_notes_dict[new_note_id] = note_to_copy
                    # Note to copy was missing sort field some how? Add it now and the meaning numbers
                    else:
                        new_note[word_sort_field] = f"{word} (m{largest_meaning_index})"
                        note_to_copy[word_sort_field] = f"{word} (m{largest_meaning_index -1})"
                        new_note_id = make_new_note_id(new_note)
                        updated_notes_dict[new_note_id] = note_to_copy
                        new_note[new_note_id_field] = new_note_id
                    if DEBUG:
                        print(f"2 notes_to_add_dict before adding new note: {notes_to_add_dict}")
                    notes_to_add_dict.setdefault(word, []).append(new_note)
                    if DEBUG:
                        print(f"2 notes_to_add_dict after adding new note: {notes_to_add_dict}")
                    # Note, new_note.id will be 0 here, we'll instead use the sort field value to
                    # find it after insertion and then update the processed_word_tuples
                    processed_word_tuples[word_index] = (word, reading, new_note[word_sort_field], new_note.id)
                    return True
                else:
                    if DEBUG:
                        print(f"Error: No note to copy for word {word} with reading {reading}")
                    return False
            elif is_matched_meaning and meaning_number is not None:
                # Action 1. update the meaning in the note with the matched meaning
                if DEBUG:
                    print(f"Matched meaning JP:'{jp_meaning}'/EN:'{en_meaning}' for word {word} with reading {reading}")
                # We have a match, so we can update the note with the matched meaning
                matched_note = None
                matched_meaning, _, matched_note_id, _, _ = meanings[meaning_number - 1]
                for note in matching_notes:
                    if note.id == matched_note_id and matched_meaning == note[meaning_field]:
                        # Ensure the matched meaning is the same as in the note to account for id=0
                        matched_note = note
                        break
                if not matched_note:
                    # This shoudln't happen as the meaning came from one of the notes in the list
                    if DEBUG:
                        print(f"Error: Matched note with ID {matched_note_id} not found for word {word} with reading {reading}")
                    return False
                # Add the matched note to processed_word_tuples
                new_word_tuple = (
                    word,
                    reading,
                    matched_note[word_sort_field],
                    matched_note.id
                )
                if DEBUG:
                    print(f"Setting processed word tuple at index {word_index}, with tuple {new_word_tuple}")
                processed_word_tuples[word_index] = new_word_tuple
                # Now we need to update the meaning in the note
                if jp_meaning and matched_note[meaning_field] != jp_meaning.strip():
                    matched_note[meaning_field] = jp_meaning.strip()
                    matched_note.add_tag("updated_jp_meaning")
                    updated_notes_dict[matched_note.id] = matched_note
                if en_meaning and matched_note[english_meaning_field] != en_meaning.strip():
                    matched_note[english_meaning_field] = en_meaning.strip()
                    updated_notes_dict[matched_note.id] = matched_note
                if DEBUG:
                    print(f"Updated note {matched_note.id} with new meaning '{jp_meaning}' and english meaning '{en_meaning}'")
                return True
            elif not is_matched_meaning and meaning_number is not None:
                # Action 3. update the meaning in the note with the new meaning
                matched_note = None
                matched_meaning, _, matched_note_id, _, _ = meanings[meaning_number - 1]
                if DEBUG:
                    print(f"Matched meaning {matched_meaning} and new meaning JP:'{jp_meaning}'/EN:{en_meaning} for word {word} with reading {reading}")
                for note in matching_notes:
                    if note.id == matched_note_id and matched_meaning == note[meaning_field]:
                        # Ensure the matched meaning is the same as in the note to account for id=0
                        matched_note = note
                        break
                if not matched_note:
                    # This shoudln't happen as the meaning came from one of the notes in the list
                    if DEBUG:
                        print(f"Error: Matched note with ID {matched_note_id} not found for word {word} with reading {reading}")
                    return False
                # Add the matched note to processed_word_tuples
                new_word_tuple = (
                    word,
                    reading,
                    matched_note[word_sort_field],
                    matched_note.id
                )
                if DEBUG:
                    print(f"Setting processed word tuple at index {word_index}, with tuple {new_word_tuple}")
                processed_word_tuples[word_index] = new_word_tuple
                # Now we need to update the meaning in the note
                if jp_meaning and matched_note[meaning_field] != jp_meaning.strip():
                    matched_note[meaning_field] = jp_meaning
                    matched_note.add_tag("updated_jp_meaning")
                    updated_notes_dict[matched_note.id] = matched_note
                if en_meaning and matched_note[english_meaning_field] != en_meaning.strip():
                    matched_note[english_meaning_field] = en_meaning
                    updated_notes_dict[matched_note.id] = matched_note
                return True
            else:
              if DEBUG:
                  print(f"Error: Unexpected invalid meaning object for word {word} with reading {reading}: {meaning_object}")
              return False
        # End of match_op function
        if DEBUG:
            print(f"Error: No valid meaning objects processed for word {word} with reading {reading}")
        return False
    
    new_tasks_count = 0
    
    def handle_return_word_tuples():
        nonlocal processed_word_tuples
        if DEBUG:
            print(f"Returning processed word tuples: {processed_word_tuples}")
        update_word_list_in_dict(processed_word_tuples)
    need_update_note = False
    
    def create_result_handler(word_index, word):
        def handle_result(_: bool):
            if DEBUG:
                print(f"Task completed for word {word} (index {word_index})")
            # At this point, processed_word_tuples[word_index] should have the final result for this word
            # Only update the word list after all tasks complete
            # Don't call update_word_list_in_dict here
        return handle_result
    
    word_list_task_count = 0
    
    for i, word_tuple in enumerate(word_tuples):
        if mw.progress.want_cancel():
            break
        if not isinstance(word_tuple, (tuple, list)):
            if DEBUG:
                print(f"Error: Invalid word tuple at index {i}: {word_tuple}")
            continue
        word = ""
        reading = ""
        if len(word_tuple) == 4:
            # If this the id is a negative number, it means the word was added as a new note
            # and should try to set the id to an actual note.id, if the note was created
            if (fake_note_id := int(word_tuple[3])) < 0:
                if DEBUG:
                    print(f"Trying to set new note ID found in word tuple at index {i}: {word_tuple} to actual note.id")
                # Is the current note the holder of this fake Id?
                unfake_note = None
                if current_note[new_note_id_field] == str(fake_note_id):
                    # Yes, so we can use the current note's ID
                    unfake_note = current_note
                    if DEBUG:
                        print(f"Current note ID {current_note.id} matches fake note ID {fake_note_id}, updating word tuple at index {i}")
                    
                else:
                    # Try to find a note with this in its 'new_note_id_field' field
                    nids = mw.col.find_notes(f'''"note:{note_type['name']}" "{new_note_id_field}:{fake_note_id}"''') 
                    if len(nids) == 1:
                        note_id = nids[0]
                        # Found a note, set the note_id into the word_tuple
                        if DEBUG:
                            print(f"Found note with ID {note_id} for new note ID {fake_note_id}, updating word tuple at index {i}")
                        unfake_note = mw.col.get_note(note_id)
                    elif len(nids) > 1:
                        if DEBUG:
                            print(f"Error: Found multiple notes with new note ID {fake_note_id}, can't determine which one to use")
                        continue
                    else:
                        if DEBUG:
                            print(f"Error: No note found with new note ID {fake_note_id}, this note was never added doesn't exist")
                        continue
                if unfake_note is not None:
                    # Set the new note ID to the actual note ID
                    word_tuple = (word_tuple[0], word_tuple[1], word_tuple[2], unfake_note.id)
                    # Update the processed word tuples with the new note ID
                    processed_word_tuples[i] = word_tuple
                    if DEBUG:
                        print(f"Setting new note ID to {unfake_note.id} for word tuple at index {i}: {word_tuple}")
                    # Also remove the fake id from the note now, so we don't try doing this again
                    if unfake_note.id in updated_notes_dict:
                        # If the note was already updated, update the note in the dict
                        updated_notes_dict[unfake_note.id][new_note_id_field] = ""
                    else:
                        unfake_note[new_note_id_field] = ""
                        updated_notes_dict[unfake_note.id] = unfake_note
                    need_update_note = True
                else:
                    if DEBUG:
                        print(f"Error: No note found to un-fake the new note ID {fake_note_id} for word tuple at index {i}")
                    continue
            if not replace_existing:
                # Not replacing existing word links
                continue
            else:
                word, reading, _, _ = word_tuple
        elif len(word_tuple) == 2:
            word, reading = word_tuple
        else:
            if DEBUG:
                print(f"Error: Invalid word tuple length at index {i}: {word_tuple}")
            continue
        if not word or not reading:
            if DEBUG:
                print(f"Error: Empty word or reading at index {i}: {word_tuple}")
            continue
        
        if DEBUG:
            print(f"Processing word tuple {word_tuple} at index {i}")
        
        new_tasks_count += 1
        def handle_op_error(e: Exception):
            print(f"Error processing word tuple {word_tuple} at index {i}: {e}")
            raise e
        handle_op_result = create_result_handler(i, word)
        
        process_word_tuple: Callable[..., Coroutine[Any,Any,bool]] = make_inner_bulk_op(
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
        task: asyncio.Task = asyncio.create_task(process_word_tuple(
            # task_index is consumed by process_op in make_inner_bulk_op and not passed back!
            task_index=len(tasks),
            notes_to_add_dict=notes_to_add_dict,
            # the below kwargs are passed back to the match_op function (and any more if we were
            # to add them)
            word=word,
            reading=reading,
            word_index=i,
          ))
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
        if DEBUG:
            print(f"Adding final update task after processing {len(note_tasks)} word tasks")
        async def final_update_task():
            # Wait for all word-specific tasks to complete
            await asyncio.gather(*note_tasks)
            if DEBUG:
                print(f"All word tasks completed, updating word list with final processed_word_tuples: {processed_word_tuples}")
            update_word_list_in_dict(processed_word_tuples)
        
        # Add this task, but don't add it to note_tasks to avoid circular waiting
        tasks.append(asyncio.create_task(final_update_task()))

    if DEBUG:
        print(f"Final processed word tuples: {processed_word_tuples}")
    if not note_tasks and need_update_note:
        if DEBUG:
            print("No word tasks were created, but we need to update the note with processed_word_tuples")
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
    updated_notes_dict: dict[NoteId, Note],
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
        updated_notes_dict (dict): Dict to append notes to be updated with new meanings and also
            to get an already updated note for additional changes if needed. Will be mutated by this
            function.
        increment_done_tasks (Callable[..., None]): Used for progress dialog updating
        increment_in_progress_tasks (Callable[..., None]): Used for progress dialog updating
        get_progress (Callable[..., str]): Used for progress dialog updating
    """
    if not note:
        print("Error: No note provided for matching words")
        return
    
    if not config:
        print("Error: Missing addon configuration")
        return
      
    replace_existing = config.get("replace_existing_matched_words", False)
    
    note_type = note.note_type()
    if not note_type:
        print(f"Error: Note {note.id} is missing note type")
        return
    
    furigana_sentence_field = get_field_config(
        config, "furigana_sentence_field", note_type
    )
    if not furigana_sentence_field:
        print("Error: Missing sentence field in config")
        return
    
    if furigana_sentence_field not in note:
        print(f"Error: Note is missing the sentence field '{furigana_sentence_field}'")
        return
    sentence = note[furigana_sentence_field]
    if not sentence:
        print(f"Error: Note's sentence field '{furigana_sentence_field}' is empty")
        return
    
    word_lists_to_process = config.get("word_lists_to_process", {})
    if not word_lists_to_process:
        print("Error: No word lists to process in the config")
    if not isinstance(word_lists_to_process, dict):
        print("Error: Invalid word lists format in the config, expected a dictionary")
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
            print(f"Error decoding JSON from word list field: {e}")
            word_list_dict = {}
        if not isinstance(word_list_dict, dict):
            print(f"Error: Invalid word list format in the note, expected a dictionary")
            return
        
        
        # Make a task for waiting for until all tasks for a single note are done before
        # updating the note
        async def wait_for_tasks(
              all_note_tasks: list[asyncio.Task],
              current_note: Note,
              updated_word_list_dict: dict[str, list[Union[raw_one_meaning_word_type, matched_word_type]]],
            ):
            await asyncio.gather(*all_note_tasks)
            if current_note.id in updated_notes_dict:
                if DEBUG:
                    print(f"Updating note {note.id} with new word list")
                current_note = updated_notes_dict[current_note.id]
                edited_nids.append(current_note.id)
            if DEBUG:
                print(f"Updating note {note.id} with word list field '{word_list_field}'"
                      f"with dict: {updated_word_list_dict}")
            current_note = updated_notes_dict.get(current_note.id, current_note)
            new_word_list = word_lists_str_format(updated_word_list_dict)
            if new_word_list is not None:
                current_note[word_list_field] = new_word_list 
                updated_notes_dict[current_note.id] = current_note
            progress_updater.increment_counts(
                notes_done=1,
            )
        
        def make_word_list_updater(current_key):
            def update_function(updated_tuples):
                if DEBUG:
                    print(f"Updating word list for key '{current_key}' with tuples: {updated_tuples}")
                word_list_dict[current_key] = updated_tuples
            return update_function
        
        for word_list_key in word_list_keys:
            # Go through each list and replace the key in the dict with the result
            word_tuples = word_list_dict.get(word_list_key, [])
            if not isinstance(word_tuples, list):
                print(f"Error: Invalid word list format for key '{word_list_key}' in the note")
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
                updated_notes_dict=updated_notes_dict,
                progress_updater=progress_updater,
                cancel_state=cancel_state,
                update_word_list_in_dict=update_word_list_in_dict,
                note_type=note_type,
                replace_existing=replace_existing,
            )
        if note_tasks:
            # Create a task to wait for all note tasks to finish
            tasks.append(asyncio.create_task(wait_for_tasks(
                all_note_tasks=note_tasks,
                current_note=note,
                updated_word_list_dict=word_list_dict,
            )))
        return
    else:
        print(f"Error: Note is missing the word list field '{word_list_field}'")
        return



def bulk_match_words_to_notes(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    notes_to_add_dict: dict[str, list[Note]],
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
        print("Error: Missing addon configuration")
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
        notes_to_add_dict=notes_to_add_dict,
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
    done_text = "Matched words to notes"
    bulk_op = bulk_match_words_to_notes
    return selected_notes_op(done_text, bulk_op, nids, parent)