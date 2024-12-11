import json
from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence
from pathlib import Path
from typing import Dict

from .base_ops import (
    get_response_from_chat_gpt,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = False

KANJI_STORIES_LOG = '_kanji_stories.txt'
KANJI_STORY_COMPONENT_WORDS_LOG = '_kanji_story_component_words.json'


def write_components_to_file(
note: Note, config: Dict[str, str], show_warning: bool = True
):
    model = note.note_type()

    try:
        components_field = get_field_config(config, "components_field", model)
        kanji_field = get_field_config(config, "kanji_field", model)
        story_field = get_field_config(config, "story_field", model)
    except Exception as e:
        print(e)
        return False

    if DEBUG:
        print("making story in note", note.id)
        print("components_field in note", components_field in note)
        print("kanji_field in note", kanji_field in note)
        print("story_field in note", story_field in note)
    # Check if the note has the required fields
    if components_field in note and kanji_field in note and story_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        components = note[components_field]
        kanji = note[kanji_field]
        current_story = note[story_field]

        media_path = Path(mw.pm.profileFolder(), 'collection.media')
        kanji_msg = f'kanji: {kanji}, components: {components}, story: {current_story}'
        # Append message to kanji_stories.log file in the media folder
        with open(Path(media_path, KANJI_STORIES_LOG), 'a', encoding='utf-8') as f:
            f.write(f'{kanji_msg}\n')
        return True
    elif DEBUG:
        print("note is missing fields")
    return False


def get_component_words_dict_from_chat_gpt():
    media_path = Path(mw.pm.profileFolder(), 'collection.media')

    # Read the file kanji_stories.log in the media folder
    with open(Path(media_path, KANJI_STORIES_LOG), 'r', encoding='utf-8') as f:
        try:
            kanji_list = f.read()
        except FileNotFoundError:
            # Can't do anything if we don't have a kanji list to read
            return False

        prompt = f'''
    In the below list are the kanji that I have learned so far. I have included the components of the kanji as a comma-separated list and the story that I have created to remember the kanji using its components. In the stories the words I use to refer to the components are wrapped in <i> tags. The order in which the words are in the story matches the listed order of the components.

    Make a new list for the radicals only. Each line would be one radical and then the words used to refer to that radical in the stories. 

    Return the complete list in a JSON string as the value of the key "component_list".
    The value in the key should be an dictionary-style object like this:
      {{
        "亻": "ひと"
        "⺍": "きらびやかなかんむり"
        "言": "いう、いうこと、いうかた、いうもの"
      }}
    {kanji_list}'''

    result = get_response_from_chat_gpt(prompt, "component_list")
    if result is not None:
        # First read existing dict from kanji_story_component_words.log file in the media folder
        with open(Path(media_path, KANJI_STORY_COMPONENT_WORDS_LOG), 'r', encoding='utf-8') as f:
            try:
                current_dict = json.loads(f.read())
            except (FileNotFoundError, json.JSONDecodeError):
                current_dict = {}
        # Then write the result to the file merging the old and new dicts
        with open(Path(media_path, KANJI_STORY_COMPONENT_WORDS_LOG), 'w', encoding='utf-8') as f:
            current_dict.update(result)
            f.write(json.dumps(current_dict, indent=2, ensure_ascii=False))

        return True
    return False


def bulk_write_component_words_op(col, notes: Sequence[Note], edited_nids: list):
    # Reset kanji_stories.log file
    media_path = Path(mw.pm.profileFolder(), 'collection.media')
    with open(Path(media_path, KANJI_STORIES_LOG), 'w', encoding='utf-8') as f:
        f.write('')
    config = mw.addonManager.getConfig(__name__)
    msg = "Writing component words to file"
    op = write_components_to_file
    return bulk_notes_op(msg, config, op, col, notes, edited_nids)


def write_components_for_selected_notes(nids: Sequence[NoteId], parent: Browser):
    msg = "Writing component words to file"
    op = bulk_write_component_words_op
    return selected_notes_op(
        msg,
        op,
        nids,
        parent,
        on_success=get_component_words_dict_from_chat_gpt,
    )
