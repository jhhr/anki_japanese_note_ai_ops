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
from .write_kanji_component_words import KANJI_STORY_COMPONENT_WORDS_LOG
from ..utils import get_field_config

DEBUG = False


def get_kanji_story_from_chat_gpt(kanji, components, current_story):
        media_path = Path(mw.pm.profileFolder(), 'collection.media')
        # Get stored dict of words used for component in the kanji_story_component_words.log file
        with open(Path(media_path, KANJI_STORY_COMPONENT_WORDS_LOG), 'r', encoding='utf-8') as f:
            try:
                component_words_dict = json.loads(f.read())
            except json.JSONDecodeError as e:
                print(f'Error reading component words dict: {e}')
                component_words_dict = {}

        component_words = []
        # Get the words for each component
        for component in components.split(','):
            if component in component_words_dict:
                component_words.append(component_words_dict[component])


        return_field = "new_story"
        prompt = f'\
        kanji: {kanji}\
        component_radicals_or_kanji: {components}\
        words_to_use_in_story_for_components: {component_words}\
        current_story_in_japanese: {current_story}\
        \
        The kanji is made up of the radicals or kanji listed above.\
        For each of those, there are words that can be used to refer to them in the mnemonic story for the kanji.\
        Come up with a new mnemonic story in Japanese for the kanji using those words.\
        1) You can inflect the component words to fit the sentence better.\
        2) The story should be very short - no more than 2 sentences - and include all the components and then a word for the kanji itself.\
        3) The story should be written in hiragana only.\
        4) Each word and particle should be separated by a space to make it easier to read.\
        5) The component words should be wrapped in <i> tags and the kanji word in <b> tags.\
        5a) Ideally the kanji word be a single word, usually a kunyomi reading. but if there is no usable kunyomi reading, use a compound word.\
        5b) For a compound word wrap the part where the kanji is used in <b> tags.\
        6) If there are no words for a component, invent a word that fits the component in a memorable way.\
        \
        Examples of other kanji:\
        kanji: 袖\
        component_radicals_or_kanji: 衤,由\
        story: <i>ころも</i> は <i>よし</> を いらず、 <b>そで</b> も いらない\
        \
    　　 kanji: 倍\
        component_radicals_or_kanji: 亻,咅\
        story: <i>ひと</i> が <i>つばをはく、</i>つば から はえて、ひと が 2<b>ばい</b> に なる\
        \
        kanji: 財\
        component_radicals_or_kanji: 貝,才\
        story: ちょっと だけ の <i>おかね</i> を てんさい に あげたら、 まもなく ばくだいな <b>ざい</b>さん を つくる\
        \
        Return the new story in a JSON string as the value of the key "{return_field}".\
        '
        result = get_response_from_chat_gpt(prompt, return_field)
        if result is None:
            # Return original story unchanged if the cleaning failed
            return current_story
        return result

def make_story_for_note(
    note: Note, config: Dict[str, str], show_warning: bool = True
):
    model = mw.col.models.get(note.mid)


    try:
        components_field = get_field_config(config, "components_field", model)
        kanji_field = get_field_config(config, "kanji_field", model)
        story_field = get_field_config(config, "story_field", model)
    except Exception as e:
        print(e)
        if show_warning:
            showWarning(str(e))
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
        # Check if the value is non-empty
        if components:
            new_story = get_kanji_story_from_chat_gpt(
                kanji, components, current_story
            )

            # Update the note with the new value
            note[story_field] = new_story
            # Return success, if the we changed something
            if new_story != current_story:
                return True
            return False
        return False

    elif DEBUG:
        print("note is missing fields")
    return False



def bulk_make_stories_op(col, notes: Sequence[Note], edited_nids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Updated stories"
    op = make_story_for_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids)


def make_stories_for_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated stories"
    bulk_op = bulk_make_stories_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
