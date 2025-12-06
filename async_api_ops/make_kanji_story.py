import json
from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence
from pathlib import Path
from typing import Dict

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from .write_kanji_component_words import KANJI_STORY_COMPONENT_WORDS_LOG
from ..utils import get_field_config

DEBUG = False


def get_kanji_story_from_model(
    config: Dict[str, str],
    kanji: str,
    components: str,
    current_story: str,
) -> str:
    media_path = Path(mw.pm.profileFolder(), "collection.media")
    # Get stored dict of words used for component in the kanji_story_component_words.log file
    with open(Path(media_path, KANJI_STORY_COMPONENT_WORDS_LOG), "r", encoding="utf-8") as f:
        try:
            component_words_dict = json.loads(f.read())
        except json.JSONDecodeError as e:
            print(f"Error reading component words dict: {e}")
            component_words_dict = {}

    component_words = []
    # Get the words for each component
    for component in components.split(","):
        if component in component_words_dict:
            component_words.append(component_words_dict[component])

    return_field = "new_story"

    prompt = (
        f"kanji: {kanji}"
        f"\ncomponent_radicals_or_kanji: {components}"
        f"\nwords_to_use_in_story_for_components: {component_words}"
    )
    if current_story:
        prompt += f"\ncurrent_story_in_japanese: {current_story}"
    prompt += (
        "\n"
        "\nThe kanji is made up of the radicals or kanji listed above."
        "\nFor each of those, there are words that can be used to refer to them in the mnemonic"
        " story for the kanji."
    )
    prompt += (
        "\nComplete the current mnemonic story in Japanese for the kanji using those words."
        if current_story
        else "\nCome up with a new mnemonic story in Japanese for the kanji using those words."
    )
    prompt += (
        "\n 1) You can inflect the component words to fit the sentence better."
        "\n 2) The story should be very short - a single sentence - and include all the"
        " components and then a word for the kanji itself."
        "\n 3) The story should be written in hiragana only."
        "\n 4) Each word and particle should be separated by a space to make it easier to read."
        "\n 5) The component words should be wrapped in <i> tags and the kanji word in <b> tags."
        "\n 5a) Ideally the kanji word be a single word, usually a kunyomi reading. but if there"
        " is no usable kunyomi reading, use a compound word."
        "\n 5b) For a compound word wrap the part where the kanji is used in <b> tags."
        "\n 6) If there are no words for a component, invent a word that fits the component in a"
        " memorable way."
        "\n"
        "\nIMPORTANT: The story should be grammatically correct but does not need to be"
        "grammatically complex, sophisticated or even make sense."
        "instead, it should focus on being memorable by connecting the component words with the"
        "example word in a simple way that is easy to remember and visualize."
        "\n"
        "\nExamples of stories for other kanji:"
        "\n  kanji: 裾"
        "\n  component_radicals_or_kanji: 衤,居"
        "\n  story: <i>ころも</i>の なかに <i>いる</i>と、 ぬけた <b>すそ</b>が ひろがる。"
        "\n"
        "\n  kanji: 熱"
        "\n  component_radicals_or_kanji: 埶,灬"
        "\n  story: <i>どろだんご</i> が <i>れっか</i>したら、たかい <b>ねつ</b> が できる"
        "\n"
        "\n  kanji: 柳"
        "\n  component_radicals_or_kanji: 木,卯"
        "\n  story: <i>き</i>が <i>うさぎの みみ</i>の ように しなやか、<b>やなぎ</b>"
        "\n"
        "\n  kanji: 捩"
        "\n  component_radicals_or_kanji: 扌,戻"
        "\n  story: <i>て</i>が <i>もどせない</i>、そんなに <b>よじっている</b>。"
        "\n"
        "\n  kanji: 移"
        "\n  component_radicals_or_kanji: 禾,多"
        "\n  story: <i>のぎ</i>が <i>おおくて</i>、それを くらに <b>うつして</b>みましょう。"
        "\n"
        "\n  kanji: 侶"
        "\n  component_radicals_or_kanji: 亻,呂"
        "\n  story: <i>ひと</i>の <i>せぼね</i>は あばらぼねの はん<b>りょ</b>だ。"
        "\n"
        "\n  kanji: 宮"
        "\n  component_radicals_or_kanji: 宀,呂"
        "\n  story: <i>したぎ かんむり</i>の したに <i>せぼね</i>の ような はしらが きゅうでんの"
        " いりぐちに たった"
        "\n"
        "\n  kanji: 蹴"
        "\n  component_radicals_or_kanji: 足,就"
        "\n  story: <i>あし</i>の <i>しゅうしょく</i>は ものを <b>ける</b> こと。"
        "\n"
        "\n  kanji: 諭"
        "\n  component_radicals_or_kanji: 言,俞"
        "\n  story: <i>いいたい</i> こと を <i>いやしの こぶね</i> を こぎ ながら つたえる と、"
        "せんちょう が しずに してくれと <b>さとした</b>"
        "\n"
        f'\nReturn the new story in a JSON string as the value of the key "{return_field}".'
    )
    model = config.get("kanji_story_model", "")
    result = get_response(model, prompt)
    if result is None:
        # Return original story unchanged if the cleaning failed
        return current_story
    try:
        return result[return_field]
    except KeyError:
        return current_story


def make_story_for_note(
    config: Dict[str, str],
    note: Note,
    notes_to_add_dict: Dict[str, list[Note]] = {},
) -> bool:
    model = note.note_type()
    if not model:
        if DEBUG:
            print("Missing note type for note", note.id)
        return False

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
        # Check if the value is non-empty
        if components:
            new_story = get_kanji_story_from_model(config, kanji, components, current_story)

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


def bulk_make_stories_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: Dict[str, list[Note]] = {},
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("kanji_story_model", "")
    message = "Updated stories"
    op = make_story_for_note
    return bulk_notes_op(
        message, config, op, col, notes, edited_nids, progress_updater, notes_to_add_dict, model
    )


def make_stories_for_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Making kanji stories")
    done_text = "Updated stories"
    bulk_op = bulk_make_stories_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
