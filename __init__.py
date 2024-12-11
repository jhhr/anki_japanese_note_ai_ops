import os
import sys

from anki import hooks
from anki.notes import Note
from aqt import gui_hooks
from aqt import mw
from aqt.browser import Browser
from aqt.qt import QAction, qconnect, QMenu

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from .utils import get_field_config

from .ops.clean_meaning import (
    clean_meaning_in_note,
    clean_selected_notes,
)
from .ops.translate_field import (
    translate_selected_notes,
    translate_sentence_in_note,
)
from .ops.make_kanji_story import (
    make_stories_for_selected_notes,
    make_story_for_note,
)
from .ops.kanjify_sentence import (
    kanjify_selected_notes,
)


# Function to be executed when the browser menus are initialized
def on_browser_will_show_context_menu(browser: Browser, menu: QMenu):
    # Create a new action for the context menu
    meaning_action = QAction("Clean dictionary meaning", mw)
    translation_action = QAction("Translate sentence", mw)
    kanji_story_action = QAction("Generate kanji story", mw)
    component_words_action = QAction("Kanjify+hiraganaify sentence", mw)
    # Connect the action to the operation
    qconnect(
        meaning_action.triggered,
        lambda: clean_selected_notes(browser.selectedNotes(), parent=browser),
    )
    qconnect(
        translation_action.triggered,
        lambda: translate_selected_notes(browser.selectedNotes(), parent=browser),
    )
    qconnect(
        kanji_story_action.triggered,
        lambda: make_stories_for_selected_notes(browser.selectedNotes(), parent=browser),
    )
    qconnect(
        component_words_action.triggered,
        lambda: kanjify_selected_notes(browser.selectedNotes(), parent=browser),
    )

    ai_menu = menu.addMenu("AI helper")
    # Add the action to the browser's card context menu
    ai_menu.addAction(meaning_action)
    ai_menu.addAction(translation_action)
    ai_menu.addAction(kanji_story_action)
    ai_menu.addAction(component_words_action)


def run_op_on_field_unfocus(changed: bool, note: Note, field_idx: int):
    model = note.note_type()
    model_name = model["name"]
    config = mw.addonManager.getConfig(__name__)

    field_name = model["flds"][field_idx]["name"]
    cur_field_value = note[field_name]

    if model_name == "Kanji draw":
        story_field = get_field_config(config, "story_field", model)
        if field_name == story_field and cur_field_value == "":
            return make_story_for_note(note, config=config)

    if model_name == "Japanese vocab note":
        translated_sentence_field = get_field_config(config, "translated_sentence_field", model)
        if field_name == translated_sentence_field and cur_field_value == "":
            return translate_sentence_in_note(note, config=config)



# Register to card adding hook
hooks.note_will_be_added.append(
    lambda _col, note, _deck_id: clean_meaning_in_note(
        note, config=mw.addonManager.getConfig(__name__), show_warning=False
    )
)
# hooks.note_will_be_added.append(lambda _col, note, _deck_id: translate_sentence_in_note(
# note, config=mw.addonManager.getConfig(__name__)))

# Register to context menu initialization hook
gui_hooks.browser_will_show_context_menu.append(on_browser_will_show_context_menu)

# Register to field unfocus hook
gui_hooks.editor_did_unfocus_field.append(run_op_on_field_unfocus)
