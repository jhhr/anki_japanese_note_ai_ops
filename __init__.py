
from anki import hooks
from aqt import mw
import sys
import os
from aqt.browser import Browser
from aqt import gui_hooks
from aqt.qt import QAction, qconnect

from .clean_meaning import (
    clean_meaning_in_note,
    clean_selected_notes,
)
from .translate_field import (
    translate_selected_notes,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))


# Function to be executed when the browser menus are initialized
def on_browser_menus_did_init(browser: Browser):
    # Create a new action for the browser menu
    meaning_action = QAction("Clean dictionary meaning", mw)
    translation_action = QAction("Translate sentence", mw)
    # Connect the action to the convert_selected_notes function
    qconnect(
        meaning_action.triggered,
        lambda: clean_selected_notes(browser.selectedNotes(), parent=browser),
    )
    qconnect(
        translation_action.triggered,
        lambda: translate_selected_notes(browser.selectedNotes(), parent=browser),
    )
    # Add the action to the browser's card context menu
    browser.form.menuEdit.addAction(meaning_action)
    browser.form.menuEdit.addAction(translation_action)


# Register to card adding hook
hooks.note_will_be_added.append(
    lambda _col, note, _deck_id: clean_meaning_in_note(
        note, config=mw.addonManager.getConfig(__name__)
    )
)
# hooks.note_will_be_added.append(lambda _col, note, _deck_id: translate_sentence_in_note(
# note, config=mw.addonManager.getConfig(__name__)))

# Register to browser menu initialization hook
gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
