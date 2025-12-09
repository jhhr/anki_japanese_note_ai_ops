import os
import sys
import logging
from datetime import datetime

from anki import hooks
from anki.notes import Note, NoteId
from aqt import gui_hooks
from aqt import mw
from aqt.browser import Browser
from aqt.qt import QAction, qconnect, QMenu

# Add the 'lib' directory to sys.path for module imports, modules will import from there
# so this needs to be done before any other imports
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if lib_path not in sys.path:
    sys.path.append(lib_path)

# E402 - module level import not at top of file
from .utils import get_field_config  # noqa: E402


from .async_api_ops.clean_meaning import (  # noqa: E402
    clean_meaning_in_note,
    clean_selected_notes,
)
from .async_api_ops.translate_field import (  # noqa: E402
    translate_selected_notes,
    translate_sentence_in_note,
)
from .async_api_ops.make_kanji_story import (  # noqa: E402
    make_stories_for_selected_notes,
    make_story_for_note,
)
from .async_api_ops.kanjify_sentence import (  # noqa: E402
    kanjify_selected_notes,
)
from .async_api_ops.extract_words import (  # noqa: E402
    extract_words_from_selected_notes,
    extract_words_in_note,
)
from .async_api_ops.match_words_to_notes import (  # noqa: E402
    match_words_to_notes_from_selected,
)


# Initialize root logger for the addon at module load
def setup_addon_logging():
    """Set up the root logger for this addon"""
    addon_logger = logging.getLogger(__name__.split(".")[0])  # Get root addon logger

    # Set initial level (will be updated from config)
    addon_logger.setLevel(logging.ERROR)

    # Prevent propagation to Anki's loggers
    addon_logger.propagate = False


setup_addon_logging()


def create_call_log_handler(function_name: str) -> logging.Handler:
    """Create a new file handler for a specific function call"""
    config = mw.addonManager.getConfig(__name__) or {}

    # Get log level from config
    log_level_str = config.get("log_level", "ERROR")
    log_level = getattr(logging, log_level_str.upper(), logging.ERROR)

    # Update the root addon logger's level to match config
    addon_logger = logging.getLogger(__name__.split(".")[0])
    addon_logger.setLevel(log_level)

    # Check if console logging is enabled
    log_to_console = config.get("log_to_console", False)

    if log_to_console:
        # Create console handler
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        return handler

    # Create logs directory
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(addon_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{function_name}_{timestamp}.log")

    # Create handler
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    return handler


# Function to be executed when the browser menus are initialized
def on_browser_will_show_context_menu(browser: Browser, menu: QMenu):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        logger.addHandler(handler)

    # Create a new action for the context menu
    meaning_action = QAction("Clean dictionary meaning", mw)
    translation_action = QAction("Translate sentence", mw)
    kanji_story_action = QAction("Generate kanji story", mw)
    component_words_action = QAction("Kanjify sentence", mw)
    extract_words_action = QAction("Extract words", mw)
    match_words_action = QAction("Match extracted words to notes", mw)
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
    qconnect(
        extract_words_action.triggered,
        lambda: extract_words_from_selected_notes(browser.selectedNotes(), parent=browser),
    )
    qconnect(
        match_words_action.triggered,
        lambda: match_words_to_notes_from_selected(browser.selectedNotes(), parent=browser),
    )

    ai_menu = menu.addMenu("AI helper")
    if ai_menu is None:
        logger.error("Error: AI helper menu could not be created.")
        return
    # Add the action to the browser's card context menu
    ai_menu.addAction(meaning_action)
    ai_menu.addAction(translation_action)
    ai_menu.addAction(kanji_story_action)
    ai_menu.addAction(component_words_action)
    ai_menu.addAction(extract_words_action)
    ai_menu.addAction(match_words_action)


def run_op_on_field_unfocus(changed: bool, note: Note, field_idx: int):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        logger.addHandler(handler)

    note_type = note.note_type()
    if not note_type:
        return
    note_type_name = note_type["name"]
    config = mw.addonManager.getConfig(__name__)
    if not config:
        logger.error("Error: Missing addon configuration")
        return

    field_name = note_type["flds"][field_idx]["name"]
    cur_field_value = note[field_name]

    if note_type_name == "Kanji draw":
        story_field = get_field_config(config, "story_field", note_type)
        if field_name == story_field and cur_field_value == "":
            return make_story_for_note(config, note, {}, {})

    if note_type_name == "Japanese vocab note":
        translated_sentence_field = get_field_config(config, "translated_sentence_field", note_type)
        if field_name == translated_sentence_field and cur_field_value == "":
            return translate_sentence_in_note(config, note, {}, {})


def run_op_on_add_note(note: Note):
    handler = create_call_log_handler("add_note")
    logger = logging.getLogger(__name__)

    if handler:
        logger.addHandler(handler)

    note_type = note.note_type()
    if not note_type:
        return
    note_type_name = note_type["name"]
    config = mw.addonManager.getConfig(__name__)
    if not config:
        logger.error("Error: Missing addon configuration")
        return

    if note_type_name == "Japanese vocab note":
        if note.has_tag("new_matched_jp_word"):
            # If the note has the tag, don't run the ops as this is happening within the
            # match_words_to_notes and causes some problems
            logger.info("Skipping ops for note with 'new_matched_jp_word' tag")
            return
        notes_to_update_dict: dict[NoteId, Note] = {}
        try:
            clean_meaning_in_note(config, note, {}, notes_to_update_dict)
            extract_words_in_note(config, note, {}, notes_to_update_dict)
        except Exception as e:
            logger.error(
                f"Error in clean_meaning_in_note or extract_words_in_note: {e}", exc_info=True
            )
        if notes_to_update_dict:
            updated_notes = list(notes_to_update_dict.values())
            # Filter out the added note itself from the updated notes
            updated_notes = [n for n in updated_notes if n.id != note.id]
            logger.info(f"Updating {len(updated_notes)} notes after adding new note")
            mw.col.update_notes(updated_notes)


# Register to card adding hook
hooks.note_will_be_added.append(lambda _col, note, _deck_id: run_op_on_add_note(note))

# hooks.note_will_be_added.append(lambda _col, note, _deck_id: translate_sentence_in_note(
# note, config=mw.addonManager.getConfig(__name__)))

# Register to context menu initialization hook
gui_hooks.browser_will_show_context_menu.append(on_browser_will_show_context_menu)

# Register to field unfocus hook
gui_hooks.editor_did_unfocus_field.append(run_op_on_field_unfocus)
