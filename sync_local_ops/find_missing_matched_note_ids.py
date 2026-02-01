import logging

from typing import Union, Sequence, Any

from aqt import mw

from anki.notes import Note, NoteId
from anki.models import NotetypeDict
from aqt.utils import showWarning

from anki.collection import Collection


from ..configuration import (
    RawOneMeaningWordType,
    RawMultiMeaningWordType,
    OneMeaningMatchedWordType,
    MultiMeaningMatchedWordType,
)
from ..utils import get_field_config

from ..async_api_ops.base_ops import (
    AsyncTaskProgressUpdater,
    bulk_notes_op,
    selected_notes_op,
)
from ..async_api_ops.extract_words import word_lists_str_format
from ..async_api_ops.match_words_to_notes import WORD_LISTS, decode_word_list_field

logger = logging.getLogger(__name__)


def find_missing_matched_note_ids_for_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    note_type: NotetypeDict = note.note_type()
    log_prefix = f"Find missing matched note ids--nid:{note.id}--"
    word_list_field = get_field_config(config, "word_list_field", note_type)
    word_lists_to_process = config.get("word_lists_to_process", {})
    if not word_lists_to_process:
        logger.error("Error: No word lists to process in the config")
    if not isinstance(word_lists_to_process, dict):
        logger.error("Error: Invalid word lists format in the config, expected a dictionary")
        return False
    # Filter the WORD_LISTS based on the config
    word_list_keys = [wl for wl in WORD_LISTS if word_lists_to_process.get(wl, False)]

    if word_list_field in note:
        word_list_dict = decode_word_list_field(
            note,
            word_list_field,
            notes_to_update_dict,
            log_prefix,
        )
        if not word_list_dict:
            logger.error(
                f"{log_prefix}Error: Invalid word list format in the note, expected a dictionary"
            )
            return False

        word_list_changed = False
        for word_list_key in word_list_keys:
            # Go through each list and replace the key in the dict with the result
            word_tuples: list[
                Union[
                    RawOneMeaningWordType,
                    RawMultiMeaningWordType,
                    OneMeaningMatchedWordType,
                    MultiMeaningMatchedWordType,
                ]
            ] = word_list_dict.get(word_list_key, [])
            if not isinstance(word_tuples, list):
                logger.error(
                    f"{log_prefix}Error: Invalid word list format for key '{word_list_key}' in"
                    " the note"
                )
                continue
            updated_word_tuples = word_tuples.copy()
            word_tuples_changed = False
            for i, wt in enumerate(word_tuples):
                word = wt[0]
                reading = wt[1]
                multimeaning_index = None
                # A matched word tuple has either 4 or 5 elements
                if len(wt) == 4:
                    word_sort_field_value = wt[2]
                    matched_note_id = wt[3]
                elif len(wt) == 5:
                    # Multi-meaning word with matched note id
                    multimeaning_index = wt[2]
                    word_sort_field_value = wt[3]
                    matched_note_id = wt[4]
                else:
                    # Not a matched word tuple
                    continue
                if matched_note_id is not None:
                    # Find the note for this id
                    nid = mw.col.find_notes(f"nid:{matched_note_id}")
                    if not nid:
                        # No note found, reset word tuple to unmatched
                        logger.debug(
                            f"{log_prefix}No note found for word tuple word='{word}',"
                            f" reading='{reading}', word_sort_field='{word_sort_field_value}',"
                            f" matched_note_id='{matched_note_id}'"
                        )
                        if multimeaning_index is not None:
                            updated_word_tuples[i] = [word, reading, multimeaning_index]
                        else:
                            updated_word_tuples[i] = [word, reading]
                        word_tuples_changed = True
            if word_tuples_changed:
                logger.debug(
                    f"{log_prefix}Updated word list for key '{word_list_key}' to new list:"
                    f" {updated_word_tuples}"
                )
                word_list_dict[word_list_key] = updated_word_tuples
                word_list_changed = True
        if word_list_changed:
            logger.debug(f"{log_prefix}Updating note with new word list data: {word_list_dict}")
            note[word_list_field] = word_lists_str_format(word_list_dict)
            if note.id > 0 and note.id not in notes_to_update_dict:
                notes_to_update_dict[note.id] = note
    return True


def bulk_find_missing_matched_note_ids_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]] = {},
    notes_to_update_dict: dict[NoteId, Note] = {},
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    message = "Finding missing matched note ids"
    op = find_missing_matched_note_ids_for_note
    return bulk_notes_op(
        message,
        config,
        op,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict,
        notes_to_update_dict,
        is_sync_op=True,
    )


def find_missing_matched_note_ids_selected_notes(nids: Sequence[NoteId], parent: Any):
    progress_updater = AsyncTaskProgressUpdater(title="Sync op: Finding missing matched note ids")
    done_text = "Updated missing matched note ids"
    bulk_op = bulk_find_missing_matched_note_ids_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
