import logging
from typing import Any, Sequence

from anki.collection import Collection
from anki.notes import Note, NoteId
from aqt import mw

from ..async_api_ops.base_ops import (
    AsyncTaskProgressUpdater,
    bulk_notes_op,
    selected_notes_op,
)
from ..async_api_ops.match_words_to_notes import (
    get_note_word_match_query,
    get_word_list_query_regex_for_word_and_reading,
)

logger = logging.getLogger(__name__)

ALL_NOTES_MATCHED_TAG = "2-all-notes-matched"
SOME_NOTES_UNMATCHED_TAG = "2-some-notes-unmatched"
NO_NOTES_MATCHABLE_TAG = "2-no-notes-matchable"


def tag_notes_matched_status_for_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
) -> bool:
    """
    Tag a note to indicate whether all, some, or no other notes have been matched to its word.

    Tags applied:
        - ALL_NOTES_MATCHED_TAG: matched_count > 0 and unmatched_count == 0
        - SOME_NOTES_UNMATCHED_TAG: unmatched_count > 0
        - NO_NOTES_MATCHABLE_TAG: matched_count == 0 and unmatched_count == 0
    """
    note_type = note.note_type()
    log_prefix = f"Tag notes matched status--nid:{note.id}--"
    note_word_info = get_note_word_match_query(config, note, note_type, log_prefix)
    if note_word_info is None:
        return False
    target_word, target_reading, word_list_field = note_word_info

    matched_regex = get_word_list_query_regex_for_word_and_reading(
        word=target_word,
        reading=target_reading,
        with_processed="only_processed",
    )
    unmatched_regex = get_word_list_query_regex_for_word_and_reading(
        word=target_word,
        reading=target_reading,
        with_processed="only_unprocessed",
    )
    note_type_name = note_type["name"]
    matched_query = f'"note:{note_type_name}" "{word_list_field}:re:{matched_regex}"'
    unmatched_query = f'"note:{note_type_name}" "{word_list_field}:re:{unmatched_regex}"'

    matched_count = len(mw.col.find_notes(matched_query))
    unmatched_count = len(mw.col.find_notes(unmatched_query))
    logger.debug(
        f"{log_prefix}Word '{target_word}' (reading '{target_reading}'):"
        f" matched_count={matched_count}, unmatched_count={unmatched_count}"
    )

    note.remove_tag(ALL_NOTES_MATCHED_TAG)
    note.remove_tag(SOME_NOTES_UNMATCHED_TAG)
    note.remove_tag(NO_NOTES_MATCHABLE_TAG)

    if matched_count > 0 and unmatched_count == 0:
        note.add_tag(ALL_NOTES_MATCHED_TAG)
    elif unmatched_count > 0:
        note.add_tag(SOME_NOTES_UNMATCHED_TAG)
    else:
        note.add_tag(NO_NOTES_MATCHABLE_TAG)

    if note.id > 0:
        notes_to_update_dict[note.id] = note
    return True


def bulk_tag_notes_matched_status_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]] = {},
    notes_to_update_dict: dict[NoteId, Note] = {},
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        return
    return bulk_notes_op(
        message="Tagging notes matched status",
        config=config,
        op=tag_notes_matched_status_for_note,
        col=col,
        notes=notes,
        edited_nids=edited_nids,
        progress_updater=progress_updater,
        notes_to_add_dict=notes_to_add_dict,
        notes_to_update_dict=notes_to_update_dict,
        is_sync_op=True,
    )


def tag_notes_matched_status_from_selected(
    nids: Sequence[NoteId],
    parent: Any,
):
    """
    Tag selected notes to indicate whether all, some, or no notes in the collection
    have been matched to each note's word.

    Tags applied per note:
        - ALL_NOTES_MATCHED_TAG ("2-all-notes-matched"): all matchable notes have been matched
        - SOME_NOTES_UNMATCHED_TAG ("2-some-notes-unmatched"): at least one note is still unmatched
        - NO_NOTES_MATCHABLE_TAG ("2-no-notes-matchable"): no notes found for this word at all
    """
    progress_updater = AsyncTaskProgressUpdater(title="Sync op: Tagging notes matched status")
    done_text = "Tagged notes matched status"
    bulk_op = bulk_tag_notes_matched_status_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
