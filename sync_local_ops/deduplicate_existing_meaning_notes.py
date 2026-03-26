import logging

from typing import Any, Sequence

from aqt import mw
from aqt.utils import showWarning

from anki.notes import Note, NoteId
from anki.collection import Collection

from ..async_api_ops.base_ops import (
    AsyncTaskProgressUpdater,
    bulk_notes_op,
    selected_notes_op,
)
from ..async_api_ops.clean_meaning import get_other_meaning_notes
from ..async_api_ops.match_words_to_notes import deduplicate_notes_list

logger = logging.getLogger(__name__)


def bulk_deduplicate_existing_meaning_notes_op(
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

    processed_group_keys: set[tuple[int, ...]] = set()
    notes_to_remove: list[NoteId] = []

    def deduplicate_existing_meaning_notes_for_note(
        config: dict,
        note: Note,
        notes_to_add_dict: dict[str, list[Note]],
        notes_to_update_dict: dict[NoteId, Note],
    ) -> bool:
        other_meaning_notes = get_other_meaning_notes(
            config=config,
            note=note,
            notes_to_add_dict=None,
            notes_to_update_dict=notes_to_update_dict,
            allow_reupdate_existing=True,
            include_pending_notes=False,
        )
        all_group_notes = [note] + other_meaning_notes
        group_note_ids = sorted({n.id for n in all_group_notes if n.id > 0})
        if not group_note_ids:
            return False

        group_key = tuple(group_note_ids)
        if group_key in processed_group_keys:
            logger.debug(
                f"Skipping dedupe for note {note.id}, group already processed: {group_key}"
            )
            return False
        processed_group_keys.add(group_key)

        filtered_notes, _updated_notes_to_update_dict = deduplicate_notes_list(
            notes_to_filter=all_group_notes,
            config=config,
            notes_to_update_dict=notes_to_update_dict,
        )
        filtered_note_obj_ids = {id(n) for n in filtered_notes}
        removed_notes = [n for n in all_group_notes if id(n) not in filtered_note_obj_ids]

        for removed_note in removed_notes:
            if removed_note.id > 0 and removed_note.id not in notes_to_remove:
                notes_to_remove.append(removed_note.id)

        logger.debug(
            f"Deduped meaning-group for note {note.id}: size {len(all_group_notes)} ->"
            f" {len(filtered_notes)}, removing {len(removed_notes)} notes"
        )
        return bool(removed_notes)

    message = "Deduplicating existing meaning notes"
    return bulk_notes_op(
        message,
        config,
        deduplicate_existing_meaning_notes_for_note,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict,
        notes_to_update_dict,
        notes_to_remove=notes_to_remove,
        is_sync_op=True,
    )


def deduplicate_existing_meaning_notes_selected_notes(nids: Sequence[NoteId], parent: Any):
    progress_updater = AsyncTaskProgressUpdater(
        title="Sync op: Deduplicating existing meaning notes"
    )
    done_text = "Deduplicated existing meaning notes"
    bulk_op = bulk_deduplicate_existing_meaning_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
