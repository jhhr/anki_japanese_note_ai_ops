import json
from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence
from openai import OpenAI
from pathlib import Path

DEBUG = False

api_key = mw.addonManager.getConfig(__name__)["api_key"]
client = OpenAI(api_key=api_key)


def get_response_from_chat_gpt(prompt):
    if DEBUG:
        print("prompt", prompt)

    config = mw.addonManager.getConfig(__name__)

    model = config["model"]

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for processing Japanese text. You are a superlative expert in the Japanese language and its writing system. You are designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,  # Adjust max_tokens as needed
    )

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(response.choices[0].message.content)
    if DEBUG:
        print("json_result", json_result)
    try:
        result = json.loads(json_result)
        if DEBUG:
            print("Parsed result from json", result)
        return result
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        return None


def extract_json_string(response_text):
    # Add logic to extract the cleaned meaning from the GPT response
    # You may need to parse the JSON or use other string manipulation techniques
    # based on the structure of the response.

    # For simplicity, let's assume that the cleaned meaning is surrounded by curly braces in the response.
    # Find the first occurrence of "{" and the last occurrence of "}" in the response.
    start_index = response_text.find("{")
    end_index = response_text.rfind("}")

    if start_index != -1 and end_index != -1:
        return response_text[start_index : end_index + 1]
    else:
        print("Did not return JSON parseable result")
        return response_text


def bulk_notes_op(message, config, op, col, notes: Sequence[Note], edited_nids: list):
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    total_notes = len(notes)
    note_cnt = 0
    for note in notes:
        try:
            note_was_edited = op(note, config)
        except Exception as e:
            print("Error processing note", note.id, e)
            note_was_edited = False
        note_cnt += 1

        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label=f"{note_cnt}/{total_notes} notes processed",
                value=note_cnt,
                max=total_notes,
            )
        )
        if mw.progress.want_cancel():
            break
        if note_was_edited and edited_nids is not None:
            col.update_note(note)
            col.merge_undo_entries(pos)
            edited_nids.append(note.id)
        if DEBUG:
            print("note_was_edited", note_was_edited)
            print("editedNids", edited_nids)
    return col.merge_undo_entries(pos)

def on_bulk_success(
        out,
        done_text: str,
        edited_nids: Sequence[NoteId],
        nids: Sequence[NoteId],
        parent: Browser,
        extra_callback=None
):
    tooltip(
        f"{done_text} in {len(edited_nids)}/{len(nids)} selected notes.",
        parent=parent,
        period=5000,
    )
    if extra_callback:
        extra_callback()


def selected_notes_op(done_text, bulk_op, nids: Sequence[NoteId], parent: Browser, on_success=None):
    edited_nids = []
    return (
        CollectionOp(
            parent=parent,
            op=lambda col: bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                edited_nids=edited_nids,
            ),
        )
        .success(
            lambda out: on_bulk_success(out, done_text, edited_nids, nids, parent, on_success)
        )
        .run_in_background()
    )
