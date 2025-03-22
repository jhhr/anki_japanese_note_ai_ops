import json
from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence
import requests

DEBUG = False

api_key = mw.addonManager.getConfig(__name__)["api_key"]

MAX_TOKENS = 2000


def get_response_from_chat_gpt(prompt):
    if DEBUG:
        print("prompt", prompt)

    config = mw.addonManager.getConfig(__name__)

    model = config["model"]

    # Use max_completion_tokens instead of max_tokens for o3

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for processing Japanese text. You are a superlative"
                " expert in the Japanese language and its writing system. You are designed to"
                " output JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": "gpt-4o",
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if any(model.startswith(m) for m in ["o3", "o1"]):
        data["max_completion_tokens"] = MAX_TOKENS
    else:
        data["max_tokens"] = MAX_TOKENS

    # Make the API call
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_text = decoded_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError as je:
        print(f"Error decoding JSON: {je}")
        print("response", response.text)
        return None
    except KeyError as ke:
        print(f"Error extracting content: {ke}")
        print("response", response.text)
        return None

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(content_text)
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


def extract_json_string(content_text):
    # Add logic to extract the cleaned meaning from the GPT response
    # You may need to parse the JSON or use other string manipulation techniques
    # based on the structure of the response.

    # For simplicity, let's assume that the stuff asked for is surrounded by curly braces in the
    # response.
    # Find the first occurrence of "{" and the last occurrence of "}" in the response.
    start_index = content_text.find("{")
    end_index = content_text.rfind("}")

    if start_index != -1 and end_index != -1:
        return content_text[start_index : end_index + 1]
    else:
        print("Did not return JSON parseable result")
        return content_text


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
    extra_callback=None,
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
        .success(lambda out: on_bulk_success(out, done_text, edited_nids, nids, parent, on_success))
        .run_in_background()
    )
