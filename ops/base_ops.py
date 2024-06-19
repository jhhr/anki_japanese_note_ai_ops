import json
from collections.abc import Sequence

from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from openai import OpenAI

DEBUG = True

api_key = mw.addonManager.getConfig(__name__)["api_key"]
client = OpenAI(api_key=api_key)


def get_response_from_chat_gpt(prompt, return_field):
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
                "content": "You are a helpful assistant for creating flash cards in Anki for Japanese studying. You are designed to output JSON.",
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
        result = json.loads(json_result)[return_field]
        if DEBUG:
            print("Parsed result from json", result)
        return result
    except Exception:
        print(f"Could not parse {return_field} from json_result", json_result)
        return None


def extract_json_string(response_text):
    # Add logic to extract the cleaned meaning from the GPT response
    # You may need to parse the JSON or use other string manipulation techniques
    # based on the structure of the response.

    # For simplicity, let's assume that the cleaned meaning is surrounded by curly braces in the response.
    start_index = response_text.find("{")
    end_index = response_text.find("}")

    if start_index != -1 and end_index != -1:
        return response_text[start_index : end_index + 1]
    else:
        print("Did not return JSON parseable result")
        return response_text


def bulk_notes_op(message, config, op, col, notes: Sequence[Note], edited_nids: list):
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    for note in notes:
        note_was_edited = op(note, config)
        if note_was_edited and edited_nids is not None:
            edited_nids.append(note.id)
        if DEBUG:
            print("note_was_edited", note_was_edited)
            print("editedNids", edited_nids)
    col.update_notes(notes)
    return col.merge_undo_entries(pos)


def selected_notes_op(done_text, bulk_op, nids: Sequence[NoteId], parent: Browser):
    edited_nids = []
    return (
        CollectionOp(
            parent=parent,
            op=lambda col: bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                editedNids=edited_nids,
            ),
        )
        .success(
            lambda out: tooltip(
                f"{done_text} in {len(edited_nids)}/{len(nids)} selected notes.",
                parent=parent,
                period=5000,
            )
        )
        .run_in_background()
    )
