def get_response_from_chatGPT(prompt, return_field):
    if debug:
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
    if debug:
        print("json_result", json_result)
    try:
        result = json.loads(json_result)[return_field]
        if debug:
            print("Parsed result from json", result)
        return result
    except:
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


def bulk_notes_op(message, config, op, col, notes: Sequence[Note], editedNids: list):
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    for note in notes:
        note_was_edited = op(note, config)
        if note_was_edited and editedNids is not None:
            editedNids.append(note.id)
        if debug:
            print("note_was_edited", note_was_edited)
            print("editedNids", editedNids)
    col.update_notes(notes)
    return col.merge_undo_entries(pos)


def selected_notes_op(
    title, done_text, bulk_op, nids: Sequence[NoteId], parent: Browser
):
    editedNids = []
    return (
        CollectionOp(
            parent=parent,
            op=lambda col: bulk_op(
                col, notes=[mw.col.get_note(nid) for nid in nids], editedNids=editedNids
            ),
        )
        .success(
            lambda out: showInfo(
                parent=parent,
                title=title,
                textFormat="rich",
                text=f"{done_text} in {len(editedNids)}/{len(nids)} selected notes.",
            )
        )
        .run_in_background()
    )
