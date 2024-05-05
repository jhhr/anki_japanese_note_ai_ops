
def clean_meaning_in_note(note: Note, config):
    model = mw.col.models.get(note.mid)
    meaning_field = config["meaning_field"][model["name"]]
    word_field = config["word_field"][model["name"]]
    sentence_field = config["sentence_field"][model["name"]]
    if debug:
        print("cleaning meaning in note", note.id)
        print("meaning_field in note", meaning_field in note)
        print("word_field in note", word_field in note)
        print("sentence_field in note", sentence_field in note)
    # Check if the note has the required fields
    if meaning_field in note and word_field in note and sentence_field in note:
        if debug:
            print("note has fields")
        # Get the values from fields
        dict_entry = note[meaning_field]
        word = note[word_field]
        sentence = note[sentence_field]
        # Check if the value is non-empty
        if dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            modified_meaning_jp = get_single_meaning_from_chatGPT(
                word, sentence, dict_entry
            )

            # Update the note with the new value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the we changed something
            if modified_meaning_jp != dict_entry:
                return True
            return False
        return False
    elif debug:
        print("note is missing fields")
    return False

def bulk_clean_notes_op(col, notes: Sequence[Note], editedNids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(message, config, op, col, notes, editedNids)


def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    title = "Cleaning done"
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(title, done_text, bulk_op, nids, parent)
