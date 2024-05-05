def get_translated_field_from_chatGPT(sentence):
    return_field = "english_sentence"
    # HTML-keeping prompt
    # keep_html_prompt = f"sentence_to_translate_into_english: {sentence}\n\nTranslate the sentence into English. Copy the HTML structure into the English translation. Return the translation in a JSON string as the value of the key \"{return_field}\". Convert \" characters into ' withing the value to keep the JSON valid."
    no_html_prompt = f'sentence_to_translate_into_english: {sentence}\n\nIgnore any HTML in the sentence.\nReturn an HTML-free English translation of the sentence in a JSON string as the value of the key "{return_field}".'
    result = get_response_from_chatGPT(no_html_prompt, return_field)
    if result is None:
        # If translation failed, return nothing
        return None
    return result


def translate_sentence_in_note(note: Note, config):
    model = mw.col.models.get(note.mid)
    sentence_field = config["sentence_field"][model["name"]]
    translated_sentence_field = config["translated_sentence_field"][model["name"]]
    if debug:
        print("sentence_field in note", sentence_field in note)
        print("translated_sentence_field in note", translated_sentence_field in note)
    # Check if the note has the required fields
    if sentence_field in note and translated_sentence_field in note:
        if debug:
            print("note has fields")
        # Get the values from fields
        sentence = note[sentence_field]
        if debug:
            print("sentence", sentence)
        # Check if the value is non-empty
        if sentence:
            # Call API to get translation
            translated_sentence = get_translated_field_from_chatGPT(sentence)
            if debug:
                print("translated_sentence", translated_sentence)
            if translated_sentence is not None:
                # Update the note with the new value
                note[translated_sentence_field] = translated_sentence
                return True
            return False
        return False
    elif debug:
        print("note is missing fields")
    return False


def bulk_translate_notes_op(col, notes: Sequence[Note], editedNids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Translating sentences"
    op = translate_sentence_in_note
    return bulk_notes_op(message, config, op, col, notes, editedNids)


def translate_selected_notes(nids: Sequence[NoteId], parent: Browser):
    title = "Translating done"
    done_text = "Updated translation"
    bulk_op = bulk_translate_notes_op
    return selected_notes_op(title, done_text, bulk_op, nids, parent)
