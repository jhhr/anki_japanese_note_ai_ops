import json
from collections.abc import Sequence
from openai import OpenAI
from anki import hooks
from aqt import mw
from aqt.browser import Browser
from anki.notes import Note, NoteId
from aqt import gui_hooks
from aqt.qt import QAction, qconnect
from aqt.operations import CollectionOp
from aqt.utils import showInfo
import sys
import os

sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lib"))

debug = True
api_key = mw.addonManager.getConfig(__name__)['api_key']
client = OpenAI(api_key=api_key)


def get_single_meaning_from_chatGPT(vocab, sentence, dict_entry):
    prompt = f"word: {vocab}\nsentence: {sentence}\ndictionary_entry_for_word_with_possibly_multiple_meanings: {dict_entry}\n\nReturn the one meaning matching the usage of the word in the sentence in a JSON string as the value of the key \"cleaned_meaning\"."
    if debug:
        print('prompt', prompt)

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant for creating flash cards in Anki for Japanese studying. You are designed to output JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200  # Adjust max_tokens as needed
    )

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(response.choices[0].message.content)
    try:
        result = json.loads(json_result)['cleaned_meaning']
        return result
    except:
        print("Could not parse cleaned_meaning from json_result", json_result)
        # Return original dict_entry
        return dict_entry


def extract_json_string(response_text):
    # Add logic to extract the cleaned meaning from the GPT-3 response
    # You may need to parse the JSON or use other string manipulation techniques
    # based on the structure of the response.

    # For simplicity, let's assume that the cleaned meaning is surrounded by curly braces in the response.
    start_index = response_text.find("{")
    end_index = response_text.find("}")

    if start_index != -1 and end_index != -1:
        return response_text[start_index:end_index + 1]
    else:
        print("Did not return JSON parseable result")
        return (response_text)


# Function to be executed when the add_cards_did_add_note hook is triggered
def clean_meaning_in_note(note: Note, config):
    meaning_field = config['meaning_field']
    word_field = config['word_field']
    sentence_field = config['sentence_field']
    if debug:
        print('cleaning meaning in note', note.id)
        print('meaning_field in note', meaning_field in note)
        print('word_field in note', word_field in note)
        print('sentence_field in note', sentence_field in note)
    # Check if the note has the required fields
    if meaning_field in note and word_field in note and sentence_field in note:
        if debug:
            print('note has fields')
        # Get the values from fields
        dict_entry = note[meaning_field]
        word = note[word_field]
        sentence = note[sentence_field]
        # Check if the value is non-empty
        if dict_entry:
            # Call API to get single meaning from the raw dictionary entry
            modified_meaning_jp = get_single_meaning_from_chatGPT(
                word, sentence, dict_entry)

            # Update the note with the modified meaning-jp value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the meaning was edited
            if modified_meaning_jp != dict_entry:
                return True
            else:
                return False
    elif debug:
        print('note is missing fields')
        return False


def bulk_clean_notes_op(col, notes: Sequence[Note], editedNids: list):
    pos = col.add_custom_undo_entry(
        f"Clean meaning-jp for {len(notes)} notes.")
    config = mw.addonManager.getConfig(__name__)
    for note in notes:
        note_was_edited = clean_meaning_in_note(note, config)
        if note_was_edited and editedNids is not None:
            editedNids.append(note.id)
    col.update_notes(notes)
    return col.merge_undo_entries(pos)


# Function to be executed when the "Clean Dictionary meaning" menu action is triggered
def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    editedNids = []
    CollectionOp(
        parent=parent, op=lambda col: bulk_clean_notes_op(
            col, notes=[mw.col.get_note(nid) for nid in nids], editedNids=editedNids)
    ).success(
        lambda out: showInfo(
            parent=parent,
            title="Cleaning done",
            textFormat="rich",
            text=f"Updated meaning in {len(editedNids)}/{len(nids)} selected notes."
        )
    ).run_in_background()


# Function to be executed when the browser menus are initialized
def on_browser_menus_did_init(browser: Browser):
    # Create a new action for the browser menu
    action = QAction("Clean Dictionary meaning", mw)
    # Connect the action to the convert_selected_notes function
    qconnect(action.triggered, lambda: clean_selected_notes(
        browser.selectedNotes(), parent=browser))
    # Add the action to the browser's card context menu
    browser.form.menuEdit.addAction(action)


# Register to card adding hook
hooks.note_will_be_added.append(lambda _col, note, _deck_id: clean_meaning_in_note(
    note, config=mw.addonManager.getConfig(__name__)))

# Register to browser menu initialization hook
gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
