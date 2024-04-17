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
    return_field = "cleaned_meaning"
    prompt = f"word: {vocab}\nsentence: {sentence}\ndictionary_entry_for_word_with_possibly_multiple_meanings: {dict_entry}\n\nReturn the one meaning matching the usage of the word in the sentence in a JSON string as the value of the key \"{return_field}\"."
    result = get_response_from_chatGPT(prompt, return_field)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    return result


def get_translated_field_from_chatGPT(sentence):
    return_field = "english_sentence"
    # HTML-keeping prompt
    # keep_html_prompt = f"sentence_to_translate_into_english: {sentence}\n\nTranslate the sentence into English. Copy the HTML structure into the English translation. Return the translation in a JSON string as the value of the key \"{return_field}\". Convert \" characters into ' withing the value to keep the JSON valid."
    no_html_prompt = f"sentence_to_translate_into_english: {sentence}\n\nIgnore any HTML in the sentence.\nReturn an HTML-free English translation of the sentence in a JSON string as the value of the key \"{return_field}\"."
    result = get_response_from_chatGPT(no_html_prompt, return_field)
    if result is None:
        # If translation failed, return nothing
        return None
    return result


def get_response_from_chatGPT(prompt, return_field):
    if debug:
        print('prompt', prompt)

    config = mw.addonManager.getConfig(__name__)

    model = config["model"]

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant for creating flash cards in Anki for Japanese studying. You are designed to output JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000  # Adjust max_tokens as needed
    )

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(response.choices[0].message.content)
    if debug:
        print('json_result', json_result)
    try:
        result = json.loads(json_result)[return_field]
        if debug:
            print('Parsed result from json', result)
        return result
    except:
        print(f"Could not parse {return_field} from json_result", json_result)
        return None


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
    model = mw.col.models.get(note.mid)
    meaning_field = config['meaning_field'][model['name']]
    word_field = config['word_field'][model['name']]
    sentence_field = config['sentence_field'][model['name']]
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

            # Update the note with the new value
            note[meaning_field] = modified_meaning_jp
            # Return success, if the we changed something
            if modified_meaning_jp != dict_entry:
                return True
            return False
        return False
    elif debug:
        print('note is missing fields')
    return False


def bulk_clean_notes_op(col, notes: Sequence[Note], editedNids: list):
    config = mw.addonManager.getConfig(__name__)
    message = f"Cleaning meaning"
    op = clean_meaning_in_note
    return bulk_notes_op(message, config, op, col, notes, editedNids)


# Function to be executed when the "Clean Dictionary meaning" menu action is triggered
def clean_selected_notes(nids: Sequence[NoteId], parent: Browser):
    title = "Cleaning done"
    done_text = "Updated meaning"
    bulk_op = bulk_clean_notes_op
    return selected_notes_op(title, done_text, bulk_op, nids, parent)


# Function to be executed when the add_cards_did_add_note hook is triggered
def translate_sentence_in_note(note: Note, config):
    model = mw.col.models.get(note.mid)
    sentence_field = config['sentence_field'][model['name']]
    translated_sentence_field = config['translated_sentence_field'][model['name']]
    if debug:
        print('sentence_field in note', sentence_field in note)
        print('translated_sentence_field in note',
              translated_sentence_field in note)
    # Check if the note has the required fields
    if sentence_field in note and translated_sentence_field in note:
        if debug:
            print('note has fields')
        # Get the values from fields
        sentence = note[sentence_field]
        if debug:
            print('sentence', sentence)
        # Check if the value is non-empty
        if sentence:
            # Call API to get translation
            translated_sentence = get_translated_field_from_chatGPT(sentence)
            if debug:
                print('translated_sentence', translated_sentence)
            if translated_sentence is not None:
                # Update the note with the new value
                note[translated_sentence_field] = translated_sentence
                return True
            return False
        return False
    elif debug:
        print('note is missing fields')
    return False


def bulk_translate_notes_op(col, notes: Sequence[Note], editedNids: list):
    config = mw.addonManager.getConfig(__name__)
    message = f"Translating sentences"
    op = translate_sentence_in_note
    return bulk_notes_op(message, config, op, col, notes, editedNids)


# Function to be executed when the "Clean Dictionary meaning" menu action is triggered
def translate_selected_notes(nids: Sequence[NoteId], parent: Browser):
    title = "Translating done"
    done_text = "Updated translation"
    bulk_op = bulk_translate_notes_op
    return selected_notes_op(title, done_text, bulk_op, nids, parent)


def bulk_notes_op(message, config, op, col, notes: Sequence[Note], editedNids: list):
    pos = col.add_custom_undo_entry(
        f"{message} for {len(notes)} notes.")
    for note in notes:
        note_was_edited = op(note, config)
        if note_was_edited and editedNids is not None:
            editedNids.append(note.id)
        if debug:
            print('note_was_edited', note_was_edited)
            print('editedNids', editedNids)
    col.update_notes(notes)
    return col.merge_undo_entries(pos)


def selected_notes_op(title, done_text, bulk_op, nids: Sequence[NoteId], parent: Browser):
    editedNids = []
    return CollectionOp(
        parent=parent, op=lambda col: bulk_op(
            col, notes=[mw.col.get_note(nid) for nid in nids], editedNids=editedNids)
    ).success(
        lambda out: showInfo(
            parent=parent,
            title=title,
            textFormat="rich",
            text=f"{done_text} in {len(editedNids)}/{len(nids)} selected notes."
        )
    ).run_in_background()


# Function to be executed when the browser menus are initialized
def on_browser_menus_did_init(browser: Browser):
    # Create a new action for the browser menu
    meaning_action = QAction("Clean dictionary meaning", mw)
    translation_action = QAction("Translate sentence", mw)
    # Connect the action to the convert_selected_notes function
    qconnect(meaning_action.triggered, lambda: clean_selected_notes(
        browser.selectedNotes(), parent=browser))
    qconnect(translation_action.triggered, lambda: translate_selected_notes(
        browser.selectedNotes(), parent=browser))
    # Add the action to the browser's card context menu
    browser.form.menuEdit.addAction(meaning_action)
    browser.form.menuEdit.addAction(translation_action)


# Register to card adding hook
hooks.note_will_be_added.append(lambda _col, note, _deck_id: clean_meaning_in_note(
    note, config=mw.addonManager.getConfig(__name__)))
# hooks.note_will_be_added.append(lambda _col, note, _deck_id: translate_sentence_in_note(
    # note, config=mw.addonManager.getConfig(__name__)))

# Register to browser menu initialization hook
gui_hooks.browser_menus_did_init.append(on_browser_menus_did_init)
