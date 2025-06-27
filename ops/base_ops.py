import json
import asyncio
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence

DEBUG = False


MAX_TOKENS = 2000

def get_response(model, prompt):
    """Get a response from the appropriate model based on the configuration.

    Args:
        model: The model to use for the request.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if model.startswith("gemini"):
        return get_response_from_gemini(model, prompt)
    elif model.startswith("gpt") or model.startswith("o3") or model.startswith("o1"):
        return get_response_from_openai(model, prompt)
    else:
        print(f"Unsupported model: {model}")
        return None



def get_response_from_gemini(model, prompt):
    """Get a response from Google's Gemini API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if DEBUG:
        print("prompt", prompt)


    # Create the request body
    data = {
        "contents": [
            {
                # "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            },
        ],
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a helpful assistant for processing Japanese text. You are a superlative"
                        " expert in the Japanese language and its writing system. You are designed to"
                        " output JSON."
                    )
                }
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            # "responseSchema": {
            #     "type": "OBJECT",
            #     "properties": {
            #         "some_prop": {
            #             "type": "STRING",
            #         },
            #     },
            #     "required": ["some_prop"],
            # },
        },
    }

    headers = {
        "Content-Type": "application/json",
        # "x-goog-api-key": google_api_key,
    }

    config = mw.addonManager.getConfig(__name__)
    if config is None:
        print("No configuration found for the addon.")
        return None
    google_api_key = config.get("google_api_key", "")
    # Make the API call
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_api_key}",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        # Extract content from Gemini response structure
        content_text = decoded_json["candidates"][0]["content"]["parts"][0]["text"]
    except json.JSONDecodeError as je:
        print(f"Error decoding JSON: {je}")
        print("response", response.text)
        return None
    except KeyError as ke:
        print(f"Error extracting content: {ke}")
        print("response", response.text)
        return None

    # Extract the JSON from the response
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


def get_response_from_openai(model, prompt):
    if DEBUG:
        print("prompt", prompt)

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
    config = mw.addonManager.getConfig(__name__)
    if config is None:
        print("No configuration found for the addon.")
        return None
    openai_api_key = config.get("openai_api_key", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    data = {
        "model": "gpt-4o",
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if any(model.startswith(m) for m in ["o3", "o1"]):
        data["max_completion_tokens"] = f"{MAX_TOKENS}"
    else:
        data["max_tokens"] = f"{MAX_TOKENS}"

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



async def bulk_notes_op(
        message,
        config,
        op,
        col,
        notes: Sequence[Note],
        edited_nids: list,
        model: str = "",
    ):
    """
    Perform a bulk operation on a sequence of notes.
    Args:
        message: A message to display in the progress dialog.
        config: Addon config dict.
        op: The operation function to apply to each note.
        col: The Anki collection object.
        notes: A sequence of Note objects to process.
        edited_nids: A list to store the IDs of edited notes, to be mutated in place.
        model: The AI model to use for the operation, to get rate limit from config.
    """
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    total_notes = len(notes)
    note_cnt = 0
    updated_notes = []
    
    if not model:
        print("Model arg missing in bulk_notes_op, aborting")
        return None
    
    config["rate_limits"] = config.get("rate_limits", {})
    rate_limit = config["rate_limits"].get(model, None)
    
    
    if rate_limit is None:
        print(f"Unsupported model: {model}")
        return None
    
    # If no rate limit is specified, use a sequential approach
    if not rate_limit:
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
                updated_notes.append(note)
                edited_nids.append(note.id)
            if DEBUG:
                print("note_was_edited", note_was_edited)
                print("editedNids", edited_nids)
    else:
        # Async approach with rate limiting
        semaphore = asyncio.Semaphore(rate_limit)
        tasks: list[asyncio.Task] = []
        start_time = time.time()
        tasks_in_progress: int = 0
        tasks_done: int = 0
        start_time = time.time()
        
        # Wrapper function to process a single note
        async def process_note(note, index):
            nonlocal tasks_done, tasks_in_progress
            
            # Calculate time between operations to maintain rate limit
            seconds_per_op = 60.0 / rate_limit
            target_time = start_time + (index * seconds_per_op)
            current_time = time.time()
            
            # If we're ahead of schedule, wait until it's time to process the next note
            if current_time < target_time:
                await asyncio.sleep(target_time - current_time)
            
            # Acquire semaphore to limit concurrent operations
            async with semaphore:
                # Use ThreadPoolExecutor for CPU-bound operations
                with ThreadPoolExecutor(max_workers=1) as executor:
                    def execute_op():
                        nonlocal tasks_in_progress, tasks_done
                        tasks_in_progress += 1
                        try:
                            return op(note, config)
                        except Exception as e:
                            print("Error processing note", note.id, e)
                            return False
                        finally:
                            nonlocal tasks_done
                            tasks_done += 1
                            tasks_in_progress -= 1
                    
                    # Run the operation in a thread
                    note_was_edited = await asyncio.get_event_loop().run_in_executor(
                        executor, execute_op
                    )
                
                progress_msg = f"Processed: {tasks_done}/{len(tasks)} notes\nWaiting response: {tasks_in_progress}"
                elapsed_time = time.time() - start_time
                # estimate time remaining from tasks_done and elapsed_time
                eta_msg = ""
                if tasks_done > 3: 
                    eta = (total_notes - tasks_done) * (elapsed_time / tasks_done)
                    avg_per_op = elapsed_time / tasks_done
                    eta_msg = f"\nETA: {eta:.2f}s avg time per note: {avg_per_op:.2f}s"
                mw.taskman.run_on_main(
                    lambda: mw.progress.update(
                        label=f"{progress_msg}{eta_msg}",
                        value=tasks_done,
                        max=total_notes,
                    )
                )
                
                # Store results
                if note_was_edited and edited_nids is not None:
                    updated_notes.append(note)
                    edited_nids.append(note.id)
                if DEBUG:
                    print("note_was_edited", note_was_edited)
                    print("editedNids", edited_nids)
                
                return note_was_edited
        
        # Start all tasks
        for i, note in enumerate(notes):
            if mw.progress.want_cancel():
                break
            task = asyncio.create_task(process_note(note, i))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    
    col.update_notes(updated_notes)
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


def selected_notes_op(
    done_text: str,
    bulk_op,
    nids: Sequence[NoteId], 
    parent: Browser,
    on_success: Optional[Callable] = None,
    ):
    edited_nids: list[NoteId] = []
    
    # Create a wrapper function that handles the async operation
    def run_bulk_op(col):
        async def async_wrapper():
            return await bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                edited_nids=edited_nids,
            )
        
        # Create and run the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_wrapper())
        finally:
            loop.close()
    
    return (
        CollectionOp(
            parent=parent,
            op=run_bulk_op,
        )
        .success(lambda out: on_bulk_success(out, done_text, edited_nids, nids, parent, on_success))
        .run_in_background()
    )
