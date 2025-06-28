import json
import asyncio
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Coroutine, Any, Union
from functools import partial

from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence

from ..make_notes_tsv import make_tsv_from_notes, import_tsv_file

DEBUG = True


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


def make_inner_bulk_op(
    config: dict,
    op: Callable[..., bool],
    rate_limit: int,
    get_total_tasks: Callable[[], int],
    increment_done_tasks: Callable[..., None],
    increment_in_progress_tasks: Callable[..., None],
    get_progress: Callable[..., tuple[str,int]],
    handle_op_error: Callable[[Exception], None],
    handle_op_result: Callable[[bool], None],
    ) -> Callable[..., Coroutine[Any, Any, bool]]:
    """
    Creates an asynchronous operation processor for bulk operations with rate limiting and progress
    tracking.
    Args:
        config (dict): Addon config
        op (Callable[[dict, ...], bool]): The operation function to execute for each item. It 
            accepts the config dictionary as the first argument, followed by any additional arguments.
        rate_limit (int): The maximum number of operations to perform per minute.
        get_total_tasks (Callable[[], int]): Callback to retrieve the total number of tasks to process.
        increment_done_tasks (Callable[..., None]): Callback to increment the count of completed tasks.
        increment_in_progress_tasks (Callable[..., None]): Callback to increment the count of tasks
            currently in progress.
        get_progress (Callable[..., str]): Callback to retrieve the current progress message.
        handle_op_error (Callable[[Exception], None]): Callback to handle exceptions raised during
            operation execution.
        handle_op_result (Callable[[bool], None]): Callback to handle the result of each operation.
    Returns:
        Callable[[int, ...], None]: An asynchronous function that processes a single 
            operation, given its index and additional arguments.
    """
    # Async approach with rate limiting
    semaphore = asyncio.Semaphore(rate_limit)
    start_time = time.time()
    start_time = time.time()
    
    # Wrapper function to process a single note
    async def process_op(
            task_index: int,
            notes_to_add_dict: dict[str, list[Note]],
            **op_args
        ) -> bool:
        """ Process a single operation with rate limiting and progress tracking.
        Args:          
            task_index (int): The index of the task being processed.
            **op_args: Additional keyword arguments to pass to the operation function.
        Returns:
            bool: The result of the operation, True if successful, False otherwise.
        """
        
        # Calculate time between operations to maintain rate limit
        seconds_per_op = 60.0 / rate_limit
        target_time = start_time + (task_index * seconds_per_op)
        current_time = time.time()
        
        # If we're ahead of schedule, wait until it's time to process the next note
        if current_time < target_time:
            await asyncio.sleep(target_time - current_time)
        
        # Acquire semaphore to limit concurrent operations
        async with semaphore:
            # Use ThreadPoolExecutor for CPU-bound operations
            with ThreadPoolExecutor(max_workers=1) as executor:
                def execute_op():
                    increment_in_progress_tasks()
                    try:
                        return op(
                                config,
                                notes_to_add_dict=notes_to_add_dict,
                                **op_args
                            )
                    except Exception as e:
                        handle_op_error(e)
                        return False
                    finally:
                        increment_done_tasks()
                
                # Run the operation in a thread
                op_result = await asyncio.get_event_loop().run_in_executor(
                    executor, execute_op
                )
            
            elapsed_s = time.time() - start_time
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
            # estimate time remaining from tasks_done and elapsed_time
            time_msg = f"<br>Time: {elapsed_time}"
            progress_msg, tasks_done = get_progress()
            if tasks_done > 3: 
                eta_s = (get_total_tasks() - tasks_done) * (elapsed_s / tasks_done)
                eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                avg_per_op_s = elapsed_s / tasks_done
                time_msg += f"""<br>ETA: {eta_time}
                <br>avg time per task: {avg_per_op_s:.2f}s"""
            mw.taskman.run_on_main(
                lambda: mw.progress.update(
                    label=f"{progress_msg}{time_msg}",
                    value=tasks_done,
                    max=get_total_tasks(),
                )
            )
            
            # Handle results
            handle_op_result(op_result)
            return op_result
    return process_op

async def bulk_nested_notes_op(
        message: str,
        config: dict,
        bulk_inner_op: Callable[..., None],
        col: Collection,
        notes: Sequence[Note],
        edited_nids: list[NoteId],
        notes_to_add_dict: dict[str, list[Note]],
        model: str = "",
    ):
    """
    Perform a bulk operation on a sequence of notes, with multiple nested async operations occurring
    per note instead of just one. Otherwise similar to `bulk_notes_op` except this cannot be
    performed synchronously and thus, requires rate limits to be set in the config.
    
    Args:
        message: A message to display in the progress dialog.
        config: Addon config dict.
        bulk_inner_op: The nested operation function to apply to each note. This op itself will handle
           calling inner_bulk_op and updating updated_notes and edited_nids.
        col: The Anki collection object.
        notes: A sequence of Note objects to process.
        edited_nids: A list to store the IDs of edited notes, to be mutated in place.
        model: The AI model to use for the operation, to get rate limit from config.
    """
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    if not model:
        print("Model arg missing in bulk_nested_notes_op, aborting")
        return None
    config["rate_limits"] = config.get("rate_limits", {})
    rate_limit = config["rate_limits"].get(model, None)
    updated_notes_dict: dict[NoteId, Note] = {}
    
    
    if not rate_limit:
        print("No rate limit set for model, can't run nested async op")
        return col.merge_undo_entries(pos)
    else:
        tasks: list[asyncio.Task] = []
        tasks_in_progress: int = 0
        tasks_done: int = 0
        notes_done: int = 0
        
        
        def increment_done_tasks():
            nonlocal tasks_done, tasks_in_progress
            tasks_done += 1
            tasks_in_progress -= 1
        def increment_in_progress_tasks():
            nonlocal tasks_in_progress
            tasks_in_progress += 1
        def increment_done_notes():
            nonlocal notes_done
            if DEBUG:
                print(f"increment_done_notes called, tasks_done: {tasks_done}, notes_done: {notes_done}")
            notes_done += 1
        def get_progress() -> tuple[str, int]:
            """Get the current progress message and the number of tasks done."""
            nonlocal tasks_done, tasks_in_progress, notes_done
            tasks_per_note = round(tasks_done / notes_done, 1) if notes_done > 0 else 0
            return f"""<strong>Processing:</strong>
                <br>{tasks_done}/{len(tasks)} tasks 
                <br>{notes_done}/{len(notes)} notes
                <br><small>avg tasks per note: {tasks_per_note}</small>
                <br><small>Waiting response: {tasks_in_progress}</small>
                """, tasks_done
                
        
        # Start all tasks
        for note in notes:
            if mw.progress.want_cancel():
                break
            bulk_inner_op(
                config,
                note,
                tasks,
                edited_nids=edited_nids,
                notes_to_add_dict=notes_to_add_dict,
                updated_notes_dict=updated_notes_dict,
                increment_done_tasks=increment_done_tasks,
                increment_in_progress_tasks=increment_in_progress_tasks,
                increment_done_notes=increment_done_notes,
                get_progress=get_progress,
            )
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
    updated_notes = list(updated_notes_dict.values())
    
    return updated_notes, pos, notes_to_add_dict
        
        
def sync_bulk_notes_op(
        pos: int,
        col: Collection,
        config: dict,
        op: Callable[..., bool],
        notes: Sequence[Note],
        edited_nids: list[NoteId],
        notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    ):
    """
    Perform a simple sync bulk operation on a sequence of notes. Will run the operation
    function on each note, updating the progress dialog and collecting edited note IDs.
    
    Used as a fallback for when the async version is not needed or rate limits are not set.
    
    Args:
        pos: The position in the undo stack to add the operation.
        col: The Anki collection object.
        config: Addon config dict.
        op: The operation function to apply to each note.
        col: The Anki collection object.
        notes: A sequence of Note objects to process.
        edited_nids: A list to store the IDs of edited notes, to be mutated in place.
        model: The AI model to use for the operation, to get rate limit from config.
    """
    total_notes = len(notes)
    note_cnt = 0
    updated_notes: list[Note] = []
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
    
    return updated_notes, pos, notes_to_add_dict

async def bulk_notes_op(
        message,
        config,
        op,
        col: Collection,
        notes: Sequence[Note],
        edited_nids: list[NoteId],
        notes_to_add_dict: dict[str, list[Note]] = {},
        model: str = "",
    ):
    """
    Perform a simple async or sync bulk operation on a sequence of notes. Will run the operation
    function on each note, updating the progress dialog and collecting edited note IDs.
    Each note will create one async task.
    
    The bulk op will be sync if rate_limit is None or 0, otherwise it will be async.
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
    config["rate_limits"] = config.get("rate_limits", {})
    if not model:
        print("Model arg missing in bulk_notes_op, aborting")
        return None
    rate_limit = config["rate_limits"].get(model, None)

    if not rate_limit:
        return sync_bulk_notes_op(
            pos=pos,
            col=col,
            config=config,
            op=op,
            notes=notes,
            edited_nids=edited_nids,
            notes_to_add_dict=notes_to_add_dict,
        )
    
    updated_notes: list[Note] = []
    tasks: list[asyncio.Task] = []
    tasks_in_progress: int = 0
    tasks_done: int = 0
    
    def increment_done_tasks():
        nonlocal tasks_done, tasks_in_progress
        tasks_done += 1
        tasks_in_progress -= 1
    def increment_in_progress_tasks():
        nonlocal tasks_in_progress
        tasks_in_progress += 1
    def get_progress():
        nonlocal tasks_done, tasks_in_progress
        return f"""<strong>Processed:</strong> {tasks_done}/{len(tasks)} notes
            <br><small>Waiting response: {tasks_in_progress}</small>""", tasks_done
    def handle_op_success(
            note: Note,
            was_success: bool,
        ):
        """Handle successful operation result."""
        if was_success and edited_nids is not None:
            updated_notes.append(note)
            edited_nids.append(note.id)
        if DEBUG:
            print("note_was_edited", was_success)
            print("editedNids", edited_nids)
    
    # Start all tasks
    for i, note in enumerate(notes):
        # Create partial functions that bind the current note value or otherwise we end up
        # adding the same note to
        handle_op_error = partial(
            lambda current_note, e: print(f"Error during operation with note {current_note.id}: {e}"), 
            note
        )

        handle_op_result = partial(
            lambda current_note, was_success: 
            handle_op_success(current_note, was_success),
            note
        )
        process_note = make_inner_bulk_op(
            config=config,
            op=op,
            rate_limit=rate_limit,
            get_total_tasks=lambda: len(tasks),
            increment_done_tasks=increment_done_tasks,
            increment_in_progress_tasks=increment_in_progress_tasks,
            get_progress=get_progress,
            handle_op_error=handle_op_error,
            handle_op_result=handle_op_result,
        )
        if mw.progress.want_cancel():
            break
        task: asyncio.Task = asyncio.create_task(process_note(
            # task_index for process_op in make_inner_bulk_op
            task_index=i,
            notes_to_add_dict=notes_to_add_dict,
            # note is passed to the op function, along with config in make_inner_bulk_op
            note=note
        ))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    return updated_notes, pos, notes_to_add_dict

def on_bulk_success(
    out,
    done_text: str,
    edited_nids: Sequence[NoteId],
    nids: Sequence[NoteId],
    parent: Browser,
    notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    extra_callback=None,
):
    tooltip(
        f"{done_text} in {len(edited_nids)}/{len(nids)} selected notes.",
        parent=parent,
        period=5000,
    )
    if DEBUG:
        print("on_bulk_success", out, notes_to_add_dict)
    if extra_callback:
        extra_callback()
    if notes_to_add_dict:
        new_notes: list[Note] = []
        for note_list in notes_to_add_dict.values():
            new_notes.extend(note_list)
        if new_notes:
            new_notes_tsv_str = make_tsv_from_notes(
                notes=new_notes,
                config=mw.addonManager.getConfig(__name__) or {},
            )
            if new_notes_tsv_str:
                # Write the TSV to the media folder
                import_tsv_file(
                    "new_notes.tsv",
                    new_notes_tsv_str,
            )


def selected_notes_op(
    done_text: str,
    bulk_op,
    nids: Sequence[NoteId],
    parent: Browser,
    on_success: Optional[Callable] = None,
    ):
    edited_nids: list[NoteId] = []
    notes_to_add_dict: dict[str, list[Note]] = {}
    
    # Create a wrapper function that handles the async operation
    def run_bulk_op(col: Collection):
        async def async_wrapper():
            result = await bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                edited_nids=edited_nids,
                notes_to_add_dict=notes_to_add_dict,
            )
            updated_notes, pos, res_notes_to_add_dict = result
            if DEBUG:
                print("selected_notes_op done", notes_to_add_dict, res_notes_to_add_dict)
            notes_to_add_dict.update(res_notes_to_add_dict)
            
            if DEBUG:
                print("selected_notes_op done", [ u['sentence-vocab-list'] for u in  updated_notes])
            mw.col.update_notes(updated_notes)
            return mw.col.merge_undo_entries(pos)
            
        
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
        .success(lambda out: on_bulk_success(
            out,
            done_text,
            edited_nids,
            nids,
            parent,
            notes_to_add_dict,
            on_success
            ))
        .run_in_background()
    )
