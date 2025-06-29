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


MAX_TOKENS_VALUE = 2000

class CancelState:
    """Shared state for cancellation that can be accessed across threads."""
    def __init__(self):
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def is_cancelled(self):
        return self._cancelled

def get_response(model, prompt, cancel_state: Optional[CancelState] = None):
    """Get a response from the appropriate model based on the configuration.

    Args:
        model: The model to use for the request.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if model.startswith("gemini"):
        return get_response_from_gemini(model, prompt, cancel_state=cancel_state)
    elif model.startswith("gpt") or model.startswith("o3") or model.startswith("o1"):
        return get_response_from_openai(model, prompt, cancel_state=cancel_state)
    else:
        print(f"Unsupported model: {model}")
        return None



def get_response_from_gemini(model, prompt, cancel_state: Optional[CancelState] = None):
    """Get a response from Google's Gemini API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if DEBUG:
        print(f"Gemini call, model: {model}")
        
    if cancel_state and cancel_state.is_cancelled():
        return None

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
    request_timeout = config.get("request_timeout", 30)
    # Make the API call
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_api_key}",
            headers=headers,
            json=data,
            timeout=request_timeout,
        )
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Error making request: {e}")
        return None

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


def get_response_from_openai(model, prompt, cancel_state: Optional[CancelState] = None):
    if DEBUG:
        print("OpenAI call, model", model)
        
    if cancel_state and cancel_state.is_cancelled():
        return None

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
    request_timeout = config.get("request_timeout", 30)
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
        data["max_completion_tokens"] = MAX_TOKENS_VALUE
    else:
        data["max_tokens"] = MAX_TOKENS_VALUE

    # Make the API call
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=request_timeout,
        )
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Error making request: {e}")
        return None

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

class CancelManager:
    """
    A class to manage cancellation of asynchronous operations.
    It provides a way to set and check cancellation requests.
    """
    def __init__(self, tasks, cancel_state: Optional[CancelState] = None):
        self.cancel_requested = False
        self.tasks = tasks  # List of tasks to cancel if needed
        self.cancel_state = cancel_state or CancelState()
        self.monitor_task = asyncio.create_task(self.monitor_for_cancellation())
        

    def request_cancel(self):
        """Request cancellation of all tasks managed by this instance."""
        self.cancel_requested = True
        if DEBUG:
            print("Cancellation requested, updating UI")
        
        # Set the shared cancel state
        self.cancel_state.cancel()
        
        # Update UI to show cancellation is in progress
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label="<b>Cancelling operations...</b><br>Please wait while tasks are cleaned up.",
                value=0,
                max=0  # Indeterminate progress
            )
        )
        
        # Cancel all tasks without waiting
        for task in self.tasks:
            if not task.done():
                task.cancel()
        self.monitor_task.cancel()

    def is_cancel_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self.cancel_requested
    
    
    def check_for_cancellation(self):
        if DEBUG:
            print("Checking for cancellation")
        if mw.progress.want_cancel() and not self.cancel_requested:
            if DEBUG:
                print("Cancellation requested, setting cancel_requested to True")   
            self.cancel_requested = True
            # Cancel all running tasks
            for t in self.tasks:
                t.cancel()
            return True
        return False
    
    async def monitor_for_cancellation(self):
        """Monitor for cancellation requests and cancel all tasks if requested."""
        try:
            while not self.cancel_requested:
                # Check for cancellation request from Anki
                if mw.progress.want_cancel():
                    if DEBUG:
                        print("Cancellation requested, setting cancel_requested to True")
                    self.request_cancel()
                    break
                
                # Check if all tasks are completed naturally
                if all(task.done() for task in self.tasks):
                    if DEBUG:
                        print("All tasks completed naturally, exiting monitor")
                    break
                    
                # Check frequently but don't hog the CPU
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            if DEBUG:
                print("Cancellation monitor task cancelled")
            # Just exit the task when cancelled

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
    cancel_state: Optional[CancelState] = None,
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
    
    cancel_state = cancel_state or CancelState()
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
        
        # If we're ahead of schedule, wait until it's time to process the next op
        if current_time < target_time:
            try:
                await asyncio.sleep(target_time - current_time)
            except asyncio.CancelledError:
                return False
        try:
            # Acquire semaphore to limit concurrent operations
            async with semaphore:
                # Check for cancel request before starting the operation
                if mw.progress.want_cancel():
                    return False
                
                # Use ThreadPoolExecutor for CPU-bound operations
                with ThreadPoolExecutor(max_workers=1) as executor:
                    def execute_op():
                        increment_in_progress_tasks()
                        try:
                            if mw.progress.want_cancel() or cancel_state.is_cancelled():
                                if DEBUG:
                                    print("Inner process op: cancellation requested")
                                return False
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
                            
                    # Check for cancellation again before running
                    if mw.progress.want_cancel():
                        return False
                    
                    # Run the operation in a thread with cancellation checking
                    try:
                        op_result = await asyncio.get_event_loop().run_in_executor(
                            executor, execute_op
                        )
                    except asyncio.CancelledError:
                        # If the operation was cancelled, return False
                        return False
                
                elapsed_s = time.time() - start_time
                elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
                # estimate time remaining from tasks_done and elapsed_time
                time_msg = f"<br><code>Time: {elapsed_time}</code>"
                progress_msg, tasks_done = get_progress()
                if tasks_done > 3: 
                    eta_s = (get_total_tasks() - tasks_done) * (elapsed_s / tasks_done)
                    eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                    avg_per_op_s = elapsed_s / tasks_done
                    time_msg += f"""| <small> Avg time per task: {avg_per_op_s:.2f}s</small>
                    <br><code>ETA: {eta_time}</code>"""
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
                
        except asyncio.CancelledError:
            return False

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
                <br><strong><code>{tasks_done}/{len(tasks)}</code></strong> tasks <small style="opacity: 0.85"> | Waiting response: {tasks_in_progress}</small>
                <br><strong><code>{notes_done}/{len(notes)}</code></strong> notes <small style="opacity: 0.85"> | Avg tasks per note: {tasks_per_note}</small>
                """, tasks_done
        
        cancel_state = CancelState()
        
        # Gather all tasks from all inner ops
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
                cancel_state=cancel_state,
            )
        cancel_manager = CancelManager(tasks, cancel_state=cancel_state)
        
        try:
            # First await for the regular tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Then cancel the monitor task if it's still running
            if not cancel_manager.monitor_task.done():
                cancel_manager.monitor_task.cancel()
                
            # Wait for it to finish cancellation
            try:
                await asyncio.wait_for(cancel_manager.monitor_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        except asyncio.CancelledError:
            pass
        finally:
            if not cancel_manager.monitor_task.done():
                cancel_manager.monitor_task.cancel()
        
        if cancel_manager.is_cancel_requested():
            updated_notes = list(updated_notes_dict.values())
            return updated_notes, pos, notes_to_add_dict
    
        
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
            print("Sync bulk notes op: Error processing note", note.id, e)
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
        print("Sync bulk notes op finished. editedNids", edited_nids)
    
    mw.taskman.run_on_main(lambda: mw.progress.finish())
            
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
        if DEBUG:
            print(f"No rate limit set for model {model}, running sync op")
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
        return f"""<strong>Processed:<code> {tasks_done}/{len(tasks)}</code></strong> notes <smallstyle="opacity: 0.85"> | Waiting response: {tasks_in_progress}</small>""", tasks_done
    def handle_op_success(
            note: Note,
            was_success: bool,
        ):
        """Handle successful operation result."""
        if was_success and edited_nids is not None:
            updated_notes.append(note)
            edited_nids.append(note.id)
        if DEBUG:
            print(f"Bulk notes op success for note {note.id}, was_success: {was_success}, tasks_done: {tasks_done}, tasks_in_progress: {tasks_in_progress}, actual tasks done: {len([t for t in tasks if t.done()])}")
    
    cancel_state = CancelState()
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
            cancel_state=cancel_state,
        )
        if mw.progress.want_cancel():
            if DEBUG:
                print("Bulk notes op cancelled before starting tasks")
            break
        task: asyncio.Task = asyncio.create_task(process_note(
            # task_index for process_op in make_inner_bulk_op
            task_index=i,
            notes_to_add_dict=notes_to_add_dict,
            # note is passed to the op function, along with config in make_inner_bulk_op
            note=note
        ))
        tasks.append(task)
    if DEBUG:
        print(f"Async bulk notes op started with {len(tasks)} tasks, rate limit: {rate_limit}")
    
    cancel_manager = CancelManager(tasks, cancel_state)

    # Wait for all tasks to complete
    try:
        if DEBUG:
            print("Bulk notes op awaiting tasks")
            
        # First, wait only for the operation tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # After all operation tasks are done, explicitly cancel the monitor task
        if not cancel_manager.monitor_task.done():
            cancel_manager.monitor_task.cancel()
            
        # Wait for the monitor task to finish cancellation
        try:
            await asyncio.wait_for(cancel_manager.monitor_task, timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
            
        if DEBUG:
            print("Bulk notes op completed successfully, all tasks finished")
    except asyncio.CancelledError:
        if DEBUG:
            print("Bulk notes op asyncio.CancelledError caught")
        cancel_manager.request_cancel()
    finally:
        if DEBUG:
            print("Bulk notes op finally block reached, cleaning up tasks")
        if not cancel_manager.monitor_task.done():
            cancel_manager.monitor_task.cancel()

    # Check if cancellation was requested and handle accordingly
    if cancel_manager.is_cancel_requested():
        if DEBUG:
            print("Bulk notes op cancellation requested, returning early")
        if not cancel_manager.monitor_task.done():
            cancel_manager.monitor_task.cancel()
        return updated_notes, pos, notes_to_add_dict
    
    if DEBUG:
        print("Bulk notes op completed successfully, updating notes")
    
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
    mw.taskman.run_on_main(lambda: mw.progress.finish())
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
