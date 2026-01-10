import json
import asyncio
import logging
import time
import requests  # type: ignore
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Coroutine, Any, Union
from functools import partial

from anki.notes import Note, NoteId
from anki.collection import Collection, OpChanges
from aqt import mw
from aqt.browser import Browser
from aqt.operations import CollectionOp
from aqt.utils import tooltip
from collections.abc import Sequence

from ..utils import get_field_config, print_error_traceback

from ..make_notes_tsv import make_tsv_from_notes, import_tsv_file

logger = logging.getLogger(__name__)

MAX_TOKENS_VALUE = 8000


class CancelState:
    """Shared state for cancellation that can be accessed across threads."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


def get_response(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from the appropriate model based on the configuration.

    Args:
        model: The model to use for the request.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    if model.startswith("gemini"):
        return get_response_from_gemini(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            json_result_corrector=json_result_corrector,
        )
    elif model.startswith("gpt") or model.startswith("o3") or model.startswith("o1"):
        return get_response_from_openai(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            json_result_corrector=json_result_corrector,
        )
    elif model.startswith("claude") or model.startswith("anthropic"):
        return get_response_from_anthropic(
            model,
            prompt,
            cancel_state=cancel_state,
            instructions=instructions,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            json_result_corrector=json_result_corrector,
        )
    else:
        logger.error(f"Unsupported model: {model}")
        return None


class CancellableRequest:
    def __init__(self):
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def post(self, url, **kwargs):
        def _request():
            return self.session.post(url, **kwargs)

        self.future = self.executor.submit(_request)
        try:
            return self.future.result()  # Blocks like normal requests
        finally:
            # Clean up the executor if done
            if self.future.done():
                self.executor.shutdown(wait=False)

    def cancel(self):
        if self.future and not self.future.done():
            # Cancel the future - this will interrupt the thread if it's not already sending data
            cancelled = self.future.cancel()
            logger.debug(f"Request future cancelled: {cancelled}")

        # Close the underlying session anyway
        self.session.close()
        self.executor.shutdown(wait=False)  # Don't wait for tasks to complete


# Global variable to track active requests
active_requests: list[CancellableRequest] = []


def decode_json_result(json_str: str):
    logging.debug("json_result", json_str)
    try:
        result = json.loads(json_str)
        logger.debug(
            "Parsed result from json: %s", json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response, json_result: {json_str}")
        return None
    except ValueError as ve:
        logger.error(f"Failed to parse JSON response - ValueError: {ve}")
        if "integer string conversion" in str(ve):
            logger.error("Large integer detected in JSON response")
        logger.error(f"json_result: {json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        logger.error(f"json_result: {json_str}")
        return None


def clean_response_schema_for_gemini(schema: dict) -> dict:
    """Cleans the response schema to be compatible with Gemini's expected format.

    Args:
        response_schema: The original response schema.

    Returns:
        A cleaned response schema compatible with Gemini.
    """
    # Clean "additionalProperties" from array items and objects to avoid Gemini rejecting the schema
    if isinstance(schema, dict):
        if schema.get("type") == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                if "additionalProperties" in items:
                    del items["additionalProperties"]
                # Recursively clean items
                schema["items"] = clean_response_schema_for_gemini(items)
        elif schema.get("type") == "object" and "properties" in schema:
            if "additionalProperties" in schema:
                del schema["additionalProperties"]
            for key, value in schema["properties"].items():
                schema["properties"][key] = clean_response_schema_for_gemini(value)
    return schema


def get_response_from_gemini(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from Google's Gemini API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    logger.debug(f"Gemini call, model: {model}")

    if cancel_state and cancel_state.is_cancelled():
        return None

    # Create the request body
    data: dict[str, Any] = {
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
                        instructions
                        if instructions
                        else (
                            "You are a helpful assistant for processing Japanese text. You are a"
                            " superlative expert in the Japanese language and its writing system."
                            " You are designed to output JSON."
                        )
                    )
                },
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            # maxOutputTokens includes both thinking and output so it needs to be large enough
            "maxOutputTokens": max_output_tokens or MAX_TOKENS_VALUE,
            "thinkingConfig": {"thinkingBudget": 6000},
        },
    }
    if max_output_tokens is not None:
        logger.debug("Using max_output_tokens %d", max_output_tokens)
    if response_schema:
        response_schema = clean_response_schema_for_gemini(response_schema)
        data["generationConfig"]["responseSchema"] = response_schema
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    headers = {
        "Content-Type": "application/json",
        # "x-goog-api-key": google_api_key,
    }

    config = mw.addonManager.getConfig(__name__)
    if config is None:
        print("No configuration found for the addon.")
        return None
    google_api_key = config.get("google_api_key", "")
    request_timeout = config.get("request_timeout", 300)

    # Make the API call
    req = CancellableRequest()
    active_requests.append(req)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={google_api_key}"
    )
    try:
        response = req.post(
            url,
            headers=headers,
            json=data,
            timeout=request_timeout,
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return None
    except Exception as e:
        logger.error(f"Error making request: {e}")
        return None
    finally:
        # Response completed, remove from active requests
        if req in active_requests:
            active_requests.remove(req)

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        # Extract content from Gemini response structure
        content_text = decoded_json["candidates"][0]["content"]["parts"][0]["text"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None

    # Extract the JSON from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def get_response_from_openai(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    logger.debug("OpenAI call, model %s", model)

    if cancel_state and cancel_state.is_cancelled():
        return None

    # Use max_completion_tokens instead of max_tokens for o3

    messages = [
        {
            "role": "system",
            "content": (
                instructions
                if instructions
                else (
                    "You are a helpful assistant for processing Japanese text. You are a"
                    " superlative expert in the Japanese language and its writing system. You are"
                    " designed to output JSON."
                )
            ),
        },
        {"role": "user", "content": prompt},
    ]
    config = mw.addonManager.getConfig(__name__)
    if config is None:
        logger.error("No configuration found for the addon.")
        return None
    openai_api_key = config.get("openai_api_key", "")
    request_timeout = config.get("request_timeout", 300)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    data: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if any(model.startswith(m) for m in ["o3", "o1", "gpt-5"]):
        # This is total completion tokens, including both reasoning and output
        data["max_completion_tokens"] = max_output_tokens or MAX_TOKENS_VALUE
    else:
        # This is for GPT models and only limits output, not reasoning
        data["max_tokens"] = max_output_tokens or MAX_TOKENS_VALUE
    if response_schema:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "schema": response_schema,
                "strict": True,
            },
        }
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    # Make the API call
    req = CancellableRequest()
    active_requests.append(req)
    try:
        response = req.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=request_timeout,
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return None
    except Exception as e:
        logger.error(f"Error making request: {e}")
        return None

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_text = decoded_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None
    finally:
        # Request completed, remove from active requests
        if req in active_requests:
            active_requests.remove(req)

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


def get_response_from_anthropic(
    model: str,
    prompt: str,
    cancel_state: Optional[CancelState] = None,
    instructions: Optional[str] = None,
    response_schema: Optional[dict] = None,
    max_output_tokens: Optional[int] = None,
    json_result_corrector: Optional[Callable[[str], str]] = None,
) -> Union[dict, None]:
    """Get a response from Anthropic's Claude API.

    Args:
        prompt: The prompt to send to the API.

    Returns:
        A dict containing the parsed JSON response, or None if there was an error.
    """
    logger.debug("Anthropic call, model %s", model)

    if cancel_state and cancel_state.is_cancelled():
        return None

    messages = [
        {"role": "user", "content": prompt},
    ]
    # Create the request body
    data: dict[str, Any] = {
        "model": model,
        "system": (
            instructions
            if instructions
            else (
                "You are a helpful assistant for processing Japanese text. You are a"
                " superlative expert in the Japanese language and its writing system. You are"
                " designed to output JSON."
            )
        ),
        "max_tokens": max_output_tokens or MAX_TOKENS_VALUE,
        "messages": messages,
    }

    if response_schema:
        data["output_format"] = {
            "type": "json_schema",
            "schema": response_schema,
        }
        logger.debug(
            "Using response schema %s", json.dumps(response_schema, ensure_ascii=False, indent=2)
        )

    config = mw.addonManager.getConfig(__name__)
    if config is None:
        logger.error("No configuration found for the addon.")
        return None
    anthropic_api_key = config.get("anthropic_api_key", "")
    request_timeout = config.get("request_timeout", 300)

    headers = {
        "x-api-key": anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Make the API call
    req = CancellableRequest()
    active_requests.append(req)
    url = "https://api.anthropic.com/v1/messages"
    try:
        response = req.post(
            url,
            headers=headers,
            json=data,
            timeout=request_timeout,
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return None
    except Exception as e:
        logger.error(f"Error making request: {e}")
        return None
    finally:
        # Response completed, remove from active requests
        if req in active_requests:
            active_requests.remove(req)

    if response.status_code != 200:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return None

    try:
        decoded_json = json.loads(response.text)
        content_text = decoded_json["content"][0]["text"]
    except json.JSONDecodeError as je:
        logger.error(f"Error decoding JSON: {je}")
        logger.error("response %s", response.text)
        return None
    except KeyError as ke:
        logger.error(f"Error extracting content: {ke}")
        logger.error("response %s", response.text)
        return None
    finally:
        # Request completed, remove from active requests
        if req in active_requests:
            active_requests.remove(req)

    # Extract the cleaned meaning from the response
    json_result = extract_json_string(content_text)

    result = decode_json_result(json_result)
    if not result and json_result_corrector:
        json_result = json_result_corrector(json_result)
        result = decode_json_result(json_result)
    return result


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
        logger.debug("Cancellation requested, updating UI")

        # Immediately end all in-flight requests
        for req in list(active_requests):
            logger.debug("Cancelling active request %s", req)
            req.cancel()

        # Set the shared cancel state
        self.cancel_state.cancel()

        # Update UI to show cancellation is in progress
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label="<b>Cancelling operations...</b><br>Please wait while tasks are cleaned up.",
                value=0,
                max=0,  # Indeterminate progress
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
        logger.debug("Checking for cancellation")
        if mw.progress.want_cancel() and not self.cancel_requested:
            logger.debug("Cancellation requested, setting cancel_requested to True")
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
                    logger.debug("Cancellation requested, setting cancel_requested to True")
                    self.request_cancel()
                    break

                # Check if all tasks are completed naturally
                if all(task.done() for task in self.tasks):
                    logger.debug("All tasks completed naturally, exiting monitor")
                    break

                # Check frequently but don't hog the CPU
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("Cancellation monitor task cancelled")
            # Just exit the task when cancelled


class AsyncTaskProgressUpdater:
    """A class to update the progress dialog in async ops."""

    def __init__(
        self, total_notes: Optional[int] = None, total_tasks: int = 0, title: Optional[str] = None
    ):
        self.total_tasks = total_tasks
        self.tasks_done = 0
        self.tasks_in_progress = 0
        self.notes_done = 0
        self.total_notes = total_notes
        # Sum of each task's execution time, for estimating average time per task
        self.cumulative_task_time = 0.0
        self.max_task_time = 0.0
        self.start_time = time.time()
        if title is None:
            title = "Processing asynchronous tasks..."
        self.set_title(title)

        # Periodic updater (started when an event loop is running)
        self.autoupdate_task: Optional[asyncio.Task] = None
        self._autoupdate_started = False
        self._autoupdate_deferred = False
        self.start_autoupdate()

    def start_autoupdate(self):
        """Start periodic progress updates if an event loop is running."""
        if self.autoupdate_task and not self.autoupdate_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet; allow caller to retry later from async context
            self._autoupdate_deferred = True
            return
        self.autoupdate_task = loop.create_task(self._periodic_update_progress())
        self._autoupdate_started = True
        self._autoupdate_deferred = False

    async def _periodic_update_progress(self):
        while True:
            self.update_progress()
            await asyncio.sleep(1)

    def stop_autoupdate(self):
        """Stop the periodic progress update task."""
        if self.autoupdate_task:
            self.autoupdate_task.cancel()
            self.autoupdate_task = None

    def set_total_notes(self, total_notes: int):
        self.total_notes = total_notes

    def set_total_tasks(self, total_tasks: int):
        self.total_tasks = total_tasks

    def set_title(self, title: str):
        mw.taskman.run_on_main(lambda: mw.progress.set_title(title))

    def increment_counts(
        self,
        total_tasks=0,
        tasks_done: int = 0,
        tasks_in_progress: int = 0,
        notes_done: int = 0,
        cumulative_task_time: float = 0.0,
    ):
        """Increment the counts of tasks and notes."""
        self.total_tasks += total_tasks
        self.tasks_done += tasks_done
        self.tasks_in_progress += tasks_in_progress
        self.notes_done += notes_done
        self.cumulative_task_time += cumulative_task_time
        if cumulative_task_time > self.max_task_time:
            self.max_task_time = cumulative_task_time

    def update_progress(self):
        """Update the Step 1 progress dialog with the current task and note counts."""
        task_progress_msg = f"""<strong>Processing:</strong>
            <br><strong><code>{self.tasks_done}/{self.total_tasks}</code></strong>
            tasks <small style="opacity: 0.85"> | Waiting response: {self.tasks_in_progress}</small>
            """
        if self.total_notes is not None:
            tasks_per_note = (
                round(self.tasks_done / self.notes_done, 1) if self.notes_done > 0 else 0
            )
            task_progress_msg += (
                f"<br><strong><code>{self.notes_done}/{self.total_notes}</code></strong> notes"
                f' <small style="opacity: 0.85"> | Avg tasks per note: {tasks_per_note}</small>'
            )

        elapsed_s = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        # estimate time remaining from tasks_done and elapsed_time
        time_msg = f"<br><code>Time: {elapsed_time}</code>"
        if self.tasks_done > 3:
            eta_s = (self.total_tasks - self.tasks_done) * (elapsed_s / self.tasks_done)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            avg_per_op_s = self.cumulative_task_time / self.tasks_done
            time_msg += f""" | <small> Avg time per task: {avg_per_op_s:.2f}s
            | Max: {self.max_task_time:.2f}s</small>
            <br><code>ETA: {eta_time}</code>"""
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label=f"{task_progress_msg}{time_msg}",
                value=self.tasks_done,
                max=self.total_tasks,
            )
        )

    def update_note_adding_progress(
        self,
        notes_added: int = 0,
        total_notes: int = 0,
        failed: int = 0,
    ):
        """
        Update the Step 2 progress dialog for note adding operations occuring after async tasks
        are done.
        """
        try:
            elapsed_s = time.time() - self.start_time
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
            time_msg = f"<br><code>Time: {elapsed_time}</code>"
            if notes_added > 0:
                eta_s = (notes_added - self.notes_done) * (elapsed_s / notes_added)
                eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                avg_per_note_s = elapsed_s / notes_added
                time_msg += f""" | <small> Avg time per note: {avg_per_note_s:.2f}s</small>
                <br><code>ETA: {eta_time}</code>"""
            task_progress_msg = f"""<strong>Adding notes:</strong>
                <br><strong><code>{notes_added}/{total_notes}</code></strong> notes"""
            if failed > 0:
                task_progress_msg += f""" | <strong style="color: red;">{failed} failed</strong>"""
        except Exception as e:
            logger.error("Error updating note adding progress: %s", e)
            return
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label=f"{task_progress_msg}{time_msg}",
                value=notes_added,
                max=total_notes,
            )
        )

    def update_new_note_processing_progress(
        self,
        new_notes_processed: int = 0,
        total_notes: int = 0,
    ):
        """Update the Step 3 progress dialog for processing new notes after they have been added."""
        elapsed_s = time.time() - self.start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
        time_msg = f"<br><code>Time: {elapsed_time}</code>"
        if new_notes_processed > 0:
            eta_s = (total_notes - new_notes_processed) * (elapsed_s / new_notes_processed)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            avg_per_note_s = elapsed_s / new_notes_processed
            time_msg += f""" | <small> Avg time per note: {avg_per_note_s:.2f}s</small>
            <br><code>ETA: {eta_time}</code>"""
        task_progress_msg = f"""<strong>Processing new notes:</strong>
            <br><strong><code>{new_notes_processed}/{total_notes}</code></strong> notes"""
        mw.taskman.run_on_main(
            lambda: mw.progress.update(
                label=f"{task_progress_msg}{time_msg}",
                value=new_notes_processed,
                max=total_notes,
            )
        )


def make_inner_bulk_op(
    config: dict,
    op: Callable[..., bool],
    rate_limit: int,
    progress_updater: AsyncTaskProgressUpdater,
    handle_op_error: Callable[[Exception], None],
    handle_op_result: Callable[[bool], None],
    cancel_state: Optional[CancelState] = None,
    one_task_per_op: bool = False,
) -> Callable[..., Coroutine[Any, Any, bool]]:
    """
    Creates an asynchronous operation processor for bulk operations with rate limiting and progress
    tracking.

    :param config (dict): Addon config
    :param op (Callable[[dict, ...], bool]): The operation function to execute for each item. It
            accepts the config dictionary as the first argument, followed by additional arguments.
    :param rate_limit (int): The maximum number of operations to perform per minute.
    :param get_total_tasks (Callable[[], int]): Callback to retrieve total number of tasks to process.
    :param increment_done_tasks (Callable[..., None]): Callback to increment count of completed tasks.
    :param increment_in_progress_tasks (Callable[..., None]): Callback to increment the count of tasks
            currently in progress.
    :param get_progress (Callable[..., str]): Callback to retrieve the current progress message.
    :param handle_op_error (Callable[[Exception], None]): Callback to handle exceptions raised during
            operation execution.
    :param handle_op_result (Callable[[bool], None]): Callback to handle the result of each operation.
    :param cancel_state (Optional[CancelState]): Shared cancellation state.
    :param one_task_per_op (bool): Whether each operation corresponds to a single task for progress
            tracking. If False, the caller is responsible for updating done note counts.

    Returns:
        Callable[[int, ...], None]: An asynchronous function that processes a single
            operation, given its index and additional arguments.
    """
    # Async approach with rate limiting
    semaphore = asyncio.Semaphore(rate_limit)
    start_time = time.time()

    cancel_state = cancel_state or CancelState()

    # Create a thread pool executor for blocking operations (API calls)
    executor = ThreadPoolExecutor(max_workers=rate_limit)

    # Wrapper function to process a single note
    async def process_op(
        task_index: int,
        notes_to_add_dict: dict[str, list[Note]],
        notes_to_update_dict: dict[NoteId, Note],
        **op_args,
    ) -> bool:
        """Process a single operation with rate limiting and progress tracking.
        Args:
            task_index (int): The index of the task being processed.
            notes_to_add_dict (dict[str, list[Note]]): Dictionary of notes to add.
            notes_to_update_dict (dict[NoteId, Note]): Dictionary of notes to update.
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
                logger.debug("Inner bulk operation CancelledError encountered")
                return False
        try:
            # Acquire semaphore to limit concurrent operations
            async with semaphore:
                # Check for cancel request before starting the operation
                if mw.progress.want_cancel():
                    logger.debug("Inner bulk operation mw.progress.want_cancel()")
                    return False

                progress_updater.increment_counts(
                    tasks_in_progress=1,
                )
                task_start_time = time.time()
                progress_updater.update_progress()

                try:
                    if mw.progress.want_cancel() or cancel_state.is_cancelled():
                        logger.debug("Inner process op: cancellation requested")
                        return False

                    # If the op itself is async, run it directly; otherwise run the blocking call
                    # in the thread pool so the event loop is not blocked by HTTP requests.
                    if asyncio.iscoroutinefunction(op):
                        op_result = await op(
                            config,
                            notes_to_add_dict=notes_to_add_dict,
                            notes_to_update_dict=notes_to_update_dict,
                            **op_args,
                        )
                    else:
                        loop = asyncio.get_event_loop()

                        def blocking_op():
                            return op(
                                config,
                                notes_to_add_dict=notes_to_add_dict,
                                notes_to_update_dict=notes_to_update_dict,
                                **op_args,
                            )

                        op_result = await loop.run_in_executor(executor, blocking_op)

                    if asyncio.iscoroutine(op_result):
                        op_result = await op_result

                except Exception as e:
                    logger.error("Inner process op error, passing to handle_op_error: %s", e)
                    handle_op_error(e)
                    return False
                finally:
                    task_time = time.time() - task_start_time
                    progress_updater.increment_counts(
                        tasks_done=1,
                        tasks_in_progress=-1,
                        cumulative_task_time=task_time,
                        notes_done=1 if one_task_per_op else 0,
                    )
                    progress_updater.update_progress()

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
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
    notes_to_update_dict: dict[NoteId, Note],
    model: str = "",
    on_end: Optional[Callable[..., None]] = None,
) -> tuple[int, dict[str, list[Note]], dict[NoteId, Note]]:
    """
    Perform a bulk operation on a sequence of notes, with multiple nested async operations occurring
    per note instead of just one. Otherwise similar to `bulk_notes_op` except this cannot be
    performed synchronously and thus, requires rate limits to be set in the config.


    :param message: A message to display in the progress dialog.
    :param config: Addon config dict.
    :param bulk_inner_op: The nested operation function to apply to each note. This op itself will
           handle calling inner_bulk_op and updating updated_notes and edited_nids.
    :param col: The Anki collection object.
    :param notes: A sequence of Note objects to process.
    :param edited_nids: A list to store the IDs of edited notes, to be mutated in place.
    :param model: The AI model to use for the operation, to get rate limit from config.
    :param on_end: An optional callback to run on completion of the bulk op. Should be running other
           side effects that do not edit or add notes as those should be handled through
    """
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    if not model:
        logger.error("Model arg missing in bulk_nested_notes_op, aborting")
        return pos, notes_to_add_dict, notes_to_update_dict
    config["rate_limits"] = config.get("rate_limits", {})
    rate_limit = config["rate_limits"].get(model, None)

    progress_updater.set_total_notes(len(notes))
    # Can start auto updater now that we're in an async context with a running loop
    progress_updater.start_autoupdate()

    if not rate_limit:
        logger.error("No rate limit set for model, can't run nested async op")
        return pos, notes_to_add_dict, notes_to_update_dict
    else:
        tasks: list[asyncio.Task] = []

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
                notes_to_update_dict=notes_to_update_dict,
                progress_updater=progress_updater,
                cancel_state=cancel_state,
            )
        cancel_manager = CancelManager(tasks, cancel_state=cancel_state)

        try:
            # First await for the regular tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Then cancel the monitor task if it's still running
            if not cancel_manager.monitor_task.done():
                logger.debug("Cancelling monitor task first check")
                cancel_manager.monitor_task.cancel()

            # Wait for it to finish cancellation
            try:
                logger.debug("Waiting for monitor task to finish cancellation")
                await asyncio.wait_for(cancel_manager.monitor_task, timeout=0.5)
                logger.debug("Monitor task finished cancellation")
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.debug("Monitor task CancelledError encountered")
                pass
        except asyncio.CancelledError:
            logger.debug("Cancelling bulk operation")
            pass
        finally:
            if not cancel_manager.monitor_task.done():
                logger.debug("Cancelling monitor task second check")
                cancel_manager.monitor_task.cancel()

        if cancel_manager.is_cancel_requested():
            logger.debug("Bulk operation was cancelled, returning results so far")
            if on_end:
                on_end()
            progress_updater.stop_autoupdate()
            return pos, notes_to_add_dict, notes_to_update_dict

    if on_end:
        on_end()
    progress_updater.stop_autoupdate()
    return pos, notes_to_add_dict, notes_to_update_dict


def sync_bulk_notes_op(
    pos: int,
    col: Collection,
    config: dict,
    op: Callable[..., bool],
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    notes_to_update_dict: Optional[dict[NoteId, Note]] = None,
    on_end: Optional[Callable[..., None]] = None,
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
        on_end: An optional callback to run on completion of the bulk op. Should be running other
            side effects that do not edit or add notes as those should be handled through
            notes_to_add_dict and notes_to_update_dict.
    """
    total_notes = len(notes)
    note_cnt = 0
    for note in notes:
        try:
            note_was_edited = op(
                config=config,
                note=note,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
            )
        except Exception as e:
            logger.error("Sync bulk notes op: Error processing note %s: %s", note.id, e)
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
            if notes_to_update_dict and note.id in notes_to_update_dict:
                edited_nids.append(note.id)

    logger.debug("Sync bulk notes op finished. editedNids %s", edited_nids)

    if on_end:
        on_end()

    mw.taskman.run_on_main(lambda: mw.progress.finish())

    return pos, notes_to_add_dict, notes_to_update_dict


async def bulk_notes_op(
    message,
    config,
    op,
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]] = {},
    notes_to_update_dict: dict[NoteId, Note] = {},
    model: str = "",
    on_end: Optional[Callable[..., None]] = None,
) -> tuple[int, dict[str, list[Note]], dict[NoteId, Note]]:
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
        on_end: An optional callback to run on completion of the bulk op. Should be running other
            side effects that do not edit or add notes as those should be handled through
            notes_to_add_dict and notes_to_update_dict.
    """
    pos = col.add_custom_undo_entry(f"{message} for {len(notes)} notes.")
    config["rate_limits"] = config.get("rate_limits", {})
    if not model:
        logger.error("Model arg missing in bulk_notes_op, aborting")
        return pos, notes_to_add_dict, notes_to_update_dict
    rate_limit = config["rate_limits"].get(model, None)

    if not rate_limit:
        logger.debug(f"No rate limit set for model {model}, running sync op")
        return sync_bulk_notes_op(
            pos=pos,
            col=col,
            config=config,
            op=op,
            notes=notes,
            edited_nids=edited_nids,
            notes_to_add_dict=notes_to_add_dict,
            notes_to_update_dict=notes_to_update_dict,
            on_end=on_end,
        )

    updated_notes: list[Note] = []
    tasks: list[asyncio.Task] = []

    progress_updater.set_total_notes(len(notes))
    progress_updater.set_total_tasks(len(notes))
    # Can start auto updater now that we're in an async context with a running loop
    progress_updater.start_autoupdate()

    def handle_op_success(
        note: Note,
        was_success: bool,
    ):
        """Handle successful operation result."""
        if was_success and edited_nids is not None:
            updated_notes.append(note)
            edited_nids.append(note.id)
        logger.debug(f"Bulk notes op success for note {note.id}, was_success: {was_success}")

    cancel_state = CancelState()
    # Start all tasks
    for i, note in enumerate(notes):
        # adding the same note to
        def handle_error(current_note, e):
            logger.error(f"Error during operation with note {current_note.id}: {e}")
            print_error_traceback(e, logger)

        handle_op_error = partial(
            lambda current_note, e: handle_error(current_note, e),
            note,
        )

        handle_op_result = partial(
            lambda current_note, was_success: handle_op_success(current_note, was_success), note
        )
        process_note = make_inner_bulk_op(
            config=config,
            op=op,
            rate_limit=rate_limit,
            progress_updater=progress_updater,
            handle_op_error=handle_op_error,
            handle_op_result=handle_op_result,
            cancel_state=cancel_state,
            one_task_per_op=True,
        )
        if mw.progress.want_cancel():
            logger.debug("Bulk notes op cancelled before starting tasks")
            break
        task: asyncio.Task = asyncio.create_task(
            process_note(
                # task_index for process_op in make_inner_bulk_op
                task_index=i,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
                # note is passed to the op function, along with config in make_inner_bulk_op
                note=note,
            )
        )
        if len(tasks) % 5 == 0:
            # Update the progress dialog every 5 tasks gathered
            progress_updater.update_progress()
        tasks.append(task)
    logger.debug(f"Async bulk notes op started with {len(tasks)} tasks, rate limit: {rate_limit}")

    cancel_manager = CancelManager(tasks, cancel_state)

    # Wait for all tasks to complete
    try:
        logger.debug("Bulk notes op awaiting tasks")

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

        logger.debug("Bulk notes op completed successfully, all tasks finished")
    except asyncio.CancelledError:
        logger.debug("Bulk notes op asyncio.CancelledError caught")
        cancel_manager.request_cancel()
    finally:
        logger.debug("Bulk notes op finally block reached, cleaning up tasks")
        if not cancel_manager.monitor_task.done():
            cancel_manager.monitor_task.cancel()

    # Check if cancellation was requested and handle accordingly
    if cancel_manager.is_cancel_requested():
        logger.debug("Bulk notes op cancellation requested, returning early")
        if not cancel_manager.monitor_task.done():
            cancel_manager.monitor_task.cancel()
        if on_end:
            on_end()

        progress_updater.stop_autoupdate()
        return pos, notes_to_add_dict, notes_to_update_dict

    logger.debug("Bulk notes op completed successfully, updating notes")

    if on_end:
        on_end()

    progress_updater.stop_autoupdate()
    return pos, notes_to_add_dict, notes_to_update_dict


def on_bulk_success(
    out,
    done_text: str,
    edited_nids: Sequence[NoteId],
    edited_other_nids: Sequence[NoteId],
    nids: Sequence[NoteId],
    parent: Browser,
    # notes_to_add_dict: Optional[dict[str, list[Note]]] = None,
    extra_callback=None,
):
    mw.taskman.run_on_main(lambda: mw.progress.finish())
    # if DEBUG:
    # print("on_bulk_success", out, notes_to_add_dict)
    if extra_callback:
        extra_callback()
    # if notes_to_add_dict:
    #     new_notes: list[Note] = []
    #     for note_list in notes_to_add_dict.values():
    #         new_notes.extend(note_list)
    #     if new_notes:
    #         new_notes_tsv_str = make_tsv_from_notes(
    #             notes=new_notes,
    #             config=mw.addonManager.getConfig(__name__) or {},
    #         )
    #         if new_notes_tsv_str:
    #             # Write the TSV to the media folder
    #             import_tsv_file(
    #                 "new_notes.tsv",
    #                 new_notes_tsv_str,
    #             )
    # Show a tooltip after the import call as otherwise the import dialog would close the tooltip
    # immediately after it had appeared
    message = f"{done_text} in {len(edited_nids)}/{len(nids)} selected notes."
    if edited_other_nids:
        message += f"<br>Edited {len(edited_other_nids)} other notes not among the selection."
    tooltip(
        message,
        parent=parent,
        period=5000,
    )


NewNotesOp = Callable[[list[Note], dict, AsyncTaskProgressUpdater], dict[NoteId, Note]]


def selected_notes_op(
    done_text: str,
    bulk_op,
    nids: Sequence[NoteId],
    parent: Browser,
    progress_updater: AsyncTaskProgressUpdater,
    new_notes_op: Optional[NewNotesOp] = None,
    on_success: Optional[Callable] = None,
):
    edited_nids: list[NoteId] = []
    edited_other_nids: list[NoteId] = []
    notes_to_add_dict: dict[str, list[Note]] = {}
    notes_to_update_dict: dict[NoteId, Note] = {}
    config = mw.addonManager.getConfig(__name__) or {}

    # Create a wrapper function that handles the async operation
    def run_bulk_op(col: Collection) -> OpChanges:
        async def async_wrapper():
            nonlocal edited_nids, edited_other_nids
            result = await bulk_op(
                col,
                notes=[mw.col.get_note(nid) for nid in nids],
                edited_nids=edited_nids,
                progress_updater=progress_updater,
                notes_to_add_dict=notes_to_add_dict,
                notes_to_update_dict=notes_to_update_dict,
            )
            pos, res_notes_to_add_dict, res_notes_to_update_dict = result
            logger.debug(f"res_notes_to_update_dict keys: {res_notes_to_update_dict.keys()}")
            notes_to_add_dict.update(res_notes_to_add_dict)
            notes_to_update_dict.update(res_notes_to_update_dict)
            logger.debug(f"notes_to_update_dict keys: {notes_to_update_dict.keys()}")
            edited_nids_set = set(edited_nids)
            for nid in res_notes_to_update_dict.keys():
                if nid not in edited_nids_set:
                    edited_other_nids.append(nid)
            edited_nids = list(filter(lambda x: x in nids, notes_to_update_dict.keys()))

            # Remove note.id=0 notes from updated_notes
            all_updated_notes_dict: dict[NoteId, Note] = {}
            for note in notes_to_update_dict.values():
                if note.id == 0:
                    logger.error(f"Found note.id=0, fields: {note.fields}")
                elif note.id not in all_updated_notes_dict:
                    all_updated_notes_dict[note.id] = note
                else:
                    # If the note is already in notes_to_update_dict, this might be a problem in the
                    # logic of the bulk op
                    logger.warning(
                        f"Note {note.id} occurring multiple times in notes_to_update_dict during"
                        " bulk op final update"
                    )
            all_updated_notes = [n for n in all_updated_notes_dict.values() if n.id != 0]
            try:
                mw.col.update_notes(all_updated_notes)
            except Exception as e:
                logger.error(f"Error updating notes: {e}")
                logger.error(f"Notes causing error: {[n.fields for n in all_updated_notes]}")
                print_error_traceback(e, logger)
            op_changes = mw.col.merge_undo_entries(pos)
            notes_to_add = []
            if notes_to_add_dict:
                for note_list in notes_to_add_dict.values():
                    notes_to_add.extend(note_list)
            if notes_to_add:
                logger.debug(
                    f"Adding {len(notes_to_add)} new notes to note_will_be_added hooks will be run"
                )
                total_notes = len(notes_to_add)
                failed_cnt = 0
                added_cnt = 0
                for index, note in enumerate(notes_to_add):
                    note_type = note.note_type()
                    if note_type is None:
                        logger.debug(
                            f"Error: Note type for note {note.id} is None, skipping note adding"
                        )
                        continue
                    insert_deck = get_field_config(config, "insert_deck", note_type)
                    insert_deck_id = None
                    if insert_deck:
                        insert_deck_id = mw.col.decks.id_for_name(insert_deck)
                    else:
                        insert_deck_id = mw.col.decks.id_for_name("Default")
                        logger.debug("No insert deck set, setting deck_id to Default")
                    if insert_deck_id is None:
                        logger.debug("Default deck not found, skipping note adding")
                        continue
                    if mw.progress.want_cancel():
                        logger.debug("Bulk notes op cancelled during note adding")
                        break
                    try:
                        logger.debug(f"Adding note {index} to deck {insert_deck_id}")
                        mw.col.add_note(note, insert_deck_id)
                        added_cnt += 1
                        op_changes = mw.col.merge_undo_entries(pos)
                    except Exception as e:
                        logger.error(f"Error adding note {index}: {e}")
                        print_error_traceback(e, logger)
                        failed_cnt += 1

                    progress_updater.update_note_adding_progress(
                        notes_added=added_cnt,
                        total_notes=total_notes,
                        failed=failed_cnt,
                    )
                if new_notes_op:
                    # Run the new notes operation if provided
                    # col.add_note mutates the note given, adding the id to it
                    additional_updates_notes_dict = new_notes_op(
                        notes_to_add, config, progress_updater
                    )

                    additional_updated_notes = list(additional_updates_notes_dict.values())
                    if additional_updated_notes:
                        # Skip notes where the id is still zero, something went wrong during adding
                        valid_notes = []
                        invalid_notes = []
                        for note in additional_updated_notes:
                            if note.id == 0:
                                invalid_notes.append(note)
                            else:
                                valid_notes.append(note)
                        if invalid_notes:
                            logger.debug(f"Invalid notes found after adding: {len(invalid_notes)}")
                            new_notes_tsv_str = make_tsv_from_notes(
                                notes=invalid_notes,
                                config=mw.addonManager.getConfig(__name__) or {},
                            )
                            if new_notes_tsv_str:
                                # Write the TSV to the media folder
                                import_tsv_file(
                                    "new_notes.tsv",
                                    new_notes_tsv_str,
                                    do_import=False,
                                )
                        try:
                            mw.col.update_notes(valid_notes)
                        except Exception as e:
                            logger.error(f"Error updating valid notes after new_notes_op: {e}")
                            print_error_traceback(e, logger)
                        op_changes = mw.col.merge_undo_entries(pos)
                        edited_nids.extend(
                            [note.id for note in valid_notes if note.id not in edited_nids]
                        )
            return op_changes

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
        .success(
            lambda out: on_bulk_success(
                out,
                done_text,
                edited_nids,
                edited_other_nids,
                nids,
                parent,
                # notes_to_add_dict,
                on_success,
            )
        )
        .run_in_background()
    )
