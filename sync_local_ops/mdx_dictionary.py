import logging
from typing import Any, Literal, Optional, TypedDict, Union
import os
import time
import sqlite3
import re

from aqt import mw

try:
    from mdict_query import IndexBuilder  # type: ignore
except ImportError:
    IndexBuilder = None
    print("mdict-query library not available, see README for installation instructions")

from ..html_stripping import strip_html_advanced
from ..configuration import ADDON_USER_FILES_DIR

logger = logging.getLogger(__name__)


class MDXDictionary:
    """Efficient MDX dictionary querying using mdict-query's IndexBuilder"""

    # Maximum recursion depth when following links to prevent infinite loops
    MAX_LINK_DEPTH = 10

    def __init__(
        self, mdx_path: str, show_progress: bool = False, progress_msg=None, finish: bool = True
    ):
        """
        Initialize MDX dictionary

        Args:
            mdx_path: Path to .mdx file
        """
        if IndexBuilder is None:
            raise ImportError(
                "mdict-query library not available, see README for installation instructions"
            )

        self.mdx_path = mdx_path
        start_time = time.time()
        load_msg = f"Loading MDX: {os.path.basename(mdx_path)}"
        if show_progress:
            mw.taskman.run_on_main(
                lambda: mw.progress.update(
                    label=progress_msg or load_msg,
                )
            )

        # IndexBuilder automatically creates a SQLite index for fast lookups
        # It will reuse existing .mdx.db file if available
        self.builder = IndexBuilder(mdx_path, sql_index=True, check=False)

        elapsed = time.time() - start_time
        loaded_msg = f"""Loaded MDX file: {os.path.basename(mdx_path)} in {elapsed:.2f}s
    Dictionary: {self.builder._title}")
    Description: {self.builder._description[:100] if self.builder._description else 'N/A'}"""
        print(loaded_msg)
        if show_progress and finish:
            mw.taskman.run_on_main(lambda: mw.progress.finish())

    def _parse_link_entries(self, result: str) -> list[str]:
        """Parse @@@LINK= entries from a dictionary result.

        Args:
            result: Dictionary result that may contain @@@LINK= markers

        Returns:
            List of linked entry names, or empty list if no links found
        """

        # Match @@@LINK=<entry_name> patterns
        # The entry name may contain Japanese characters, brackets, and other symbols
        link_pattern = r"@@@LINK=(.+?)(?=\n|$)"
        matches = re.findall(link_pattern, result)

        if not matches:
            return []

        # Clean up the linked entries (remove extra whitespace)
        linked_entries = [match.strip() for match in matches]
        logger.debug(f"Found {len(linked_entries)} link(s): {linked_entries}")
        return linked_entries

    def _is_link_only_result(self, result: str) -> bool:
        """Check if result contains only links without actual content.

        Args:
            result: Dictionary result to check

        Returns:
            True if result only contains @@@LINK= markers
        """
        # Remove all @@@LINK= lines and whitespace
        import re

        cleaned = re.sub(r"@@@LINK=.+?(?=\n|$)", "", result)
        cleaned = cleaned.strip()
        return len(cleaned) == 0

    def _follow_links(self, result: str, depth: int = 0) -> Union[str, None]:
        """Follow @@@LINK= references to get actual definitions.

        Args:
            result: Dictionary result that may contain links
            depth: Current recursion depth

        Returns:
            Actual definition content, or None if links lead nowhere
        """
        # Prevent infinite recursion
        if depth >= self.MAX_LINK_DEPTH:
            logger.warning(f"Max link depth ({self.MAX_LINK_DEPTH}) reached, stopping recursion")
            return result

        # Check if this result contains only links
        if not self._is_link_only_result(result):
            # Result has actual content, return it
            return result

        # Parse out the linked entries
        linked_entries = self._parse_link_entries(result)
        if not linked_entries:
            # No links found but also no content - return as is
            return result

        # Try to follow the links
        all_linked_results = []
        for entry in linked_entries:
            logger.debug(f"Following link to: {entry} (depth {depth + 1})")
            # Query the linked entry
            linked_result = self.builder.mdx_lookup(entry, ignorecase=False)

            if linked_result:
                # Join all results for this entry
                joined_result = "\n".join(r for r in linked_result if r)

                # Recursively follow links in the result
                final_result = self._follow_links(joined_result, depth + 1)
                if final_result:
                    all_linked_results.append(final_result)

        if not all_linked_results:
            # None of the links led to actual content
            return None

        # Combine all results
        return "\n\n".join(all_linked_results)

    def query(
        self,
        query: Union[str, list[str]],
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
        ignorecase: bool = True,
        match_whole_word: bool = False,
    ) -> Union[str, None]:
        """
        Query a single word using mdict-query's smart lookup

        Args:
            query: The word or list of words to look up. If a list is provided, will perform
                   a lookup for keys containing all words (using SQL LIKE for partial matching).
            strip_html_tags: If True, remove HTML tags from result
            preserve_structure: If True and strip_html_tags=True, preserve line breaks
            ignorecase: If True, perform case-insensitive lookup (default: True)
            match_whole_word: If True and query is a list, only match keys where each word
                             appears as a complete unit (not as part of a longer word)

        Returns:
            HTML or plain text definition string, or None if not found
        """
        logger.debug(f"Querying MDX dictionary {os.path.basename(self.mdx_path)} for: {query}")
        try:
            # If query is a list, find keys containing all the words
            if isinstance(query, list):
                matching_keys = self._find_keys_containing_all_words(query, match_whole_word)
                if not matching_keys:
                    logger.debug(
                        f"No keys found containing all words: {query} in"
                        f" {os.path.basename(self.mdx_path)}"
                    )
                    return None

                # Collect all matching entries
                all_results = []
                for key in matching_keys:
                    entry_result = self.builder.mdx_lookup(key, ignorecase=False)
                    if entry_result:
                        all_results.extend(entry_result)

                if not all_results:
                    return None

                result = "\n\n".join(result for result in all_results if result)
            else:
                # Single word lookup using mdict-query's built-in method
                # mdx_lookup returns a list of matching results
                # With ignorecase=True, it will find matches regardless of case
                results = self.builder.mdx_lookup(query, ignorecase=ignorecase)

                if not results:
                    logger.debug(f"Word '{query}' not found in {os.path.basename(self.mdx_path)}")
                    return None

                # Take all results and join them
                result = "\n".join(result for result in results if result)

            # Follow any @@@LINK= references to get actual definitions
            result = self._follow_links(result)

            if not result:
                logger.debug(
                    f"Links for '{query}' led to no content in {os.path.basename(self.mdx_path)}"
                )
                return None

            if strip_html_tags:
                result = strip_html_advanced(result, preserve_structure)

            logger.debug(f"Query result for '{query}': {result}")
            return result
        except Exception as e:
            logger.error(f"Error querying '{query}': {e}")
            return None

    def query_japanese(
        self,
        word: str,
        reading: Optional[str] = None,
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
    ) -> Union[str, None]:
        """
        Query Japanese word with multiple fallback strategies

        Args:
            word: The word to look up (can be kanji, hiragana, or katakana)
            reading: Optional hiragana reading to try as fallback
            strip_html_tags: If True, remove HTML tags from result
            preserve_structure: If True and strip_html_tags=True, preserve line breaks

        Returns:
            HTML or plain text definition string, or None if not found
        """
        # Strategy 1: If both word and reading provided, search for entries containing both
        # This handles formats like "reading【kanji】" common in Japanese dictionaries
        if reading and reading != word:
            result = self.query(
                [word, reading],
                strip_html_tags,
                preserve_structure,
                ignorecase=False,
                match_whole_word=True,
            )
            if result:
                return result

            # Strategy 2: Try to find partial matches using wildcard search
            # This helps with Japanese dictionaries that have compound entries
            try:
                keys = self.get_keys_by_prefix(reading)
                if keys:
                    # Filter keys that contain the word
                    filtered_keys = [k for k in keys if word in k]
                    if filtered_keys:
                        all_results = []
                        for key in filtered_keys:
                            entry_result = self.query(
                                key, strip_html_tags, preserve_structure, ignorecase=False
                            )
                            if entry_result:
                                all_results.append(entry_result)

                        if all_results:
                            return "\n\n".join(all_results)
            except Exception as e:
                logger.debug(f"Wildcard search failed for '{word}': {e}")

        # Strategy 3: Try just the word
        result = self.query(word, strip_html_tags, preserve_structure, ignorecase=True)
        if result:
            return result

        # Strategy 4: If no result yet, try just the reading
        if reading and reading != word:
            result = self.query(reading, strip_html_tags, preserve_structure, ignorecase=True)
            if result:
                return result

        # Nothing worked
        return None

    def _find_keys_containing_all_words(
        self, words: list[str], match_whole_word: bool = False
    ) -> list[str]:
        """
        Find dictionary keys that contain all specified words

        Args:
            words: List of words/strings that must all appear in the key
            match_whole_word: If True, only match keys where each word appears as a complete
                             unit (not as part of a longer word). Uses regex-like word boundary
                             logic where word characters are separated by non-word characters.

        Returns:
            List of matching keys
        """
        try:
            # Query the SQLite index for keys containing all words
            db_path = self.builder._mdx_db
            if not os.path.exists(db_path):
                return []

            with sqlite3.connect(db_path) as conn:
                if match_whole_word:
                    # Get candidate keys using LIKE for initial filtering
                    conditions = " AND ".join(["key_text LIKE ?"] * len(words))
                    sql = f"SELECT key_text FROM MDX_INDEX WHERE {conditions}"
                    params = tuple(f"%{word}%" for word in words)

                    cursor = conn.execute(sql, params)
                    candidate_keys = [row[0] for row in cursor.fetchall()]

                    # Filter to only keys where each word appears as a complete unit
                    # A word is complete if it's not preceded or followed by other word characters
                    import re

                    filtered_keys = []
                    for key in candidate_keys:
                        all_words_match = True
                        for word in words:
                            # Create pattern that matches word with word boundaries
                            # Word boundary = start/end of string or non-alphanumeric character
                            # Escape special regex characters in the word
                            escaped_word = re.escape(word)
                            # Match word that is either at boundaries or surrounded by non-word chars
                            # Using lookahead/lookbehind for zero-width boundary assertions
                            pattern = f"(?<![a-zA-Z0-9ぁ-ゟァ-ヿ一-龯]){escaped_word}(?![a-zA-Z0-9ぁ-ゟァ-ヿ一-龯])"
                            if not re.search(pattern, key):
                                all_words_match = False
                                break

                        if all_words_match:
                            filtered_keys.append(key)

                    return filtered_keys
                else:
                    # Standard partial matching with LIKE
                    conditions = " AND ".join(["key_text LIKE ?"] * len(words))
                    sql = f"SELECT key_text FROM MDX_INDEX WHERE {conditions}"
                    params = tuple(f"%{word}%" for word in words)

                    cursor = conn.execute(sql, params)
                    keys = [row[0] for row in cursor.fetchall()]
                    return keys
        except Exception as e:
            logger.error(f"Error searching for keys containing all words {words}: {e}")
            return []

    def get_keys_by_prefix(self, prefix: str, max_results: int = 10) -> list[str]:
        """
        Get all dictionary keys starting with prefix using SQL LIKE query

        Args:
            prefix: Prefix to search for
            max_results: Maximum number of results to return

        Returns:
            List of matching keys
        """
        try:
            # Use wildcard pattern for prefix search
            pattern = f"{prefix}*"
            keys = self.builder.get_mdx_keys(pattern)
            return keys[:max_results] if keys else []
        except Exception as e:
            logger.error(f"Error getting keys by prefix '{prefix}': {e}")
            return []

    def query_multiple(
        self,
        words: list[str],
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
    ) -> dict[str, Union[str, None]]:
        """
        Query multiple words efficiently

        Args:
            words: List of words to look up
            strip_html_tags: If True, remove HTML tags from results
            preserve_structure: If True and strip_html_tags=True, preserve line breaks

        Returns:
            Dictionary mapping words to their definitions
        """
        results = {}
        for word in words:
            results[word] = self.query(word, strip_html_tags, preserve_structure)
        return results


MDXDictionaryEntry = TypedDict(
    "MDXDictionaryEntry",
    {
        "path": str,
        "name": str,
        "dict": MDXDictionary,
    },
)

PickDictionaryResult = Union[Literal["first", "all", "shortest", "longest"], str]


class MultiDictionaryQuery:
    """Query multiple MDX dictionaries simultaneously"""

    def __init__(
        self, mdx_paths: list[str], show_progress: bool = False, finish_progress: bool = True
    ):
        """
        Initialize with multiple MDX files

        Args:
            mdx_paths: List of paths to .mdx files
        """
        self.dictionaries: list[MDXDictionaryEntry] = []
        start_time = time.time()
        print("\nLoading MDX dictionaries...")
        path_count = len(mdx_paths)
        for i, path in enumerate(mdx_paths):
            if os.path.exists(path):
                try:
                    progress_msg = (
                        f"Loading MDX dictionary {i + 1} / {path_count}: {os.path.basename(path)}"
                    )
                    is_last = i == path_count - 1
                    mdx_dict = MDXDictionary(
                        path,
                        show_progress=show_progress,
                        progress_msg=progress_msg,
                        finish=is_last and finish_progress,
                    )
                    dict_entry: MDXDictionaryEntry = {
                        "path": path,
                        "name": mdx_dict.builder._title or os.path.basename(path),
                        "dict": mdx_dict,
                    }
                    self.dictionaries.append(dict_entry)
                except Exception as e:
                    print(f"Failed to load MDX file {path}: {e}")
            else:
                print(f"MDX file not found: {path}")
        elapsed = time.time() - start_time
        print(f"All MDX dictionaries loaded in {elapsed:.2f}s\n")

    def query(
        self,
        word: str,
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
        pick_dictionary: PickDictionaryResult = "all",
    ) -> list[dict[str, str]]:
        """
        Query word in all dictionaries

        Args:
            word: Word to look up
            strip_html_tags: If True, remove HTML tags from results
            preserve_structure: If True and strip_html_tags=True, preserve line breaks
            pick_dictionary: Strategy for selecting results, one of "first", "all", "shortest",
                "longest"

        Returns:
            List of dicts with 'dictionary' and 'definition' keys
        """
        results: list[dict[str, str]] = []
        for d in self.dictionaries:
            definition = d["dict"].query(word, strip_html_tags, preserve_structure)
            if definition:
                results.append({"dictionary": d["name"], "definition": definition})
                if pick_dictionary == "first":
                    return results
        if pick_dictionary == "shortest" and results:
            shortest = min(results, key=lambda x: len(x["definition"]))
            return [shortest]
        elif pick_dictionary == "longest" and results:
            longest = max(results, key=lambda x: len(x["definition"]))
            return [longest]
        # Default: return all results
        return results

    def query_all_japanese(
        self,
        word: str,
        reading: Optional[str] = None,
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
        pick_dictionary: PickDictionaryResult = "all",
    ) -> list[dict[str, str]]:
        """
        Query Japanese word in all dictionaries with fallback strategies

        Args:
            word: Word to look up
            reading: Optional reading
            strip_html_tags: If True, remove HTML tags from results
            preserve_structure: If True and strip_html_tags=True, preserve line breaks
            pick_dictionary: Strategy for selecting results, one of "first", "all", "shortest",
                "longest"

        Returns:
            List of dicts with 'dictionary' and 'definition' keys
        """
        results = []
        for d in self.dictionaries:
            definition = d["dict"].query_japanese(
                word, reading, strip_html_tags, preserve_structure
            )
            if definition:
                results.append({"dictionary": d["name"], "definition": definition})
                if pick_dictionary == "first":
                    return results
        if pick_dictionary == "shortest" and results:
            shortest = min(results, key=lambda x: len(x["definition"]))
            return [shortest]
        elif pick_dictionary == "longest" and results:
            longest = max(results, key=lambda x: len(x["definition"]))
            return [longest]
        # Default: return all results
        return results


# For Anki addon usage
class AnkiMDXHelper:
    """Helper class for using MDX dictionaries in Anki addons"""

    def __init__(self):
        """
        Initialize empty helper. Call init_helper(config) to load dictionaries.
        """
        self.multi_dict: Union[MultiDictionaryQuery, None] = None
        self._init_failed = False

    def load_mdx_dictionaries_if_needed(
        self, config: dict[str, Any], show_progress: bool = False, finish_progress: bool = True
    ) -> Union["AnkiMDXHelper", None]:
        """
        Initialize with Anki addon config and load MDX dictionaries.
        Returns self if successful, None if initialization fails.

        Args:
            config: Anki addon config dict with 'mdx_filenames' key

        Returns:
            Self if successful, None if all dictionaries fail to load
        """
        if self._init_failed:
            return None

        if self.multi_dict is not None:
            return self  # Already initialized

        try:
            mdx_filenames = config.get("mdx_filenames", [])
            if not mdx_filenames:
                print("No MDX filenames configured")
                self._init_failed = True
                return None

            mdx_paths = [os.path.join(ADDON_USER_FILES_DIR, fn) for fn in mdx_filenames]
            self.multi_dict = MultiDictionaryQuery(
                mdx_paths, show_progress=show_progress, finish_progress=finish_progress
            )

            # Check if any dictionaries were actually loaded
            if not self.multi_dict.dictionaries:
                print("No MDX dictionaries were successfully loaded")
                self._init_failed = True
                self.multi_dict = None
                return None

            return self
        except Exception as e:
            print(f"Failed to initialize MDX helper: {e}")
            self._init_failed = True
            self.multi_dict = None
            return None

    def get_definition_text(
        self,
        word: str,
        reading: Optional[str] = None,
        pick_dictionary: PickDictionaryResult = "all",
        max_length: Optional[int] = None,
    ) -> Union[str, None]:
        """
        Get plain text definition

        Args:
            word: Word to look up
            reading: Optional reading (for Japanese)
            max_length: Optional maximum character length for truncation

        Returns:
            Plain text string with definitions from all dictionaries
        """
        if self.multi_dict is None:
            return None

        if reading:
            results = self.multi_dict.query_all_japanese(
                word,
                reading,
                strip_html_tags=True,
                preserve_structure=True,
                pick_dictionary=pick_dictionary,
            )
        else:
            results = self.multi_dict.query(
                word, strip_html_tags=True, preserve_structure=True, pick_dictionary=pick_dictionary
            )

        if not results:
            return None

        # Build plain text output
        lines = [word]
        if reading:
            lines.append(f"({reading})")
        lines.append("")

        for result in results:
            lines.append(f"[{result['dictionary']}]")
            definition = result["definition"]

            # Truncate if needed
            if max_length and len(definition) > max_length:
                definition = definition[:max_length] + "..."

            lines.append(definition)
            lines.append("")

        return "\n".join(lines)

    def get_definition_html(
        self,
        word: str,
        reading: Optional[str] = None,
        pick_dictionary: PickDictionaryResult = "all",
    ) -> Union[str, None]:
        """
        Get formatted HTML definition for word (for UI display)

        Args:
            word: Word to look up
            reading: Optional reading (for Japanese)

        Returns:
            HTML string with definitions from all dictionaries
        """
        if self.multi_dict is None:
            return None

        if reading:
            results = self.multi_dict.query_all_japanese(
                word, reading, pick_dictionary=pick_dictionary
            )
        else:
            results = self.multi_dict.query(word, pick_dictionary=pick_dictionary)
        if not results:
            return f"<p>No definition found for '{word}'</p>"

        html = f"<h2>{word}</h2>"
        if reading:
            html += f"<p><i>{reading}</i></p>"

        for result in results:
            html += f"<h3>{result['dictionary']}</h3>"
            html += f"<div>{result['definition']}</div>"
            html += "<hr>"

        return html
