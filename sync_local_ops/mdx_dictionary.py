from mdict_utils.base.readmdict import MDX  # type: ignore
from typing import Any, Literal, Optional, TypedDict, Union
import os
import time

from aqt import mw

from ..html_stripping import strip_html_advanced
from ..configuration import ADDON_USER_FILES_DIR


class MDXDictionary:
    """Efficient MDX dictionary querying using mdict-utils"""

    def __init__(self, mdx_path: str):
        """
        Initialize MDX dictionary

        Args:
            mdx_path: Path to .mdx file
        """
        self.mdx_path = mdx_path
        start_time = time.time()
        print(f"Loading MDX file: {os.path.basename(mdx_path)}...")
        self.mdx: MDX = MDX(mdx_path)
        elapsed = time.time() - start_time
        print(f"Loaded MDX file: {os.path.basename(mdx_path)} in {elapsed:.2f}s")
        self._build_index()

    def _build_index(self):
        """Build an in-memory index for fast lookups"""
        start_time = time.time()
        print(f"Building index for {os.path.basename(self.mdx_path)}...")
        self.index: dict[str, str] = {}
        entry_count = 0
        for key, value in self.mdx.items():
            # Decode bytes to strings if necessary
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            self.index[key] = value
            entry_count += 1
        elapsed = time.time() - start_time
        print(f"    indexed {entry_count} entries in {elapsed:.2f}s")

    def query(
        self,
        word: str,
        strip_html_tags: bool = False,
        preserve_structure: bool = False,
        _visited: Optional[set[str]] = None,
    ) -> Union[str, None]:
        """
        Query a single word

        Args:
            word: The word to look up
            strip_html_tags: If True, remove HTML tags from result
            preserve_structure: If True and strip_html_tags=True, preserve line breaks
            _visited: Internal parameter to track visited links and prevent infinite recursion

        Returns:
            HTML or plain text definition string, or None if not found
        """

        # Check for infinite recursion
        if _visited is not None and word in _visited:
            print(f"Warning: Circular reference detected for '{word}'")
            return None

        # Try exact match first
        if word in self.index:
            result = self.index[word]
        elif word.lower() in self.index:
            result = self.index[word.lower()]
        else:
            return None

        if result.startswith("@@@LINK="):
            if _visited is None:
                _visited = set()
            # Handle cross-reference links
            link_key = result[len("@@@LINK=") :].strip()
            _visited.add(word)
            return self.query(link_key, strip_html_tags, preserve_structure, _visited)

        if strip_html_tags:
            result = strip_html_advanced(result, preserve_structure)

        return result

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
        # Strategy 1: Try the word as-is
        result = self.query(word, strip_html_tags, preserve_structure)
        if result:
            return result

        # Strategy 2: If a reading was provided, try it
        if reading and reading != word:
            result = self.query(reading, strip_html_tags, preserve_structure)
            if result:
                return result

        # Strategy 3: Try katakana conversion (requires pykakasi)
        try:
            import pykakasi

            kks = pykakasi.kakasi()
            converted = kks.convert(word)
            for item in converted:
                # Try hiragana version
                if "hira" in item and item["hira"] != word:
                    result = self.query(item["hira"], strip_html_tags, preserve_structure)
                    if result:
                        return result
                # Try katakana version
                if "kana" in item and item["kana"] != word:
                    result = self.query(item["kana"], strip_html_tags, preserve_structure)
                    if result:
                        return result
        except ImportError:
            pass  # pykakasi not available

        return None

    def get_keys_by_prefix(self, prefix: str) -> list[str]:
        """
        Get all dictionary keys starting with prefix

        Args:
            prefix: Prefix to search for

        Returns:
            List of matching keys
        """
        prefix_lower = prefix.lower()
        return [key for key in self.index.keys() if key.lower().startswith(prefix_lower)]

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

PickDictionaryResult = Literal["first", "all", "shortest", "longest"]


class MultiDictionaryQuery:
    """Query multiple MDX dictionaries simultaneously"""

    def __init__(self, mdx_paths: list[str]):
        """
        Initialize with multiple MDX files

        Args:
            mdx_paths: List of paths to .mdx files
        """
        self.dictionaries: list[MDXDictionaryEntry] = []
        start_time = time.time()
        print("\nLoading MDX dictionaries...")
        for path in mdx_paths:
            if os.path.exists(path):
                try:
                    self.dictionaries.append(
                        {"path": path, "name": os.path.basename(path), "dict": MDXDictionary(path)}
                    )
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
        reading: Union[str, None] = None,
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

    def load_mdx_dictionaries_if_needed(self, config) -> Union["AnkiMDXHelper", None]:
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
            self.multi_dict = MultiDictionaryQuery(mdx_paths)

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
        self, word: str, reading: Optional[str] = None, max_length: Optional[int] = None
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

        cur_config: dict[str, Any] = mw.addonManager.getConfig(__name__) or {}
        pick_dictionary: PickDictionaryResult = cur_config.get("mdx_pick_dictionary", "all")
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

    def get_definition_html(self, word: str, reading: Optional[str] = None) -> Union[str, None]:
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

        cur_config: dict[str, Any] = mw.addonManager.getConfig(__name__) or {}
        pick_dictionary: PickDictionaryResult = cur_config.get("mdx_pick_dictionary", "all")
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
