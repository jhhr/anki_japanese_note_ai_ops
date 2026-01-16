import os
from enum import Enum
from typing import TypedDict, Union
from anki.notes import NoteId

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_USER_FILES_DIR = os.path.join(
    ADDON_DIR,
    "user_files",
)

# Ensure the user_files directory exists
os.makedirs(ADDON_USER_FILES_DIR, exist_ok=True)


# Raw word, tuple of 1) word, 2) reading
RawOneMeaningWordType = tuple[str, str]
# Raw word, tuple of 1) word, 2) reading and 3) meaning number
# meaning number is used to indicate the same word and reading occurring with different meanings
RawMultiMeaningWordType = tuple[str, str, int]
# Matched word, same but with note sort field value and note ID
# 1) word, 2) reading, 3) note_sort_field_value, 4) note_id +int or fake note Id -int)
# note_sort_field_value is different for each meaning a word with the same reading can have
# so it is used to distinguish between them
# The note_id references the exact note that the word is matched to, it can be a real note ID
# or a placeholder ID used to identify new note that is to be created but hasn't yet
OneMeaningMatchedWordType = tuple[str, str, str, Union[NoteId, int]]
MultiMeaningMatchedWordType = tuple[str, str, int, str, Union[NoteId, int]]

MEANINGS_DICT_FILE = "_all_meanings_dict.json"
KANJI_STORY_COMPONENT_WORDS_LOG = "_kanji_story_component_words.json"


# The json is a dict of "word_reading" to an array of dicts
class GeneratedMeaningType(TypedDict):
    jp_meaning: str
    en_meaning: str


GeneratedMeaningsDictType = dict[str, list[GeneratedMeaningType]]

NO_DICTIONARY_ENTRY_TAG = "2-no-dictionary-entry"
MEANINGS_GENERATED_TAG = "2-meanings-generated-to-json"
MEANING_MAPPED_TAG = "2-note-mapped-to-generated-meaning"


class EnAndJPSentence(TypedDict):
    jp_sentence: str
    en_sentence: str


class WordAndSentences(TypedDict):
    jp_meaning: str
    en_meaning: str
    sentences: list[EnAndJPSentence]


class MakeMeaningsResult(Enum):
    SUCCESS = 1
    NO_DICTIONARY_ENTRY = 2
    ERROR = 3
