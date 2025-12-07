import os
from typing import Union
from anki.notes import NoteId

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_USER_FILES_DIR = os.path.join(
    ADDON_DIR,
    "user_files",
)

# Ensure the user_files directory exists
os.makedirs(ADDON_USER_FILES_DIR, exist_ok=True)
# Raw word, tuple of 1) word, 2) reading
raw_one_meaning_word_type = tuple[str, str]
# Raw word, tuple of 1) word, 2) reading and 3) meaning number
# meaning number is used to indicate the same word and reading occurring with different meanings
raw_multi_meaning_word_type = tuple[str, str, int]
# Matched word, same but with note sort field value and note ID
# 1) word, 2) reading, 3) note_sort_field_value, 4) note_id +int or fake note Id -int)
# note_sort_field_value is different for each meaning a word with the same reading can have
# so it is used to distinguish between them
# The note_id references the exact note that the word is matched to, it can be a real note ID
# or a placeholder ID used to identify new note that is to be created but hasn't yet
matched_word_type = tuple[str, str, str, Union[NoteId, int]]
