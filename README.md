Simple Anki addon for some custom AI prompts I use.

- Provides a framework of functions to use for adding new prompts.
- Prompts are available in right-click menu of notes in Anki's card browser

Prompts:

- `clean_meaning`: Automatically reduce a full jp dictionary entry gotten from Yomichan/Yomitan/Rikaitan etc. into just the part that pertains to the specific example sentence used for the word. Also generates new meanings when no dict entry is present. Applied on adding new notes, so I used it often. Intended to be used with a deck structure where you have multiple notes for different meanings of a word which each contains an example sentence for that particular meaning.
- `translate_field`: Translate a field from Japanese to English. Used rarely, since I mostly mine from anime and get the translation from the english subs.
- `make_kanji_story`: Write a mnemonic story for the components a kanji is written with, in Japanese. Used daily on new kanji drawing practice notes. Expects a JSON file `_kanji_story_component_words.json` to exist and be a simple dict of component phrases.
- `write_kanji_component_words`: Used once to generate the `_kanji_story_component_words.json` from my notes before when I was beginning to use the `make_kanji_story` operation. Didn't work well, I needed to add lots of component phrases I'd already come up with manually anyway and it generated a bunch of junk too.

## Installing dependencies

Run `pip3 install --upgrade -t lib --no-cache-dir --python-version 3.9 --only-binary=:all: -r requirements.txt`
