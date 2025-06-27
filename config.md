# Config

## models

Needs to be one of the models that supports structured output

### OpenAI

- `gpt-4o` and `gpt-4o-*` models
- `gpt-4` and `gpt-4-*` models
- `gpt-3.5-turbo`

### Google

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash` (default, free)

## sentence_field / meaning_field / word_field

### model

Define which model to use for each task

- `word_meaning_model`
- `kanji_story_model`
- `translate_sentence_model`
- `kanjify_sentence_model`
- `extract_words_model`

### model rate limits

Must be defined for each model. Default are set very low. Check the respective API docs
for what rate limits you may be able to / want to use for each model.

Add the fields by note type like this. You can set multiple different note types. You can't set
multiple fields per note type though.

```
{
  "note type name A": {
    "meaning_field": "note A meaning field",
    "word_field": "note A word field",
    "sentence_field": "note A sentence field"
  },
  "note type name B": {
    "meaning_field": "note B meaning field",
    "word_field": "note B word field",
    "sentence_field": "note B sentence field",
    "translation_field": "note B translation field"
  },
  ...etc
}
```

You must define

- for cleaning/generating word meanings:
  1. `meaning_field`
  2. `word_field`
  3. `sentence_field`
- for translating sentences
  1. `sentence_field`
  2. `translated_sentence_field`
- for generating kanji stories:
  1. `kanji_field`
  2. `kanji_story_field`
- for kanjifying sentences:
  1. `furigana_sentence_field`
  2. `kanjified_sentence_field`
- for extracting words:
  1. `furigana_sentence_field`
  2. `word_list_field`
