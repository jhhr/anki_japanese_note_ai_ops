# Config

## model

Either `gpt-4-1106-preview` or `gpt-3.5-turbo-1106`. Use GPT4, if you're ok with using the better but more expensive one.

## sentence_field / meaning_field / word_field

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

- for cleaning/generating meanings:
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
  3. `hiraganaified_sentence_field`
