# Config

## model

Either `gpt-4-1106-preview` or `gpt-3.5-turbo-1106`. Use GPT4, if you're ok with using the better but more expensive one.

## sentence_field / meaning_field / word_field

Add the fields by note type like this. You can set multiple different note types. You can't set
multiple fields per note type though.

```json
"meaning_field: {
  "note type name A"`: `"note A meaning field"
  "note type name B"`: `"note B meaning field"
  }
"sentence_field": {
  "note type name A"`: `"note A english field"
  "note type name B"`: `"note B english field"
}
...etc
```

You must define

- for cleaning/generating meanings:
  1. `meaning_field`
  2. `word_field`
  3. `sentence_field`
- for translating sentences
  1. `sentence_field`
  2. `translated_sentence_field`
