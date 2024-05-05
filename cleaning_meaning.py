def get_single_meaning_from_chatGPT(vocab, sentence, dict_entry):
    return_field = "cleaned_meaning"
    prompt = """word: {vocab}
    sentence: {sentence}
    dictionary_entry_for_word: {dict_entry}

    The dictionary entry may contain multiple meanings for the word.
    Extract the one meaning matching the usage of the word in the sentence.
    Omit any example sentences the matching meaning included.
    If the meaning is more than four sentences long, shorten it to explain the basics only.
    Otherwise, keep an already short meaning as-is.
    In case there is only meaning, return that.

    Return the extracted meaning in a JSON string as the value of the key \"{return_field}\".
    """.format(
        vocab=vocab, sentence=sentence, dict_entry=dict_entry, return_field=return_field
    )
    result = get_response_from_chatGPT(prompt, return_field)
    if result is None:
        # Return original dict_entry unchanged if the cleaning failed
        return dict_entry
    return result
