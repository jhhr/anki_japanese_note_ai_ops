import json
from typing import Union
from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = False


def get_extracted_words_from_model(
        sentence: str,
        config: dict[str, str]
    ) -> Union[str, None]:
    prompt = f"""Below is a Japanese sentence that contains furigana in brackets after kanji words. Your task is to examine each word and phrase in the sentence, categorize each into either nouns, proper nouns, verbs, compound verbs, adjectives, adverbs, adjectivals, particles (including copula), pronouns, suffixes, idiomatic phrases and 4-kanji idioms. You will convert convert inflected words into their dictionary forms.

More details on the categorization
- Compound words, phrases or aphorisms should be listed as well, along with their components. That is, if "XYZ" is such a sequence and "XY" and "Z" are valid words, include "XYZ", "XY" and "Z" in the result.
- This applies to compound verbs as well - include both the compound verb and its component verbs when they exist as separate valid words. For example: 飲[の]み 込[こ]まれた --> 飲み込む, 飲む and 込む
- Don't list verbs いる or される when they occur as auxiliary verbs in verbs inflected forms, e.g. 食べている, 行かせる.
- する verbs are to be listed as nouns and the する verb ignored.
- Only list proper nouns a single time, ignoring their component nouns.
- List 4-kanji idioms only once as well, disregarding 2-kanji words that they may contain.
- Take note of words withing <gikun> tags. The kanji used for words wrapped in <gikun> tags are to be ignored and the word listed in hiragana. For example: k><gikun> 不埒[だら]し</gikun></k><k> 無[な]い</k> should be processed as if it was だらし 無[な]い
- Otherwise ignore any HTML that may be in the text, leaving any HTML out of the word lists.


Example sentence 1: 私[わたし]も<b> 連[つ]れて 行[い]って</b><k> 下[くだ]さい</k>。
Example results 1:
{{
  "nouns": [],
  "proper_nouns": [],
  "verbs": [ ["連れる","つれる"], ["行く","いく"]],
  "compound_verbs": [ ["連れて行く","つれていく"]],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [ ["も","も"]],
  "pronouns", [ ["私","わたし"]],
  "suffixes": [],
  "phrases": [ ["下さい","ください"]],
  "four_kanji_idioms": []
}}


Example sentence 2: <k> 彼[あ]の</k> 飛行機[ひこうき]は<b> 間[ま]も<k> 無[な]く</k></b> 着陸[ちゃくりく]<k> 為[し]ます</k>ね。
Example results 2:
{{
  "nouns": [ ["飛行機","ひこうき"], ["各陸","ちゃくりく"], ["間","ま"] ],
  "proper_nouns": [],
  "verbs": [],
  "compound_verbs": [],
  "adjectives": [ ["無い","ない"]],
  "adverbs": [],
  "adjectivals": [ ["彼の","あの"]],
  "particles": [ ["は","は"], ["ね","ね"]],
  "pronouns", [],
  "suffixes": [],
  "phrases": [ ["間も無く","まもなく"] ],
  "four_kanji_idioms": []
}}
Example sentence 3:  <k> 此[こ]れ</k>は 正[まさ]に 天高[てんたか]く 馬肥[うまこ]ゆる 秋[あき]と 言[い]った<k> 物[も]ん</k>だな。
Example result 3:
{{ 
  "nouns": [ ["天","てん"], ["馬","うま"], ["秋","あき"], ["物","もの"] ],
  "proper_nouns": [],
  "verbs": [ "肥える","こえる", ["言う","いう"] ],
  "compound_verbs": [],
  "adjectives": [ ["高い","たかい"] ],
  "adjectivals": [],
  "adverbs": [ ["正に","まさに"] ],
  "particles": [ ["に","に"], ["と","と"], ["だ","だ"] ],
  "pronouns": [ ["此れ","これ"] ],
  "suffixes": [],
  "phrases": [ ["天高く馬肥ゆる秋", "てんたかくうまこゆるあき"] ],
  "four_kanji_idioms": [],
}}

Example sentence 4: 昭和[しょうわ]10 年[ねん](1935 年[ねん]) 頃[ごろ]から、<b>八紘一宇[はっこういちう]</b><k> 等[など]</k>のスローガンが 掲[かか]げられる<k> 様[よう]に</k><k> 成[な]った</k>。
Example result 4:
{{
  "nouns": [ ["昭和","しょうわ"], ["年","ねん"], ["スローガン","すろーがん"], ["様","よう"]],
  "proper_nouns": [],
  "verbs": [ ["掲げる","かかげる"], ["成る","なる"]],
  "compound_verbs": [],
  "adjectives": [],
  "adjectivals": [],
  "adverbs": [ ["頃","ごろ"]],
  "particles": [ ["から","から"], ["等", "など"], ["の","の"], ["が","が"], ["に","に"]],
  "pronouns", [],
  "suffixes": [],
  "phrases": [],
  "four_kanji_idioms": [ ["八紘一宇","はっこういちう"]]
}}

Example sentence 5: <b> 不甲斐[ふがい]ない</b> 里樹[りしゅ]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>を 阿多[ああでぅお]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>が<k> 諫[いさ]めていた</k>。
Example result 5:
{{
  "nouns": [ ["侍女","じじょ"] ],
  "proper_nouns": [ ["里樹","りしゅ"], ["阿多","ああでぅお"]],
  "verbs": [ ["諫める","いさめる"]],
  "compound_verbs": [],
  "adjectives": [ "不甲斐ない","ふがいない"],
  "adverbs": [],
  "adjectivals": [],
  "particles": [ ["の","の"], ["を","を"],["が","が"] ],
  "pronouns", [],
  "suffixes": [ ["様","さま"], ["達","たち"]],
  "phrases": [],
  "four_kanji_idioms": []
}}


Return only the JSON formatted result containing all properties with at least empty arrays. Values inside the arrays must be arrays of two strings.
 
The sentence to process: {sentence} 
"""
    model = config.get("extract_words_model", "")
    result = get_response(model, prompt)
    if result is None:
        if DEBUG:
            print("Failed to get a response from the API.")
        # If the prompt failed, return nothing
        return None
    return json.dumps(result, ensure_ascii=False, indent=2)


def extract_words_in_note(
    note: Note, config: dict, show_warning: bool = True
) -> bool:
    model = note.note_type()
    if not model:
        if DEBUG:
            print("Missing note type for note", note.id)
        return False
    try:
        word_extraction_sentence_field = get_field_config(config, "word_extraction_sentence_field", model)
        word_list_field = get_field_config(
            config, "word_list_field", model
        )
    except Exception as e:
        print(e)
        return False

    if DEBUG:
        print("word_extraction_sentence_field in note", word_extraction_sentence_field in note)
        print("word_list_field in note", word_list_field in note)
    # Check if the note has the required fields
    if word_extraction_sentence_field in note and word_list_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        sentence = note[word_extraction_sentence_field]
        if DEBUG:
            print("sentence", sentence)
        # Check if the value is non-empty
        if sentence:
            # Call API to get translation
            if DEBUG:
                print("Calling API with sentence")
            word_list_json = get_extracted_words_from_model(sentence, config)
            if DEBUG:
                print("result from API", word_list_json)

            if word_list_json is not None:
                if DEBUG:
                    print("word_list_json", word_list_json)

                # Update the note with the new values
                note[word_list_field] = word_list_json
                return True
            return False
        return False
    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_extract_from_notes_op(col, notes: Sequence[Note], edited_nids: list):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("extract_words_model", "")
    message = "Extracting words"
    op = extract_words_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids, model)


def extract_words_from_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated word lists"
    bulk_op = bulk_extract_from_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
