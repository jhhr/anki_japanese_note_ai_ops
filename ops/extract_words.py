import json
import re
from collections.abc import Sequence
from typing import Union
from anki.notes import Note, NoteId
from anki.collection import Collection
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning

from .base_ops import (
    get_response,
    bulk_notes_op,
    selected_notes_op,
    AsyncTaskProgressUpdater,
)
from ..utils import get_field_config
from ..configuration import (
    raw_one_meaning_word_type,
    raw_multi_meaning_word_type,
    matched_word_type,
)

DEBUG = False


def word_tuple_sort_key(
    word_tuple: Sequence,
) -> tuple[
    Union[str, None], Union[str, None], Union[str, None], Union[int, None], Union[int, None]
]:
    """
    Sort key for word tuples. This is used to sort the words in the word list.
    The sort order is:
    1. word
    2. reading
    3. check sort_word and note_id, if they are present and sort them before those that lack them
    4. meaning_number, if present, should be sorted after the above
    """
    word, reading, meaning_number, sort_word, note_id = None, None, None, None, None
    if len(word_tuple) == 2:
        word, reading = word_tuple
    elif len(word_tuple) == 3:
        word, reading, meaning_number = word_tuple
    elif len(word_tuple) == 4:
        word, reading, sort_word, note_id = word_tuple
    # at least word and reading must be present
    if not word or not reading:
        # Some invalid data, sort last
        return (None, None, None, None, None)
    if sort_word is None:
        sort_word = ""
    if note_id is None:
        note_id = 0
    else:
        # If note_id is present, it should be a number, so convert it to int
        try:
            note_id = int(str(note_id).strip())
        except ValueError:
            if DEBUG:
                print(f"Invalid note_id {note_id} for word {word}, setting to 0")
            note_id = 0
    if meaning_number is None:
        meaning_number = 0
    else:
        # If meaning_number is present, it should be a number, so convert it to int
        try:
            meaning_number = int(str(meaning_number).strip())
        except ValueError:
            if DEBUG:
                print(f"Invalid meaning_number {meaning_number} for word {word}, setting to 0")
            meaning_number = 0
    # Return a tuple that can be used for sorting
    return (word, reading, sort_word, note_id, meaning_number)


def compared_word_lists(
    cur_word_list: list[tuple],
    new_word_list: list[tuple],
) -> list[Union[raw_one_meaning_word_type, raw_multi_meaning_word_type, matched_word_type]]:
    """
    Compare two word lists and return a list of words that combine both lists, keeping all previous
    words and only adding new words that are not already in the current list.
    """
    if not cur_word_list:
        return new_word_list
    if not new_word_list:
        return cur_word_list
    # Make sets of the whole tuples in the lists
    cur_set = set(tuple(word) for word in cur_word_list)
    new_set = set(tuple(word) for word in new_word_list)
    # Get all tuples that are only in the new set
    added_set = new_set - cur_set
    # check each new tuple for validity
    added_list: list[Union[raw_one_meaning_word_type, raw_multi_meaning_word_type]] = []
    for word_tuple in added_set:
        word, reading, meaning_number, sort_word, note_id = None, None, None, None, None

        if len(word_tuple) == 2:
            word, reading = word_tuple
        elif len(word_tuple) == 3:
            word, reading, meaning_number = word_tuple
        elif len(word_tuple) == 4:
            word, reading, sort_word, note_id = word_tuple
        else:
            if DEBUG:
                print(f"Word tuple with invalid length {word_tuple} in added_set, skipping")
            continue

        # at least word and reading must be present
        if word and reading:
            # sort_word and note_id should not be present in the added_set, any new words that had
            # should've matched an existing word in the current set, so a new word that has
            # them is invalid and was probably hallucinated by the model. If this happens a lot
            # the prompt's instructions should be adjusted as the model should be returning
            # existing words sort sort_word and note_id as-is
            if sort_word is not None or note_id is not None:
                if DEBUG:
                    print(
                        f"Invalid new word tuple with sort_word/note_id preset {word_tuple} in"
                        " added_set, skipping"
                    )
                continue
            elif meaning_number is not None:
                added_list.append((word, reading, meaning_number))
            else:
                # if meaning_number is not present, this is allowed
                added_list.append((word, reading))
        else:
            if DEBUG:
                print(
                    f"Invalid new word tuple with missing word/reading {word_tuple} in added_set,"
                    " skipping"
                )
            continue
    # Return the sorted combined list of current words and new words
    # Ensure all elements are tuples
    combined_list = [tuple(w) for w in cur_word_list] + [tuple(w) for w in added_list]
    combined_list.sort(key=word_tuple_sort_key)
    return combined_list


def word_lists_str_format(
    word_lists: dict,
) -> Union[str, None]:
    """
    Convert the word list dict into the same format used in the prompt
    """
    if not word_lists:
        return None
    return (
        "{\n"
        + ",\n".join(
            [f'  "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in word_lists.items()]
        )
        + "\n}"
    )


def get_extracted_words_from_model(
    sentence: str, current_lists: Union[str, None], config: dict[str, str]
) -> Union[dict, None]:
    current_lists_addition = ""
    if current_lists:
        # If current_lists is not empty, add the portion to the instructions to examine it
        current_lists_addition = f"""The sentence has been processed before and the current word lists are shown below. Your task should be to consider only whether more words should be added to any lists and whether any words may have been categorized differently from the current instructions below - the instructions when you processed these last time may have been different.

Instructions on modifying the current lists:
- Linked words are word arrays that have 4 elements: 3 strings and one positive or negative long integer may only be moved to another list, not modified and definitely not removed.
  - The most common case is that the linked multi-meaning word's 3rd string is the same as the 1st string. For example ["上る","のぼる",上る", 1378555077520]
  - When there are more than one multi-meaning words in this form will have the two first strings be identical but the 3rd string and final integer will differ. For example: ["控える","おさえる","控える (m1)", 1378555133370] and [控える","おさえる","控える (m2)", 1616058016685]
- Generally you should only add more words, not remove any. Removal can be considered for compound verbs or expressions that are sufficiently accounted for by their individual components that are already listed in other word categories. Or, if there appears to be too many multi-meaning words, when one less meaning may suffice to account for each usage of the word. However, if the multi-meaning words are already linked, they should not be touched.
- If there is a case of a pair or more of homophone+homograph words occurring in the sentence but the current list does not list the word enough times, the to-be-added additional multi-meaning words' meaning index number depends on whether the current words are linked or not.
  a. If there is only a single 2-element non-linked word, you should modify it to add the meaning index number to it, starting from 1, and add new word(s) with meanings number incrementing from there. For example, there being ["上がる","あがる"] only but 2 meanings of 上がる used in the sentence --> the result would contain ["上がる","あがる", 1] and ["上がる","あがる", 2]
  b. If there is more than one 2-element word - which should contain meaning numbers already - continue adding more words with meaning numbers beginning from the highest index + 1 of the current words. For example, ["上がる","あがる", 1] and ["上がる","あがる", 2] being present but 3 meanings of 上がる are used in the sentence --> add one more, so ["上がる","あがる", 3]
  c. If there are any 4-element linked words, the meaning numbers you use for the new word or words you add do not need count the linked word(s). For example, there is ["当て","あて","当て (m1)", 1744043020707] and 2 meanings of 当て used --> only add ["当て","あて", 1]. If there is ["当て","あて","当て (m5)", 1744043020711] and ["当て","あて", 1] but 3 meanings of 当て used --> only add ["当て","あて", 2]

Current word lists: {current_lists}

"""
    prompt = f"""Below is a Japanese sentence that contains furigana in brackets after kanji words. Your task is to examine each word and phrase in the sentence, categorize each into either nouns, proper nouns, numbers, counters, verbs, compound verbs, adjectives, adverbs, adjectivals, particles (and copula), conjunctions, pronouns, suffixes, prefixes, idiomatic expressions or common phrases and 4-kanji idioms (yojijukugo). You will convert convert inflected words into their dictionary forms. When two or more words are both homophones and homographs a number is added to indicate that they are different meanings.

More details on the categorization
- Compound words, expressions or aphorisms should be listed as well, along with their components. That is, if "XYZ" is such a sequence and "XY" and "Z" are valid words, include "XYZ", "XY" and "Z" in the result.
- This applies to compound verbs as well - include both the compound verb and its component verbs when they exist as separate valid words. For example: 飲[の]み 込[こ]まれた --> 飲み込む, 飲む and 込む
- However, don't list compound words that do not form a significantly different meaning from their components. For example, from 委員会議長[いいんかいぎちょう] the words to list would be just 委員会 ("committee") and 議長 ("chairman") as the compound is simply "committee chairman" and thus perfectly described by the two components.
- Don't list verbs いる or される when they occur as auxiliary verbs in verbs inflected forms, e.g. 食べている, 行かせる.
- する verbs are to be listed as nouns and the する verb ignored.
- Avoid listing words ending in particles or copula, as this would create many variants of the same word. The exceptions would be when the copula/particle-added form is overwhelmingly more common than the word being used without the particle/copula. For example, with the particle に, the adverb 共に is overhelmingly more common over the plain noun form 共, so whenever 共に occurs, 共に and not 共 should be listed. Only, if 共 were to occur alone (not as part of a compound word), it should be listed.
- Don't list verbs in たい, たくない, せる or other non-base forms, except when such a form has a special meaning. Examples of special meanings, 食えない "shrewd" vs literal "cannot eat", 唸らせる "to impress" vs literal "to make someone groan". Example of non-special meaning: 齧らせる is simply "to make someone bite"
- Don't list adjectives in さ form, list them in their い-form. Avoid listing adjectives in く-form as well, excpect when they have a meaning that isn't merely adverbial; for example, 大きく has the meaning "on a grand scale / extensively"
- Don't list nouns which may take the genitive case の with the particle, list them in their plain form. For example, 上の should be listed as just 上.
- Only list proper nouns a single time, ignoring their component nouns.
- List 4-kanji idioms only once as well, disregarding 2-kanji words that they may contain.
- Take note of words withing <gikun> tags. The kanji used for words wrapped in <gikun> tags are to be ignored and the word listed in hiragana. For example: <k><gikun> 不埒[だら]し</gikun></k><k> 無[な]い</k> should be processed as if it was だらし 無[な]い
- Otherwise ignore any HTML that may be in the text, leaving any HTML out of the word lists.
- A word occuring twice or more with the same kanji form and reading needs to considered for homonymity. If it is a used in the same meaning, the word should be listed just once. If the meanings differ, the word listed once for each different meaning, with a 1-based index number included to differentiate them. For example, 行く as "physically move to a place" vs "participate in an activity" vs "reach a point (in an activity, not physical place)".
- Note, homonym listing of individual can only be done, if a word actually occurs more than once.
- Additionally, a word occuring twice with the same meaning but, for some reason in kanji form and in hiragana, should result in one entry using the kanji form.


Example sentence 1: 私[わたし]も<b> 連[つ]れて 行[い]って</b><k> 下[くだ]さい</k>。
Example results 1:
{{
  "nouns": [],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["連れる","つれる"],["行く","いく"]],
  "compound_verbs": [["連れて行く","つれていく"]],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["も","も"]],
  "conjunctions": [],
  "pronouns",[["私","わたし"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["下さい","ください"]],
  "yojijukugo": []
}}


Example sentence 2: <k> 彼[あ]の</k> 飛行機[ひこうき]は<b> 間[ま]も<k> 無[な]く</k></b> 着陸[ちゃくりく]<k> 為[し]ます</k>ね。
Example results 2:
{{
  "nouns": [["飛行機","ひこうき"],["各陸","ちゃくりく"],["間","ま"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [],
  "compound_verbs": [],
  "adjectives": [["無い","ない"]],
  "adverbs": [],
  "adjectivals": [["彼の","あの"]],
  "particles": [["は","は"],["ね","ね"]],
  "conjunctions": [],
  "pronouns",[],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["間も無く","まもなく"]],
  "yojijukugo": []
}}
Example sentence 3:  <k> 此[こ]れ</k>は 正[まさ]に 天高[てんたか]く 馬肥[うまこ]ゆる 秋[あき]と 言[い]った<k> 物[も]ん</k>だな。
Example result 3:
{{
  "nouns": [["天","てん"],["馬","うま"],["秋","あき"],["物","もの"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [ "肥える","こえる",["言う","いう"]],
  "compound_verbs": [],
  "adjectives": [["高い","たかい"]],
  "adjectivals": [],
  "adverbs": [["正に","まさに"]],
  "particles": [["に","に"],["と","と"],["だ","だ"]],
  "conjunctions": [],
  "pronouns": [["此れ","これ"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["天高く馬肥ゆる秋", "てんたかくうまこゆるあき"]],
  "yojijukugo": [],
}}

Example sentence 4: 昭和[しょうわ]10 年[ねん](1935 年[ねん]) 頃[ごろ]から、<b>八紘一宇[はっこういちう]</b><k> 等[など]</k>のスローガンが 掲[かか]げられる<k> 様[よう]に</k><k> 成[な]った</k>。
Example result 4:
{{
  "nouns": [["昭和","しょうわ"],["年","ねん"],["スローガン","すろーがん"],["様","よう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["掲げる","かかげる"],["成る","なる"]],
  "compound_verbs": [],
  "adjectives": [],
  "adjectivals": [],
  "adverbs": [["頃","ごろ"]],
  "particles": [["から","から"],["等", "など"],["の","の"],["が","が"],["に","に"]],
  "conjunctions": [],
  "pronouns",[],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": [["八紘一宇","はっこういちう"]]
}}

Example sentence 5: <b> 不甲斐[ふがい]ない</b> 里樹[りしゅ]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>を 阿多[ああでぅお]<k> 様[さま]</k>の 侍女[じじょ]<k> 達[たち]</k>が<k> 諫[いさ]めていた</k>。
Example result 5:
{{
  "nouns": [["侍女","じじょ"]],
  "proper_nouns": [["里樹","りしゅ"],["阿多","ああでぅお"]],
  "numbers": [],
  "counters: [],
  "verbs": [["諫める","いさめる"]],
  "compound_verbs": [],
  "adjectives": [ "不甲斐ない","ふがいない"],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["を","を"],["が","が"]],
  "conjunctions": [],
  "pronouns",[],
  "suffixes": [["様","さま"],["達","たち"]],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 6: <k> 危[あや]うく</k><b>某[ぼう]</b> 業者[ぎょうしゃ]の 甘言[かんげん]に 騙[だま]され、 大損[おおそん]<k> 為[す]る</k><k> 所[ところ]</k>でした。
Example result 6:
{{
  "nouns": [["業者","ぎょうしゃ"],["甘言","かんげん"],["大損","おおそん"],["所","ところ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["騙す","だます"],["する","する"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["に","に"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [["某","ぼう"]]
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 7: <k> 一[ひと]つ</k>の 仕事[しごと]に<b> 於[お]いて</b> 困難[こんなん] 性[せい]の 尺度[しゃくど]で、 仕事[しごと]の 遂行[すいこう] 能力[のうりょく]が、<k> 其[そ]の</k> 頂上[ちょうじょう]を 越[こ]えない 場合[ばあい]は、 何時[いつ]まで 待[ま]っても 解決[かいけつ]<k> 為[し]ない</k>。
Example results 7:
{{
  "nouns": [["仕事","しごと"],["困難","こんなん"],["性","せい"],["尺度","しゃくど"],["遂行","すいこう"],["能力","のうりょく"],["頂上","ちょうじょう"],["場合","ばあい"],["解決","かいけつ"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["越える","こえる"],["待つ","まつ"],["為る","する"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["何時","いつ"]],
  "adjectivals": [],
  "particles": [["に","に"],["で","で"],["が","が"],["を","を"],["は","は"],["まで","まで"]],
  "conjunctions": [["於いて","おいて"]],
  "pronouns": [],
  "suffixes": [["一つ","ひとつ"],["せい","せい"]],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 8: 二<b>隻[せき]</b>の 船[ふね]が 同時[どうじ]に 沈[しず]んだ。
Example result 8:
{{
  "nouns": [["船","ふね"],["同時","どうじ"]],
  "proper_nouns": [],
  "numbers": [["二","に"]],
  "counters: [["隻","せき"]],
  "verbs": [["沈む","しずむ"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["の","の"],["が","が"],["に","に"]],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["二隻","にせき"],["同時に","どうじに"]],
  "yojijukugo": []
}}

Example sentence 9: 最近[さいきん] 行[い]ったデート、どのベースまで 行[い]けた？
Example result 9:
{{
  "nouns": [["デート","でーと"],["ベース","ベーす"]]
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["行く","いく",1],["行く","いく",2]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["最近","最近"]],
  "adjectivals": [["どの","どの"]],
  "particles": [["まで","まで"]],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

Example sentence 10: そう 言[い]えば、 昨日[きのう]なにいった？
Example results 10:
{{
  "nouns": [["昨日","きのう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters: [],
  "verbs": [["言う","いう"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [],
  "pronouns": [["何","なに"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["そう言えば","そういえば"]],
  "yojijukugo": []
}}

Example sentence 11: <k> 例えば[たとえば]</k>、イギリスや 香港[ほんこん]では3 月[がつ]1 日[にち]に 加齢[かれい]<k> 為[さ]れ</k>、 日本[にっぽん]やニュージーランドでは2 月[がつ]28 日[にち]に 加齢[かれい]<k> 為[さ]れる</k>。 日本[にっぽん]でグレゴリオ 暦[れき]を 採用[さいよう]<k> 為[する]</k> 際[さい]、2 月[がつ]29 日[にち]を<b> 閏[うるう] 日[び]</b>と 定[さだ]めた。
Example results 11:
{{
  "nouns": [["加齢","かれい"],["グレゴリオ暦","ぐれごりおれき"],["暦","れき"],["採用","さいよう"],["際","さい"],["閏日","うるうび"],["閏","うるう"],["日","ひ"]],
  "proper_nouns": [["イギリス","いぎりす"],["香港","ほんこん"],["日本","にっぽん"],["ニュージーランド","にゅーじーらんど"]],
  "numbers": [["3","さん"],["1","いち"],["2","に"]],
  "counters": [["月","がつ"],["日","にち"]],
  "verbs": [["定める","さだめる"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["や","や"],["で","で"],["は","は"],["に","に"],["を","を"],["と","と"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["例えば","たとえば"]],
  "yojijukugo": []
}}

Example sentence 12: <b> 鳥肌[とりはだ]</b>が 立[た]つ<k> 位[くらい]</k><k> 痺[しび]れる</k> 演奏[えんそう] 聴[き]かせて<k> 遣[や]っから</k>
Example results 12:
{{
  "nouns": [["鳥肌", "とりはだ", "鳥肌", 1753015449046], ["演奏", "えんそう"], ["位", "くらい"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["立つ", "たつ"], ["痺れる", "しびれる"], ["聴く", "きく"], ["遣る", "やる"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [],
  "adjectivals": [],
  "particles": [["が", "が"], ["から", "から"]],
  "conjunctions": [],
  "pronouns": [],
  "suffixes": [],
  "prefixes": [],
  "expressions": [["鳥肌が立つ", "とりはだがたつ"]],
  "yojijukugo": []
}}

Example sentence 13: 私[わたし]<k> 達[たち]</k>は 今[いま] 生徒会[せいとかい]に<b> 頭[あたま]ごなし</b>に 出展[しゅってん] 拒否[きょひ]<k> 為[さ]れている</k> 状況[じょうきょう]で 。
Example results 13:
{{
  "nouns": [["生徒会", "せいとかい"], ["頭", "あたま"], ["出展", "しゅってん"], ["拒否", "きょひ"], ["状況", "じょうきょう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["今", "いま"]],
  "adjectivals": [],
  "particles": [["は", "は"], ["に", "に"], ["で", "で"]],
  "conjunctions": [],
  "pronouns": [["私", "わたし"]],
  "suffixes": [["達", "たち"]],
  "prefixes": [],
  "expressions": [["頭ごなし", "あたまごなし"]],
  "yojijukugo": []
}}

Example sentence 14: 私[わたし]には 生徒会[せいとかい]の<b> 総意[そうい]</b>を 覆[くつがえ]す<k> 様[よう]な</k> 力[ちから]は<k> 有[あ]りません</k>よ！ 前[まえ]も 言[い]いましたが<k> 先[ま]ずは</k> 証拠[しょうこ]！
Example results 14:
{{
  "nouns": [["生徒会", "せいとかい"], ["総意", "そうい"], ["力", "ちから"], ["前", "まえ"], ["証拠", "しょうこ"], ["様", "よう"]],
  "proper_nouns": [],
  "numbers": [],
  "counters": [],
  "verbs": [["覆す", "くつがえす"], ["有る", "ある"], ["言う", "いう"]],
  "compound_verbs": [],
  "adjectives": [],
  "adverbs": [["先ず", "まず"]],
  "adjectivals": [],
  "particles": [["に", "に"], ["は", "は"], ["の", "の"], ["を", "を"], ["も", "も"], ["よ", "よ"]],
  "conjunctions": [],
  "pronouns": [["私", "わたし"]],
  "suffixes": [],
  "prefixes": [],
  "expressions": [],
  "yojijukugo": []
}}

{current_lists_addition}
Return only the JSON formatted result containing all properties with at least empty arrays. Values inside the arrays must be arrays of two strings, or two strings and one number for multi-meaning words.

The sentence to process: {sentence}
"""
    model = config.get("extract_words_model", "")
    result = get_response(model, prompt)
    if result is None:
        if DEBUG:
            print("Failed to get a response from the API.")
        # If the prompt failed, return nothing
        return None
    return result


def extract_words_in_note(
    config: dict,
    note: Note,
    notes_to_add_dict: dict[str, list[Note]] = {},
) -> bool:
    note_type = note.note_type()
    if not note_type:
        if DEBUG:
            print("Missing note type for note", note.id)
        return False
    try:
        word_extraction_sentence_field = get_field_config(
            config, "word_extraction_sentence_field", note_type
        )
        word_list_field = get_field_config(config, "word_list_field", note_type)
    except Exception as e:
        print(e)
        return False

    ignore_current_word_lists = config.get("ignore_current_word_lists", False)

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
            # Remove text within <i> tags, as it is not relevant for word extraction
            sentence = re.sub(r"<i>.*?</i>", "", sentence, flags=re.DOTALL)
            current_word_lists_raw = note[word_list_field]
            current_word_lists = None
            # Reformat the current word lists to the same format so formatting differences do not
            # cause issues
            if current_word_lists_raw and not ignore_current_word_lists:
                try:
                    current_word_lists = word_lists_str_format(json.loads(current_word_lists_raw))
                except json.JSONDecodeError as e:
                    print("Error decoding JSON from current word lists:", e)
                    return False
                if DEBUG:
                    print("Calling API with sentence")
            else:
                if DEBUG:
                    print("Calling API with sentence and no current word lists")

            word_lists = get_extracted_words_from_model(sentence, current_word_lists, config)
            if DEBUG:
                print("result from API", word_lists)

            if word_lists is not None:
                if DEBUG:
                    print("word_list_json", word_lists)

                # Update the note with the new values
                if not isinstance(word_lists, dict):
                    if DEBUG:
                        print("API response is not a dictionary, resetting to empty")
                    word_lists = {}
                if DEBUG:
                    print("new_list", word_lists)
                # compare and add the new words to the current word list
                try:
                    cur_word_lists = json.loads(note[word_list_field])
                except json.JSONDecodeError:
                    cur_word_lists = {}
                if not isinstance(cur_word_lists, dict):
                    cur_word_lists = {}
                if DEBUG:
                    print("cur_word_lists", cur_word_lists)
                # Compare each matching key in the new and current word lists
                for key in word_lists:
                    if key in cur_word_lists:
                        if DEBUG:
                            print(f"Comparing word lists for key {key}")
                        # Compare the current and new word lists
                        cur_word_list = cur_word_lists[key]
                        new_word_list = word_lists[key]
                        combined_list = compared_word_lists(cur_word_list, new_word_list)
                        if DEBUG:
                            print(f"Combined list for key {key}: {combined_list}")
                        # Update the current word list with the combined list
                        cur_word_lists[key] = combined_list
                    else:
                        if DEBUG:
                            print(f"Adding new key {key} to word lists")
                        # If the key is not present in the current word list, add it
                        # Use the compare func to validate the new list elements
                        cur_word_lists[key] = compared_word_lists([], word_lists[key])
                # Finally, update the field value in the note
                note[word_list_field] = json.dumps(cur_word_lists, ensure_ascii=False, indent=2)
                return True
            return False
        return False
    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_extract_from_notes_op(
    col: Collection,
    notes: Sequence[Note],
    edited_nids: list[NoteId],
    progress_updater: AsyncTaskProgressUpdater,
    notes_to_add_dict: dict[str, list[Note]],
):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("extract_words_model", "")
    message = "Extracting words"
    op = extract_words_in_note
    return bulk_notes_op(
        message,
        config,
        op,
        col,
        notes,
        edited_nids,
        progress_updater,
        notes_to_add_dict,
        model,
    )


def extract_words_from_selected_notes(nids: Sequence[NoteId], parent: Browser):
    progress_updater = AsyncTaskProgressUpdater(title="Async AI op: Extracting words")
    done_text = "Updated word lists"
    bulk_op = bulk_extract_from_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent, progress_updater)
