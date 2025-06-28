from typing import Union
from anki.notes import Note, NoteId
from anki.collection import Collection
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


def get_kanjified_sentence_from_model(
        config: dict[str, str],
        sentence: str,
    ) -> Union[list[str], None]:
    kanjified_sentence_return_field = "kanjified_sentence"

    prompt = f"""Below is a sentence in Japanese that includes furigana in brackets after kanji words. Your task is to convert the sentence into a fully kanjified version, where all hiragana and katakana words are replaced with their kanji equivalents"

Carefully examine each and every word written in hiragana or katakana and determine whether it can be written in kanji.
Common words that always written in hiragana should be kanjified. Examples of common words in kanjified form (not to be considered an exhaustive list of what to kanjify!): 此[こ]れ, 無[な]い,出来[でき]る, 御前[おまえ], 愈々[いよいよ]
The result should be text where only foreign words or names in katakana, particles and words for which no kanjified form exists or should be used (see edge cases below), are left in hiragana or katakana.

Content modification rules:
- To signify the changes made to the text, wrap each kanjified word in <k> tags with a space before the kanji: "これ" becomes "<k> 此[こ]れ</k>". Include the okurigana of verbs and adjectives within the <k> tags, for example "まわってた" becomes "<k> 回[まわ]ってた</k>".
- <k> tags should wrap a contiguous sequence of kanji conversions, stopping on a word that was already in kanji. For example "とうもろこし" becomes "<k> 玉蜀黍[とうもろこし]</k>" and "タンパク 質[しつ]" becomes "<k> 蛋白[たんぱく]</k> 質[しつ]".
- Keep any existing HTML tags in the sentence as they are. Added <k> tags should be placed inside existing tags, leaving them outermost so that multiple <k> tags can, if necessary, be within the existing tag. For example "<b>これ見よがしに</b>" becomes "<b><k> 此[こ]れ</k> 見[み]よがしに</b>".

Policy on edge cases (not be considered an exhaustive list, but can be used as a guideline for cases not listed here):

Do not kanjify:
- あげる when meant as "to give". あげる is written in hiragana specifically to indicate that it is different from 揚げる, 上げる or 挙げる
- ない auxiliary in negated verbs. ない is not considered 無い written in hiragana, but rather a grammatical auxiliary.
- いる and いく as an auxiliary verb in verb conjugation, for example 書いている, 食べていく, etc. These auxiliaries should be considered a part of a verb's conjugation and, thus, its okurigana, and so enclosed within the <k> tag of a kanjified verb.
- なんか when it is acting more as a particle or filler and could removed without (significant) loss of meaning, for example 今日なんか暑いですね
- もう used purely as exclamatory particle, for example もう！ or もう、やめてよ！
- もっと as it is not truly component in もっとも which does have a kanjified form as 最も or 尤も
- そんな, こんな, あんな, どんな
Do kanjify:
- ない when used as a standalone word, including conjugated forms like なかった, なくて. 無い is even currently used in modern text, but is simply often written in hiragana.
- いる and いく when is used as a standalone verb, for example 彼は家にいる, あっちにいく
- する, even in suru-verbs. Historically, suru-verbs were written with 為る so this is a valid kanjification.
― semantically equivalance but no actual historical usage of the reading: kanjify but wrap with additional <gikun> tags within the <k> tags.
  - Examples ケチ in ケチがつく means "flaw/blemish" and matches 疵 in meaning but 疵 has never been read as けち; "ケチがつく" --> converts "<k><gikun> 疵[けち]</gikun></k>が<k> 付[つ]く</k>"
  - すっかり means ことごとく but 悉 has no historical usage of the reading すっかり; "すっかり" --> converts "<k><gikun> 悉[すっかり]</gikun></k>"
- なんか when it is clearly a contraction of なにか its removal would change the questioning meaning of a phrase, for example なんか食べたい
- やすい as used in verbs like 食べやすい, 書きやすい, etc. This is a a form of 易い
- the honorific prefix お
- the adverb もう as semantically equivalent to 最早, when the meaning is "already/now/no longer" or as semantically equivalent to 復, when the meaning is "again/once more". These are both <gikun> cases.
- よう as 様[よう] in all its forms, ような, ように, ようだ, etc.

# Examples to illustrate the conversions:
Example sentence 1: 「これでもちゃんと 皆[みな]さんのことを 考[かんが]えてるつもりなんですよ！」<br>「なんかいよいよお 前[まえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」
Kanjified example 1: 「 <k> 此[こ]れ</k>でもちゃんと 皆[みな]さんの<k> 事[こと]</k>を 考[かんが]えてる<k> 積[つも]り</k>なんですよ！」<br>「なんか<k> 愈々[いよいよ]</k><k> 御[お]</k> 前[まえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」

Example sentence 2: ナツキ 殿[どの]ですよね？ 兄[あに]から 聞[き]いています。その 数々[かずかず]のうわさも<b>かねがね</b>。
Kanjified example 2: ナツキ 殿[どの]ですよね？ 兄[あに]から 聞[き]いています。<k> 其[そ]</k>の 数々[かずかず]の<k> 噂[うわさ]</k>も<b><k> 兼々[かねがね]</k></b>。

Example sentence 3: ズボンのすそを<b>まくって</b> 作業[さぎょう]をした。
Kanjified example 3: <k> 洋袴[ズボン]</k>の<k> 裾[すそ]</k>を<b><k> 捲[まく]って</k></b> 作業[さぎょう]を<k> 為[し]た</k>。

Example sentence 4: おかげで　この 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]に ぼったくられたぜ。
Kanjified example 4: <k> 御蔭[おかげ]</k>で<k> 此[この]</k> 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]にぼっ<k> 手繰[たく]られた</k>ぜ。

Example sentence 5: 毎日[まいにち]一キロ 以上[いじょう] 水泳[すいえい]をしてきただけのことはあって、 彼[かれ]は九十 歳[さい]の 今[いま]もかくしゃくとしている。
Kanjified example 5: 毎日[まいにち] 一[いち]キロ 以上[いじょう] 水泳[すいえい]を<k> 為[し]て</k><k> 来[き]た</k>だけの<k> 事[こと]</k>は<k> 有[あ]って</k>、 彼[かれ]は 九十[きゅうじゅう] 歳[さい]の 今[いま]も<k> 矍鑠[かくしゃく]</k>と<k> 為[し]ている</k>。

Example sentence 6: 俺はちっぽけでどうしようもないろくでなしですよ。
Kanjified example 6: 俺[おれ]は<k> 小[ち]</k>っぽけで<k> 如何[どう]</k><k> 仕様[しよう]</k>も<k> 無[な]い</k><k> 碌[ろく]</k>で<k> 無[な]し</k> ですよ。

Example sentence 7: <b>しのごの</b> 言[い]わずさっさと 手[て]を 貸[か]せ。
Kanjified example 7: <b><k> 四[し]</k>の<k> 五[ご]</k>の</b> 言[い]わずさっさと 手[て]を 貸[か]せ。

Example sentence 8: <ul><li>いろいろなもので 遊[あそ]んでいるうちに 家[いえ]の 中[なか]は<span style="font-weight: bold;">めちゃくちゃ</span>になっていた。ふと 気[き]が 付[つ]けば 時計[とけい]は11 時[じ]を 指[さ]していた。</li></ul><br>
Kanjified example 8: <ul><li><k> 色々[いろいろ]</k>な<k> 物[もの]</k>で 遊[あそ]んでいる<k> 内[うち]に</k> 家[いえ]の 中[なか]は<span style="font-weight: bold;"><k> 滅茶苦茶[めちゃくちゃ]</k></span>に<k> 成[な]っていた</k>。<k> 不図[ふと]</k> 気[き]が 付[つ]けば 時計[とけい]は11 時[じ]を 指[さ]していた。</li></ul><br>

Example sentence 9: しかもけっして 食欲[しょくよく]がないからではなかったのだ。また、 彼[かれ]の 口[くち]にもっと 合[あ]うような 別[べつ]な 食[た]べものをもってくるのだろうか。 妹[いもうと]が 自分[じぶん]でそうしてくれないだろうか。
Kanjified example 9: <k> 然[しか]も</k><k> 決[け]っして</k> 食欲[しょくよく]が<k> 無[な]い</k>からでは<k> 無[な]かった</k>のだ。<k> 又[また]</k>、 彼[かれ]の 口[くち]にもっと<k> 合[あ]う</k><k> 様[よう]な</k><k> 別[べつ]</k>な<k> 食[た]べ物[もの]</k>を<k> 持[も]って</k><k> 来[く]る</k>のだろうか。 妹[いもうと]が 自分[じぶん]で<k> 然[そ]う</k><k> 為[し]て</k><k> 呉[く]れない</k>だろうか。

Return a JSON string with the following key-value pairs: 
 "{kanjified_sentence_return_field}": The fully kanjified sentence.
 
The sentence to process: {sentence} 
"""
    model = config.get("kanjify_sentence_model", "")
    result = get_response(model, prompt)
    if result is None:
        if DEBUG:
            print("Failed to get a response from the API.")
        # If the prompt failed, return nothing
        return None
    try:
        return [result[kanjified_sentence_return_field]]
    except KeyError:
        if DEBUG:
            print(f"Key '{kanjified_sentence_return_field}' not found in the result.")
        return None


def kanjify_sentence_in_note(
    config: dict[str, str],
    note: Note,
    notes_to_add_dict: dict[str, list[Note]] = {},
    ) -> bool:
    model = note.note_type()
    if not model:
        if DEBUG:
            print("Missing note type for note", note.id)
        return False
    try:
        furigana_sentence_field = get_field_config(config, "furigana_sentence_field", model)
        kanjified_sentence_field = get_field_config(
            config, "kanjified_sentence_field", model
        )
    except Exception as e:
        print(e)
        return False

    if DEBUG:
        print("furigana_sentence_field in note", furigana_sentence_field in note)
        print("kanjified_sentence_field in note", kanjified_sentence_field in note)
    # Check if the note has the required fields
    if furigana_sentence_field in note and kanjified_sentence_field in note:
        if DEBUG:
            print("note has fields")
        # Get the values from fields
        sentence = note[furigana_sentence_field]
        if DEBUG:
            print("sentence", sentence)
        # Check if the value is non-empty
        if sentence:
            # Clean any <b> tags from the sentence
            # sentence = sentence.replace("<b>", "").replace("</b>", "")
            if DEBUG:
                print("cleaned sentence", sentence)
            # Call API to get translation
            result = get_kanjified_sentence_from_model(config, sentence)
            if DEBUG:
                print("result from API", result)

            if result is not None:
                [kanjified_sentence] = result
                if DEBUG:
                    print("kanjified_sentence", kanjified_sentence)

                # Update the note with the new values
                note[kanjified_sentence_field] = kanjified_sentence
                return True
            return False
        return False
    elif DEBUG:
        print("note is missing fields")
    return False


def bulk_kanjify_notes_op(
        col: Collection,
        notes: Sequence[Note],
        edited_nids: list[NoteId],
        notes_to_add_dict: dict[str, list[Note]] = {},
    ):
    config = mw.addonManager.getConfig(__name__)
    if not config:
        showWarning("Missing addon configuration")
        return
    model = config.get("kanjify_sentence_model", "")
    message = "Kanjifying sentences"
    op = kanjify_sentence_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids, notes_to_add_dict, model)


def kanjify_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated kanjified sentences"
    bulk_op = bulk_kanjify_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
