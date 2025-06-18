from anki.notes import Note, NoteId
from aqt import mw
from aqt.browser import Browser
from aqt.utils import showWarning
from collections.abc import Sequence

from .base_ops import (
    get_response_from_chat_gpt,
    bulk_notes_op,
    selected_notes_op,
)
from ..utils import get_field_config

DEBUG = True


def get_kanjified_sentence_from_chat_gpt(sentence):
    kanjified_sentence_return_field = "kanjified_sentence"
    hiraganaified_sentence_return_field = "hiraganaified_furikanji_sentence"

    prompt = f"""Below is a sentence in Japanese that includes furigana in brackets after kanji words. Your task is to convert the sentence into a fully kanjified version, where all hiragana and katakana words are replaced with their kanji equivalents"
    
    Carefully examine each and every word written in hiragana or katakana and determine whether it can be written in kanji.
    Every word that can be kanjified must be kanjified!
    Add one space character before the new kanjified word. That is, "これ" into " 此[こ]れ" etc. 
    Keep any HTML tags in the sentence as they are. 
    
    # Examples to illustrate the conversions:
    Example sentence 1: 「これでもちゃんと 皆[みな]さんのことを 考[かんが]えてるつもりなんですよ！」<br>「なんかいよいよお 前[まえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」
    Kanjified example 1: 「  此[こ]れでもちゃんと 皆[みな]さんの 事[こと]を 考[かんが]えてる 積[つも]りなんですよ！」<br>「なんか 愈々[いよいよ] 御前[おまえ]も 完全[かんぜん]に 内政[ないせい] 官[かん]だな。」
    
    Example sentence 2: ナツキ 殿[どの]ですよね？ 兄[あに]から 聞[き]いています。その 数々[かずかず]のうわさもかねがね。
    Kanjified example 2: ナツキ 殿[どの]です よね？ 兄[あに]から 聞[き]いています。 其[そ]の 数々[かずかず]の 噂[うわさ]も 兼々[かねがね]。
    
    Example sentence 3: ズボンのすそをまくって 作業[さぎょう]をした。
    Kanjified example 3: 洋袴[ズボン]の 裾[すそ]を 捲[まく]って 作業[さぎょう]をした。
    
    Example sentence 4: おかげで　この 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]に ぼったくられたぜ。
    Kanjified example 4: 御蔭[おかげ]で 此[この] 守銭奴[しゅせんど]の 性[しょう] 悪天使[あくてんし]にぼっ 手繰[たく]られたぜ。
    
    Example sentence 5: 毎日[まいにち]一キロ 以上[いじょう] 水泳[すいえい]をしてきただけのことはあって、 彼[かれ]は九十 歳[さい]の 今[いま]もかくしゃくとしている。
    Kanjified example 5: 毎日[まいにち] 一[いち]キロ 以上[いじょう] 水泳[すいえい]をして 来[き]ただけの 事[こと]は 有[あ]って、 彼[かれ]は 九十[きゅうじゅう] 歳[さい]の 今[いま]も 矍鑠[かくしゃく]としている。
    
    Example sentence 6: 俺はちっぽけでどうしようもないろくでなしですよ。
    Kanjified example 6: 俺[おれ]は 小[ち]っぽけで 如何[どう] 仕様[しよう]も 無[な]い 碌[ろく]で 無[な]し ですよ。
    
    Example sentence 7: しのごの 言[い]わずさっさと 手[て]を 貸[か]せ。
    Kanjified example 7: 四[し]の 五[ご]の 言[い]わずさっさち 手[て]を 貸[か]せ。
    
    Return a JSON string with the following key-value pairs: 
     "{kanjified_sentence_return_field}": The fully kanjified sentence.
     
    The sentence to process: {sentence} 
    """
    result = get_response_from_chat_gpt(prompt)
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
    note: Note, config: dict, show_warning: bool = True
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
            sentence = sentence.replace("<b>", "").replace("</b>", "")
            if DEBUG:
                print("cleaned sentence", sentence)
            # Call API to get translation
            result = get_kanjified_sentence_from_chat_gpt(sentence)
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


def bulk_kanjify_notes_op(col, notes: Sequence[Note], edited_nids: list):
    config = mw.addonManager.getConfig(__name__)
    message = "Kanjifying sentences"
    op = kanjify_sentence_in_note
    return bulk_notes_op(message, config, op, col, notes, edited_nids)


def kanjify_selected_notes(nids: Sequence[NoteId], parent: Browser):
    done_text = "Updated kanjified sentences"
    bulk_op = bulk_kanjify_notes_op
    return selected_notes_op(done_text, bulk_op, nids, parent)
