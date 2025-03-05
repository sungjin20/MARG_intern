from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text.characters import Graphemes
from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer

KOREAN_PHONEMES = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ"
korean_punctuations = ".,!?~ "

gp = Graphemes(
    pad="_",
    eos="&",
    bos="*",
    blank=None,
    characters=KOREAN_PHONEMES,
    punctuations=korean_punctuations,
    is_unique=True,
    is_sorted=True
)

print(gp.id_to_char(0))
print(gp.id_to_char(1))
print(gp.id_to_char(2))
print(len(gp.vocab))

phonemizer = KO_KR_Phonemizer()

tokenizer = TTSTokenizer(characters=gp, phonemizer=phonemizer)

text = "저를 따라오세요."
print(text)
text = phonemizer.phonemize(text)
print(text)
ids = tokenizer.text_to_ids(text)
print(ids)
text_hat = tokenizer.ids_to_text(ids)
print(text_hat)