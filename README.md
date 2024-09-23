# VoiceRecognition

## Projekto užduotis:
1. Audio įrašo pavertimas į tekstą
2. Teksto santraukos generavimas

## Projekto aprašymas:
Projektas yra padalintas į dvi dalis. Pirmoji dalis apima audio įrašo konvertavimą į tekstą, o antroji - gauto teksto santraukos generavimą. 

### 1. Audio įrašo pavertimas į tekstą

Pirmajai užduočiai buvo naudojamas [Whisper modelis](https://huggingface.co/openai/whisper-large-v3), sukurtas automatinio kalbos atpažinimo užduotims. Šis modelis buvo integruotas naudojant „Transformers“ bibliotekos **pipeline** funkciją. Kadangi pateiktas audio įrašas buvo MP3 formato, jis buvo konvertuotas į **WAV** formatą naudojant **pydub** biblioteką. 

Norint, kad modelis geriau apdorotų ilgesnius garso įrašus, jie buvo padalyti į 30 sekundžių segmentus. Gautą rezultatą programa atspausdina ir išsaugo kaip tekstinį failą.

#### Pavyzdinis kodo fragmentas:
```python
from pydub import AudioSegment
from transformers import pipeline

# Whisper modelio įkėlimas
model_name = "openai/whisper-large-v3"
asr = pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30)

# MP3 failo konvertavimas į WAV
mp3_file = "ITStraipsnis.mp3"
audio = AudioSegment.from_mp3(mp3_file)
wav_file = "output.wav"
audio.export(wav_file, format="wav")

# Teksto atpažinimas
audio_file = "output.wav"
result = asr(audio_file)
print(result["text"])

# Teksto išsaugojimas
with open("output.txt", "w") as f:
    f.write(result["text"])
```

### 2. Teksto santraukos generavimas

Užduočiai atlikti buvo pasitelkti du modeliai: [BART] (https://huggingface.co/facebook/bart-large-cnn) ir [T5] (https://huggingface.co/LukasStankevicius/t5-base-lithuanian-news-summaries-175), apmokytu lietuviškais duomenimis. Abo modeliai veikė pagal tą patį principą. Vėl buvo panaudotą „Transformers“ biblioteka ir **pipeline** funkcija. Santraukos generavimui atidaromas ir nuskaitomas prieš tai sugeneruotas tekstinis failas. Nustatomi tam tikri parametrai (maksimalus ir minimalus ilgiai it kt.). Galiausiai santrauka atspausdinama ir išsaugoma tekstinio failo pavidalu. 

#### Pavyzdinis kodo fragmentas:
```python
from transformers import pipeline

# BART modelio įkelimas
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Tekstinio failo nuskaitymas
with open("output.txt", "r") as f:
    text_to_summarize = f.read()

# Santraukos generavimas, nurodant parametrus
summary = summarizer(
    text_to_summarize,
    max_length=150,
    min_length=60,
    do_sample=False,
    num_beams=4,
    length_penalty=2
)
print(summary[0]['summary_text'])

# Santraukos išsaugojimas
with open("summary_BART.txt", "w") as f:
    f.write(summary[0]["summary_text"])
```

### 3. Panaudojimas su pavyzdžiais
Testavimui buvo panaudotas straipsnis iš 15min.lt portalo. Straipsnis įgarsintas, vėliau garso įrašas (failas **ITStraipsnis.mp3**) pateiktas teksto atpažinimo programai. Nuskaitytas tekstas pateiktas **output.txt** faile. Gautos santraukos pateiktos atitinkamai **summary_BART.txt** ir **summary_T5LT.txt** failuose.


### 4. Išvados
Apibendrinant gautą rezultatą galima pasakyti, kad lietuvių kalbai pritaikytas **T5** modelis su užduotimi susidorojo geriau, gautas rišlesnis ir sklandesnis rezultatas. Rezultato tikslumui įtaką daro ir pirminis žingsnis - garso įrašo konvertavimas į tekstą. Atidžiau peržvelgus sugeneruotą tekstą, galima pastabėti, kad yra tam tikrų gramatinių klaidų bei netiksliai atpažintų žodžių. Tai neabejotinai daro įtaką ir galutiniam rezultati.
