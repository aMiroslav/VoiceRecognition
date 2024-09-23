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

Antrojoje dalyje gautą tekstą turime apdoroti ir paruošti jo santrauką. tai užduočiai atlikti buvo pasitelkti du modeliai: BART ir T5 (apmokytu lietuviškais duomenimis). Abo modeliai veikė pagal tą patį principą. Vėl buvo panaudotą transformers
biblioteka ir pipeline funkcija. Santraukos generavimui atidaromas ir nuskaitomas prieš tai sugeneruotas tekstinis failas. Nustatomi tam tikri parametrai (maksimalus ir minimalus ilgiai it kt.). Galiausiai santrauka atspausdinama ir išsaugoma tekstinio failo pavidalu. 

Testavimui buvo panaudotas straipsnis iš 15min.lt portalo. Straipsnis įgarsintas, vėliau garso įrašas pateiktas teksto atpažinimo programai. Nuskaitytas tekstas pateiktas output.txt faile. Gautos santraukos pateiktos atitinkamai summary_BART.txt ir summary_T5LT.txt failuose.

Apibendrinant gautą rezultatą galima pasakyti, kad lietuvių kalbai pritaikytas T5 modelis su užduotimi susidorojo geriau, gautas rišlesnis ir sklandesnis rezultatas.
