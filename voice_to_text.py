from pydub import AudioSegment
from transformers import pipeline

model_name = "openai/whisper-large-v3"
asr = pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30, )

mp3_file = "ITStraipsnis.mp3"
audio = AudioSegment.from_mp3(mp3_file)
wav_file = "output.wav"
audio.export(wav_file, format="wav")

audio_file = "output.wav"

result = asr(audio_file)
print(result["text"])

with open("output.txt", "w") as f:
    f.write(result["text"])