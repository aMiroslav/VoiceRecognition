from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

with open("output.txt", "r") as f:
    text_to_summarize = f.read()

summary = summarizer(
    text_to_summarize,
    max_length=150,
    min_length=60,
    do_sample=False,
    num_beams=4,
    length_penalty=2
)

print(summary[0]['summary_text'])

with open("summary_BART.txt", "w") as f:
    f.write(summary[0]["summary_text"])
