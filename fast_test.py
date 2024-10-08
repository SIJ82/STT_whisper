from faster_whisper import WhisperModel
import time
from jiwer import wer, wil, wip, mer

print("hello")

reference = "hello world"
hypothesis = "hello in world"

error = mer(reference, hypothesis)
print("wer", error)

#model_size = "/models/faster-whisper-tiny/"
#model_size = "/models/faster-whisper-base/"
#model_size = "/models/faster-whisper-small/"
model_size = "/models/faster-whisper-medium/"
#model_size = "/models/faster-whisper-large-v3/"

print("Loading model")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Model loaded", model_size)

initial_prompt="Ora diamo dei comandi ad Alexa per il volume"


print("starting transcribe...")
segments, info = model.transcribe("audio2.mp3", beam_size=5, language="it", initial_prompt=initial_prompt)
print("transcribe started...")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

start = time.time()
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print("End of transcribe :", time.time()-start, "seconds")