from faster_whisper import WhisperModel
import time

print("hello")

model_size = "tiny"
#model_size = "base"
#model_size = "small"
#model_size = "medium"
#model_size = "large-v3"

print("Loading model")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Model loaded", model_size)


start = time.time()
segments, info = model.transcribe("audio.mp3", beam_size=5, language="it")


print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

print("starting transcibe")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print("End of transcribe :", time.time()-start, "seconds")