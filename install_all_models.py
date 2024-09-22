#questo script serve a scaricare e installare tutti i modelli in modo che siano pronti per l'utilizzo
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu", compute_type="float32")
model = WhisperModel("base", device="cpu", compute_type="float32")
model = WhisperModel("small", device="cpu", compute_type="float32")
model = WhisperModel("medium", device="cpu", compute_type="float32")
model = WhisperModel("large-v3", device="cpu", compute_type="float32")
print("Installation completed")