from faster_whisper import WhisperModel
import time

base_path = "/models/faster-whisper-" 
#base_path = ""   #leave blank to download the models directly

models = ["tiny", "base", "small", "medium", "large-v3"]