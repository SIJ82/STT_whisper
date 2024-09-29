from faster_whisper import WhisperModel
import time
import csv
import pandas as pd
from jiwer import wer, wil, wip, mer
import jiwer

base_path = "/models/faster-whisper-" 
#base_path = ""   #leave blank to download the models directly

models = ["tiny", "base", "small", "medium", "large-v3"]
#models = ["tiny", "base", "small"]

dataset_path = "/models/it/"
dataset = pd.read_csv(dataset_path+"validated.tsv", sep="\t")
dataset=dataset.drop(columns=["client_id", "sentence_id", "sentence_domain", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"])
dataset = dataset.head(10)

transforms = jiwer.Compose(
    [
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def errors(data:dict,text:str,  ref:str):
    data["wer"] = wer(ref, text, transforms, transforms)
    data["wil"] = wil(ref, text, transforms, transforms)
    data["wip"] = wip(ref, text, transforms, transforms)
    data["mer"] = mer(ref, text, transforms, transforms)
    return data

def transcribe(model:WhisperModel, audio_path:str):
    data={"model":"m"}
    start = time.time()
    #print("starting transcription...", dataset_path+"clips/"+audio_path)

    segments, info = model.transcribe(dataset_path+"clips/"+audio_path, beam_size=5, language="it", vad_filter=True, without_timestamps=True)
    text: str = ""
    for segment in segments:
        text+=segment.text

    end:float = time.time() - start

    data["audio"] = audio_path
    data["duration"] = info.duration
    data["duration_after_vad"] = info.duration_after_vad
    data["time"] = end

    return (data, text)



with open('./data/results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, 
        fieldnames=["model","audio", "duration", "duration_after_vad", "time", "wer", "wil", "wip", "mer"])
    writer.writeheader()
    
    for m in models:
        try:
            model = WhisperModel(base_path+m, device="cpu", compute_type="int8")
            print("\n------------------------")
            print(f"Testing model {m}\n")

            data={}
            for i in range(len(dataset)):
                try:
                    print(f"Audio {i}/{len(dataset)}- {dataset.iloc[i]["path"]}")
                    (data,text) = transcribe(model, dataset.iloc[i]["path"])
                except:
                    print(f"Audion n.{i} failed")
                
                data = errors(data, text,  dataset.iloc[i]["sentence"])
                data["model"] = m
                writer.writerow(data)

        except:
            pass
    

