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

ref = [
    "Allora, questo è un test di registrazione, sto provando a parlare, ora seguiranno un paio di comandi, prima piano, poi veloce. Alexa, avvia una conversazione rapida. Alexa, avvia una conversazione rapida. Alexa, avvia una conversazione rapida. Alza il volume, abbassa il volume. Alza il volume, abbassa il volume. Alza il volume, abbassa il volume.",
    "Questa è un'altra prova, barbabietola da zucchero. Vorrei fare un ordine. Ok, ciao"
    ]

dataset :list[dict[str, any]]= []


for m in models:
    try:
        model = WhisperModel(base_path+m, device="cpu", compute_type="int8")
        print("\n------------------------")
        print(f"Testing model {m}\n")

        n_a=0
        for a in audios:
            start = time.time()
            segments, info = model.transcribe(a, beam_size=5, language="it", vad_filter=True, without_timestamps=True) 
            print(f"\tTranscribing : {a} - {info.duration}s")
            
            text = ""
            for segment in segments:
                text+=segment.text
            
            e_wer = wer(ref[n_a], text, transforms, transforms)
            e_wil = wil(ref[n_a], text, transforms, transforms)
            e_wip = wip(ref[n_a], text, transforms, transforms)
            e_mer = mer(ref[n_a], text, transforms, transforms)

            end = time.time() - start
            #print(text)
            print(f"\tDone in {end} - wer:{e_wer}\n")

            data = {
                "model" : m,
                "audio" : a,
                "duration" : info.duration,
                "duration_after_vad" : info.duration_after_vad,
                "time" : end,
                "wer" : e_wer,
                "wil" : e_wil,
                "wip" : e_wip,
                "mer" : e_mer
            }

            dataset.append(data)
            n_a += 1
    except:
        dataset.append(  { "model": m, "audio": "aborted" }  )



with open('./data/results.csv', 'w', newline='') as csvfile:
    fieldnames = [key for key in dataset[0]]
    print(fieldnames)
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)
    

