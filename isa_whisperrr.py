"""
@author: Isa
"""
import os
import random
import re
import time
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from argparse import ArgumentParser
from pathlib import Path
from string import punctuation
from typing import List

import jiwer
import pandas as pd
import tracemalloc
import soundfile as sf
import whisper
from telwoord import cardinal
from tqdm import tqdm

from utils import load_frits_annotations_isa


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir_transcription", type=str, default="data")
    parser.add_argument("--data_dir_audio", type=str, default="data")
    parser.add_argument("--save_dir_transcriptions", type=str, default="results")
    parser.add_argument("--save_dir_whisper", type=str, default="results")
    parser.add_argument("--save_dir_performance", type=str, default="results")
    parser.add_argument("--size", type=str, default="large-v2")
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--samplesize", type=int, default=100)
    #parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args

    
def main():
    #Make variables        
    args = get_args()
    data_dir_transcription = args.data_dir_transcription
    data_dir_audio = args.data_dir_audio
    save_dir_transcriptions = args.save_dir_transcriptions
    save_dir_whisper = args.save_dir_whisper
    save_dir_performance = args.save_dir_performance
    size = args.size
    device = args.device
    samplesize = args.samplesize
    #threshold = args.threshold
    
    path = data_dir_audio
    
    random.seed(10)
    
    file_names = []
    for path, subdirs, files in os.walk(path):
      for name in files:
        file_names.append(name)
    
        
    #sample size
    sample_size = round(len(file_names) * (samplesize/100))
    
    audio_sample = random.sample(file_names, sample_size)


    def audio_duration(audio_file):
      with sf.SoundFile(audio_file) as f:
        duration = len(f) / f.samplerate
      return duration
  
    
    def map_to_dutch_number(word: str) -> str:
      # Map a string representation of an integer to a Dutch written version
      # of the integer, e.g. "100" becomes "honderd"
      return cardinal(int(word), friendly=False)
  
    #Function to parse .ort files
    def parse_ort(data_dir_transcription):
        # Read the .ort file
        with open(data_dir_transcription, 'r', encoding = 'latin-1') as file:
            lines = file.readlines()
        speakers_text = []
        
        for i in range(len(lines)): 
            line = lines[i]
    
            if line.startswith('"IntervalTier"'):
                i+=1
                speaker_name = lines[i].strip()
                i+=1
                start_time = float(lines[i])
                i+=1
                end_time = float(lines[i])
                i+=1
                talk_count = int(lines[i])
                i+=1
                for _ in range(talk_count):
                    start_time = float(lines[i])
                    i+=1
                    end_time = float(lines[i])
                    i+=1
                    text = str(lines[i].strip().replace('"', ""))
                    i+=1
                    speakers_text.append((speaker_name,start_time,end_time,text))
        return speakers_text
    
    # function for getting transcripts
    def get_transcripts(data_dir_transcription):
    
       # audio_sample = get_sample(data_dir_audio, samplesize)
        #print(audio_sample)
        
        # Create an empty dataframe
        df_orig = pd.DataFrame(columns=["Audio File", "Original Transcription"])
    
        # Loop to go through all files in the folder
        for root, subdirs, files in os.walk(data_dir_transcription):
          if "/CGN/" in data_dir_transcription:
            files = [Path(f).stem+".ort" for f in files if Path(f).stem+".wav" in audio_sample]
          else:
            files = [f for f in files if f in audio_sample]
          for name in files:
          
            filepath = os.path.join(root, name)


            # Extract the name of the transcription without the extension
            file_name = os.path.splitext(name)[0]

            # Parse the transcription using your `parse_ort` function
            if "/CGN/" in data_dir_transcription:
                speaker_texts = parse_ort(filepath)  # (speaker_name,start_time,end_time,text)
            else:
                speaker_texts = load_frits_annotations_isa(
                    os.path.join(os.path.dirname(data_dir_transcription), "1mono", filepath),
                    os.path.dirname(data_dir_transcription))

                    
          #   load_frits_annotations, load_frits_annotation
  
            # Remove empty rows
            filtered_speaker_texts = []
            for item in speaker_texts:
              if item[3].strip():  # Check if the text is not empty or whitespace
                filtered_speaker_texts.append(item)
  
            # Create the transcript dictionary
            transcript = []
            for item in filtered_speaker_texts:
              transcript.append({
                  "speaker": item[0],
                  "start": item[1],
                  "end": item[2],
                  "text": item[3]})
  
            # Sort the transcript according to the starting time stamp
            transcript.sort(key=lambda x: x["start"])
  
            # Extract only the text
            longtext = [item["text"] for item in transcript]
  
            # Lowercase + remove punctuation
            cleaned_orig = preprocess_texts(longtext)
  
            # Remove single quotation marks around sentences
            orig_trans = " ".join(cleaned_orig)
  
            # Remove notes *d, *u, *a, *v, *x, *z
            cleaned_orig = re.sub(r'\*[duavxz]', '', orig_trans)
  
            # Remove notes ggg, xxx, and Xxx
            cleaned_orig = re.sub(r'\bggg\b|\bxxx\b|\bXxx\b', '', cleaned_orig)
  
            # Append the cleaned transcription to the dataframe
            
            df_orig = df_orig.append({"Audio File": file_name, 
                                      "Original Transcription": cleaned_orig}, 
                                      ignore_index=True)
            
    
        # Return the dataframe
        #print(df_orig)
        return df_orig
    
    # Function to pre-process the transcription (lowercase + removing punctuation)
    def preprocess_texts(texts: List[str],
                         to_lowercase: bool = True,
                         remove_punctuation: bool = True,
                         remove_stopwords: bool = False,
                         map_to_number: bool = False
                         ) -> List[str]:
        # Preprocess text by mapping to lowercase, removing punctuation,
        # removing stopwords, and converting numbers to written form
        new_texts = []
        for sentence in texts:
            if to_lowercase:
                sentence = sentence.lower()
            if remove_punctuation:
                for s in punctuation:
                    sentence = sentence.replace(s, "")
            if map_to_number:
                words = sentence.split(" ")
                converted_words = []
                for word in words:
                    if word.isdigit():
                        converted_word = map_to_dutch_number(word)
                        converted_words.append(converted_word)
                    else:
                        converted_words.append(word)
                sentence = " ".join(converted_words)
            if len(sentence) > 0:
                new_texts.append(sentence)
        return new_texts
    
    # function to do Whisper for multiple files
    def original_whisper(data_dir_audio, model):
    
        # Create an empty dataframe
        df_whisper = pd.DataFrame(columns=["Audio File", 
                                           "Whisper Transcription", 
                                           "Audio Duration",
                                           "Running Time", 
                                           "Real-time Factor",
                                           "Memory usage",
                                           "Audio size"])
    
        # Loop to go through all files in the folder
        for root, subdirs, files in os.walk(data_dir_audio):
           files = [f for f in files if f in audio_sample]
           for name in tqdm(files):
              filepath = os.path.join(root, name)
            
  
              # Extract the name of the audio file without the extension
              file_name = os.path.splitext(name)[0]
  
              # Extract audio duration
              duration = audio_duration(filepath)
  
              # Extract audio size
              size = os.path.getsize(filepath)
  
              # Track time
              start = time.time()
  
              # Track memory
              tracemalloc.start()
  
              # Load the model
              #model = whisper.load_model("tiny")
  
              # Transcribe audio file + save
              result = model.transcribe(filepath)
              only_text = result["text"]
              result_whisper = preprocess_texts([only_text], map_to_number = True)
  
              # Calculate running time
              end = time.time()
              running_time = end - start
  
              # Calculate RTF
              rtf = running_time / duration
  
              # Memory usage
              memory = tracemalloc.get_traced_memory()
  
              # Append the transcription, audio duration, and running time to the dataframe
              df_whisper = df_whisper.append(
                  {
                      "Audio File": file_name,
                      "Whisper Transcription": result_whisper,
                      "Audio Duration": duration,
                      "Running Time": running_time,
                      "Real-time Factor": rtf,
                      "Memory usage" : memory,
                      "Audio size" : size
                  },
                  ignore_index=True
              )
        # Return the dataframe
        return df_whisper
               
    

    # Function for performance metrics
    def performance(reference, hypothesis):
        # Data frame to store the performance metrics
        df_performance = pd.DataFrame(columns=["Audio File", "WER", "Precision", "Recall", "F1"])
    
        # Sort dataset according to audio files
        sorted_ref = reference.sort_values("Audio File")
        sorted_hyp = hypothesis.sort_values("Audio File")
    
        # Check if dataframes are of equal length
        if len(sorted_ref) == len(sorted_hyp):
    
          # If so, loop through dataframe
          for i in range(len(sorted_ref)):
    
            # Save audio name
            audio_file = sorted_ref.iloc[i]["Audio File"]
    
            # Get transcriptions only
            orig_transcription = sorted_ref.iloc[i]["Original Transcription"]
            whisper_transcription = sorted_hyp.iloc[i]["Whisper Transcription"]
    
            # Alignment original transcript + whisper output
            output = jiwer.process_words(reference=orig_transcription,
                                         hypothesis=whisper_transcription)
            
            # Calculate WER
            wer_score = jiwer.wer(orig_transcription, whisper_transcription)
    
            substitutions = output.substitutions
            deletions = output.deletions
            insertions = output.insertions
            hits = output.hits
                
            # Calculate precision, recall, and F1
            precision = hits / (hits + substitutions + insertions)
            recall = hits / (hits + substitutions + deletions)
            f1 = (2 * precision * recall) / (precision + recall)
    
            # Append the performance metrics to the dataframe
            df_performance = df_performance.append(
                {
                    "Audio File": audio_file,
                    "WER": wer_score,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                },
                ignore_index=True
            )
    
        return df_performance

    # Get transcriptions
    orig_trans = get_transcripts(data_dir_transcription)
    
    dataset_name = os.path.basename(data_dir_audio).split(" - ")[0]

    # Save transcriptions in save dir
    for dir_name in [save_dir_transcriptions, save_dir_whisper, save_dir_performance]:
      os.makedirs(dir_name, exist_ok=True)
    
    path = os.path.join(save_dir_transcriptions,
                        f"{dataset_name}_whisper_annotation.csv")
    with open(path, 'w', encoding = 'utf-8-sig') as f:
      orig_trans.to_csv(f, index = False)
      
    # Define model size Whisper
    #print(device)
    model = whisper.load_model(size, device)
    
    # Run Whisper
    orig_whis = original_whisper(data_dir_audio, model)
    
    # Save Whisper output in Google Drive
    path = os.path.join(save_dir_whisper,
                        f"{dataset_name}_whisper_prediction.csv")
    with open(path, 'w', encoding = 'utf-8-sig') as f:
      orig_whis.to_csv(f, index = False)
      
    # Check performance
    performance_output = performance(orig_trans, orig_whis)
    
    # Save performance dataframe in folder
    path = os.path.join(save_dir_performance,
                        f"{dataset_name}_whisper_performance.csv")
    with open(path, 'w', encoding = 'utf-8-sig') as f:
        performance_output.to_csv(f, index = False)



if __name__ == "__main__":
    main()
