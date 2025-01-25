import os
import time
import torch
import numpy as np
import pandas as pd
import torchaudio

from torchaudio.datasets import LIBRISPEECH
from faster_whisper import WhisperModel, BatchedInferencePipeline
from jiwer import wer
from sacrebleu import corpus_bleu
from silero_vad import load_silero_vad, get_speech_timestamps
import re

DATASET_PATH = "./librispeechs"                  # Path to store/download LibriSpeech
OUTPUT_CSV = "whisper_turbo_test_other.csv"    # CSV output
LIBRISPEECH_URL = "test-other"                  # The noisy test set
SAMPLE_RATE = 16000                             # Silero VAD and most Whisper models expect 16kHz

print("Loading/Downloading the LibriSpeech 'test-other' dataset...")
librispeech_test = LIBRISPEECH(
    root=DATASET_PATH,
    url=LIBRISPEECH_URL,
    download=True
)

print("Loading Silero VAD model...")
vad_model = load_silero_vad()

print("Loading Faster Whisper model...")
whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="int8")
batched_model = BatchedInferencePipeline(model=whisper_model)

def apply_silero_vad_in_memory(audio: torch.Tensor, sampling_rate: int):
    """
    Applies Silero VAD to an in-memory audio Tensor and returns:
        - Concatenated speech Tensor
        - Speech timestamps in seconds
    """
    # Silero expects a single-channel (mono), float32, 16kHz audio input
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0)  # Convert stereo to mono

    # Ensure the audio is at 16kHz
    if sampling_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=SAMPLE_RATE)
        sampling_rate = SAMPLE_RATE

    # Normalize audio to [-1, 1] if not already
    if audio.dtype != torch.float32:
        max_val = float(torch.max(torch.abs(audio)))
        if max_val > 0:
            audio = audio / max_val

    # Convert to float32 NumPy array
    wav = audio.numpy().astype(np.float32)

    # Ensure wav is 1D
    if wav.ndim > 1:
        wav = wav.squeeze()

    # Get speech timestamps using VAD
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=0.2,  # Lowered threshold for better sensitivity
        return_seconds=True  # Return timestamps in seconds instead of samples
    )

    # If no speech detected, return empty
    if not speech_timestamps:
        return torch.tensor([]), []

    # Concatenate speech segments based on timestamps
    speech_segments = [
        torch.tensor(wav[int(ts['start'] * sampling_rate): int(ts['end'] * sampling_rate)])
        for ts in speech_timestamps
    ]

    # Concatenate all speech segments
    speech_audio = torch.cat(speech_segments) if speech_segments else torch.tensor([])

    return speech_audio, speech_timestamps

def transcribe_audio_faster_whisper(audio_tensor: torch.Tensor, beam_size: int = 5):
    """
    Transcribes audio using Faster Whisper. Expects a 16-bit PCM-like range.
    Returns the combined transcription string and total time taken.
    """
    # Convert to float32, scale from int16 range (-32768, 32767) to (-1, 1)
    audio_np = audio_tensor.numpy().astype(np.float32)

    start_time = time.time()
    segments, info = whisper_model.transcribe(audio_np, beam_size=beam_size, language="en")
    texts = [seg.text.strip() for seg in segments]
    transcription = " ".join(texts)
    transcription_time = time.time() - start_time
    return transcription, transcription_time

def compute_metrics(hypothesis: str, reference: str):
    """
    Compute WER and BLEU scores for a single hypothesis-reference pair.
    """
    # jiwer for WER
    w = wer(reference, hypothesis)

    # sacrebleu for BLEU
    # references must be in an extra list dimension => [[reference]]
    bleu = corpus_bleu([hypothesis], [[reference]]).score

    return w, bleu

def preprocess_text(text):
    """
    Preprocess text by removing punctuation and converting to lowercase.
    """
    # Remove punctuation using regex
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

results = []

for idx, (audio, sr, transcript, speaker_id, chapter_id, utterance_id) in enumerate(librispeech_test):
    sample_index = idx + 1
    if sample_index % 100 == 0:
        print(f"Processing sample {sample_index}...")

    try:
        # 3.1 Resample if needed
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Ensure mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 3.2 Get the duration (in seconds) of the raw audio
        audio_length_seconds = audio.shape[-1] / sr

        # 3.3 Apply Silero VAD
        speech_audio, speech_timestamps = apply_silero_vad_in_memory(audio, sr)

        # If no speech is detected, skip
        if speech_audio.numel() == 0:
            print(f"No speech detected for sample {sample_index}, skipping...")
            # Optionally, save the audio for debugging
            # torchaudio.save(f"no_speech_sample_{sample_index}.wav", audio.unsqueeze(0), sr)
            continue

        # 3.4 Transcribe with Faster Whisper
        hypothesis, time_taken = transcribe_audio_faster_whisper(speech_audio)

        # 3.5 Compute Metrics
        hypothesis = preprocess_text(hypothesis)
        reference = preprocess_text(transcript)
        w, bleu = compute_metrics(hypothesis, reference)

        # 3.6 Append results
        results.append({
            "Sample Index": sample_index,
            "Audio Length (s)": round(audio_length_seconds, 2),
            "Transcription Time (s)": round(time_taken, 8),
            "WER": round(w, 4),
            "BLEU Score": round(bleu, 3),
            "Hypothesis": hypothesis,
            "Reference": reference,
            "Speech Timestamps": str(speech_timestamps)
        })

    except Exception as e:
        print(f"Error processing sample {sample_index}: {e}")
        continue
# -----------------------------
# 4. Save Results to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Results have been saved to {OUTPUT_CSV}.\n")

hypothesis_list = []
reference_list = []
for i in df["Hypothesis"]:
    i = i.split()
    hypothesis_list.append(i)

for j in df["Reference"]:
    j = j.split()
    reference_list.append(j)

reference_lists = [[ref] for ref in reference_list]

bleu_score_corpus = corpus_bleu(reference_lists, hypothesis)
print("Corpus BLEU Score: ", bleu_score_corpus)

