# Faster Whisper and Silero VAD for LibriSpeech Transcription and Evaluation

## Overview
This script processes and evaluates the "test-other" subset of the LibriSpeech dataset using:
- **Silero Voice Activity Detection (VAD)** for speech segmentation.
- **Faster Whisper** for transcription.
- **jiwer** and **sacrebleu** for Word Error Rate (WER) and BLEU score computation.

The script saves results, including evaluation metrics, to a CSV file for further analysis.

---

## Key Features
1. **Dataset**: Downloads and processes the "test-other" set of LibriSpeech.
2. **Voice Activity Detection**: Applies Silero VAD to isolate speech segments.
3. **Transcription**: Uses Faster Whisper to transcribe segmented audio.
4. **Metrics Computation**: Calculates:
   - WER using `jiwer`.
   - BLEU scores using `sacrebleu`.
5. **Results Storage**: Saves processed data and metrics to a CSV file.

---

## Dependencies
The script requires the following Python packages:
- `torch`
- `torchaudio`
- `numpy`
- `pandas`
- `faster-whisper`
- `jiwer`
- `sacrebleu`
- `silero-vad`

---

## How It Works
### 1. Dataset Setup
- Downloads the LibriSpeech `test-other` set.
- Each audio file is loaded as a tensor with its metadata.

### 2. Audio Preprocessing
- Ensures mono-channel and resamples to 16 kHz.
- Normalizes audio to float32 format for compatibility with Silero VAD.

### 3. Speech Segmentation (Silero VAD)
- Detects timestamps of speech in the audio using Silero VAD.
- Concatenates detected speech segments into a single tensor.

### 4. Transcription (Faster Whisper)
- Transcribes speech audio using the Faster Whisper model.
- Supports English language transcription with beam search decoding.

### 5. Evaluation
- Preprocesses text by removing punctuation and converting to lowercase.
- Computes:
  - **WER**: Compares the hypothesis transcription with the reference text.
  - **BLEU Score**: Measures transcription accuracy at the corpus level.

### 6. Results Storage
- Saves detailed results for each audio sample to a CSV file:
  - Sample index
  - Audio length
  - Transcription time
  - WER
  - BLEU score
  - Hypothesis (transcription)
  - Reference text
  - Detected speech timestamps

---

## Output
1. **CSV File**:
   - Results are stored in `whisper_turbo_test_other.csv`.
2. **Corpus BLEU Score**:
   - Prints the overall BLEU score across all transcriptions.
