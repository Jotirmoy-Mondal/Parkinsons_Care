"""🌊 The Grand Inference Flowchart
[ 📥 STEP 1: The Request Arrives ]

Your Django server receives a new voice recording from a patient.

It calls run_voice_inference(), handing it three things:

The new audio file.

The patient's Baseline Fingerprint (their healthy voice from Day 1).

The patient's Recent Fingerprint (their average voice over the last 30 days).
⬇️

[ 🧠 STEP 2: Awaken the AI Detective (Singleton Load) ]

The system calls _load_model().

Smart Feature: Instead of wasting 5 seconds loading the massive PyTorch AI into memory every single time a patient uploads a file, it checks if the AI is already awake (_model is None). If it is, it skips loading and reuses the awake AI. This makes your server lightning-fast!
⬇️

[ 🕵️‍♂️ STEP 3: Generate Today's Fingerprint ]

The audio file is handed to compute_embedding().

The audio is converted into a Mel-Spectrogram picture.

The AI looks at the picture and outputs a unique 128-number Vocal Fingerprint (Embedding) representing the patient's voice today.
⬇️

[ 📏 STEP 4: The Mathematical Comparison ]

The system calls euclidean_distance().

It takes Today's Fingerprint and compares it to the Baseline Fingerprint.

It calculates the exact mathematical distance between them. (Distance = 0 means the voice is perfectly identical. A high distance means the voice has severely changed).
⬇️

[ 📉 STEP 5: Translate to a Human Score ]

The system calls distance_to_stability().

Telling a doctor "The Euclidean distance is 1.42" is completely useless.

This function uses Exponential Decay to translate that raw distance into a beautiful, easy-to-understand 0 to 100% Stability Score. (e.g., 85.5% Stable).
⬇️

[ 🔬 STEP 6: Call the Medical Physicist ]

The AI has done its job. Now, the system calls extract_acoustic_features() (the Parselmouth code we just built!).

It extracts the clinical Jitter, Shimmer, and HNR numbers from the exact same audio file.
⬇️

[ 📅 STEP 7: The Short-Term Check (Optional) ]

If Django provided a 30-day "Recent" fingerprint, the system runs the exact same distance and stability math again, this time comparing Today vs. the Last 30 Days to see if the patient had a sudden drop this month.
⬇️

[ 📦 STEP 8: Package & Ship ]

The system bundles everything into one neat Dictionary:

The raw 128-number AI Fingerprint (to be saved to the database).

The AI Stability Score vs. Baseline.

The AI Stability Score vs. Recent.

The Clinical Biomarkers (Jitter, Shimmer, HNR).

It hands this completed package back up to Django to be converted into JSON and sent to the Android app!"""

import math
import torch

from ml_engine.voice.model import SiameseVoiceNet
from ml_engine.voice.preprocess import preprocess_audio
from ml_engine.voice.features import extract_acoustic_features  # jitter/shimmer/HNR

MODEL_VERSION = "voice_cnn_v1"
WEIGHTS_PATH = "ml_engine/voice/weights/voice_model.pth"
EMBEDDING_DIM = 128

# Calibrate this against your training data's distance distribution —
# see calibrate_decay_rate() below. This is a placeholder until you do.
#The Decay Rate is the "Teacher" that you hire to grade the AI's raw distance and turn it into a 0-100% Stability Score.
DECAY_RATE = 2.0

#"_"This is Private. For internal use only
_model = None  # module-level singleton, loaded once

"""Singleton is a strict architectural rule. It simply means: "No matter how many times you ask for this object, I will only ever create ONE single copy of it in memory, and everyone has to share it."
"""

def _load_model() -> SiameseVoiceNet:
    """Load model weights once; reused across all requests."""
    global _model # to pass the value to outside _model variable
    if _model is None:
        _model = SiameseVoiceNet(embedding_dim=EMBEDDING_DIM)
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        _model.load_state_dict(state_dict)
        _model.eval()  # disables dropout/batchnorm training behavior
    return _model


def compute_embedding(audio_path: str) -> torch.Tensor:
    """Preprocess an audio file and return its embedding vector."""
    model = _load_model()
    mel_spec = preprocess_audio(audio_path)      # [1, 80, time_frames]
    mel_spec = mel_spec.unsqueeze(0)              # add fake batch dim -> [1, 1, 80, time_frames]


    # --- Extract the Vocal Fingerprint ---
    # torch.no_grad() explicitly turns off the PyTorch calculus/learning engine. 
    # This saves massive amounts of server RAM and makes inference lightning fast.
    # The model extracts the 128-dimensional embedding from the spectrogram.
    # Finally, .squeeze(0) rips open the "fake box" (batch dimension) we created 
    # earlier, allowing us to safely return just the raw 1D list of coordinates.
    
    # 1. We enter the "safe room". Python automatically turns OFF the engine.
    with torch.no_grad():
        
        # 2. The AI extracts the fingerprint
        embedding = model.embed(mel_spec)
        
    # 3. We un-indent. Python automatically turns the engine back ON right here!
    # Even if the extraction crashed, Python guarantees the engine is flipped back on.
    
    return embedding.squeeze(0)


def euclidean_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:

    # Calculates the Euclidean distance (L2 Norm) between the two 128-number vectors.
    # It subtracts the coordinates (emb1 - emb2), applies the distance formula (p=2),
    # and uses .item() to rip the raw Python float out of the PyTorch tensor wrapper.
    distance = torch.norm(emb1 - emb2, p=2).item()


def distance_to_stability(distance: float, decay_rate: float = DECAY_RATE) -> float:
    """
    Converts a raw distance into a 0-100 stability score.
    distance=0 (identical) -> 100%. Larger distance -> decays toward 0.
    NOTE: decay_rate must be calibrated against real distance distributions
    (see calibrate_decay_rate) — an uncalibrated rate gives a meaningless number.
    """
    stability = 100 * math.exp(-decay_rate * distance)
    return round(stability, 1)


def run_voice_inference(
    audio_path: str,
    baseline_embedding: torch.Tensor,
    recent_mean_embedding: torch.Tensor = None,
) -> dict:
    """
    Full inference for one uploaded recording.

    baseline_embedding: patient's fixed enrollment anchor (long-term trend)
    recent_mean_embedding: rolling 30-day mean embedding, optional (short-term trend)
    """
    today_embedding = compute_embedding(audio_path)

    dist_to_baseline = euclidean_distance(today_embedding, baseline_embedding)
    stability_vs_baseline = distance_to_stability(dist_to_baseline)

    result = {
        "embedding": today_embedding.tolist(),  # store as JSON in VoiceTestResult
        "model_version": MODEL_VERSION,
        "stability_vs_baseline": stability_vs_baseline,
        "raw_features": extract_acoustic_features(audio_path),
    }

    if recent_mean_embedding is not None:
        dist_to_recent = euclidean_distance(today_embedding, recent_mean_embedding)
        result["stability_vs_recent"] = distance_to_stability(dist_to_recent)

    return result


    # ml_engine/voice/inference.py, temporarily bypass the .pth load

if __name__ == "__main__":
    model = SiameseVoiceNet(embedding_dim=EMBEDDING_DIM)
    model.eval()  # random weights, untrained — that's fine for this test

    mel_spec = preprocess_audio("data/raw/voice_sample/file_example_WAV_1MG.wav")
    mel_spec = mel_spec.unsqueeze(0)

    with torch.no_grad():
        embedding = model.embed(mel_spec)

    print("Embedding shape:", embedding.shape)   # should be [1, 128]
    print("Embedding sample:", embedding[0][:5])  # first 5 values, just to see it's not NaN/garbage