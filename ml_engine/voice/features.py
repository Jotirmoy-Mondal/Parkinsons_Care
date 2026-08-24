"""Here is the step-by-step flowchart of exactly what this "Medical Physicist" code is doing behind the scenes
### 🌊 The Acoustic Extraction Flowchart

**[ 📥 STEP 1: Receive the Audio ]**

* The system takes in the patient's audio recording.
* It sets a "human voice search window" (between 75 Hz and 500 Hz). This tells the physics engine to ignore extreme background noises (like low rumbles or high-pitched squeaks) and only look for human vocal cords.
⬇️

**[ 🔊 STEP 2: Load the Physics Engine ]**

* The raw audio file is swallowed by the Praat physics engine, converting the digital MP3/WAV file into a pure mathematical sound wave.
⬇️

**[ 🗺️ STEP 3: Map the Voice (Two Parallel Tracks) ]**
The engine immediately creates two different maps of the patient's voice:

* **Map A (The Pitch Tracker):** It scans the whole file and tracks the overall melody/pitch of the voice over time.
* **Map B (The Peak Pinpointer):** It creates a highly precise map that places a microscopic dot on the exact peak of *every single sound wave* in the recording.
⬇️

**[ 🔬 STEP 4: Extract the Micro-Tremors (The Biomarkers) ]**
Using those maps, the engine runs three distinct medical tests:

* **Test 1 (JITTER):** It looks at Map B and checks the horizontal timing between wave peaks. If the timing stutters or varies, it records a Jitter score (Pitch instability).
* **Test 2 (SHIMMER):** It looks at Map B and checks the vertical height of the wave peaks. If the loudness rapidly flickers up and down, it records a Shimmer score (Volume instability).
* **Test 3 (HNR):** It calculates a ratio comparing the amount of pure ringing voice against the amount of "white noise" (raspy air escaping through the vocal cords).
⬇️

**[ 📊 STEP 5: Analyze Overall Pitch Stability ]**

* The engine looks back at **Map A** (The Pitch Tracker).
* First, it throws away any moments of pure silence or unvoiced breathing (so it doesn't accidentally calculate silence into the patient's score).
* Then, it looks at the surviving voice data and calculates four things:
1. The Average Pitch
2. The Pitch Variance (How wildly their voice swings up and down)
3. The Lowest Pitch hit
4. The Highest Pitch hit
⬇️



**[ 📦 STEP 6: Clean Up & Ship Out ]**

* The system gathers all these medical measurements into a single neat package.
* It rounds every number down to exactly 4 decimal places (so the database doesn't get clogged with infinite fractions).
* It hands the final, clean "Medical Report" back to your app to be saved!"""





# import parselmouth
# What it is: A Python port of Praat, the global gold standard for speech science.
# Why we need it: Unlike Torchaudio (which acts as an "AI Detective" looking for 
# hidden patterns), Parselmouth uses pure acoustic physics to measure exact, 
# clinically proven Parkinson's biomarkers—specifically Jitter (pitch tremors) 
# and Shimmer (volume tremors).
import parselmouth

# from parselmouth.praat import call
# What it is: The Python-to-Praat translator function.
# How: The core physics engine (Praat) doesn't naturally speak Python; it uses 
# its own scripting language. The 'call' function bridges this gap. It allows 
# us to pass raw Praat string commands (like "Get jitter (local)") directly 
# into the engine and get standard Python numbers back.
from parselmouth.praat import call

import numpy as np


def extract_acoustic_features(audio_path: str, min_f0: float = 75, max_f0: float = 500) -> dict:
    """
    Extract clinically-established PD voice biomarkers using Praat (via parselmouth).
    min_f0/max_f0: expected pitch range in Hz — 75-500Hz covers typical adult
    speech; narrows the pitch-detection search window for more reliable results.
    """
    
    sound = parselmouth.Sound(audio_path)
    # The Operating Table: Fetches the raw audio file from the server's hard 
    # drive and unpacks it into a pure mathematical sound wave object so the 
    # Praat physics engine can analyze it.

    # Pitch (F0) object — needed for jitter and F0 stats
    # Takes: The raw sound object, plus the strict 75Hz-500Hz search window.
    # Returns: A Praat Pitch Object (a map of the vocal melody over time).
    # How: It chops the audio into tiny frames and calculates the vibration 
    # speed (pitch) at each millisecond. The floor and ceiling act as strict 
    # guardrails for the math algorithm, preventing it from accidentally 
    # tracking "harmonics" (high-pitched acoustic echoes) or low background rumble.
    pitch = sound.to_pitch(pitch_floor=min_f0, pitch_ceiling=max_f0)

    # PointProcess — required by Praat's jitter/shimmer calls

    # Takes: The raw sound, the specific Praat algorithm command, and the pitch guardrails.
    # Returns: A Praat PointProcess Object.
    # How: This uses "Cross-Correlation" math to find the exact repeating peaks of 
    # the vocal sound wave. It drops a microscopic pinpoint on the exact millisecond 
    # of every single vocal cord vibration. These precise mile-markers are 
    # absolutely required to calculate the Jitter and Shimmer tremors in the next step.

    """ different button in praat:
    1. To track the melody:
    "To Pitch"

    2. To find the wave peaks:
    "To PointProcess (periodic, cc)"

    3. To measure Pitch Tremors:
    "Get jitter (local)"

    4. To measure Volume Tremors:
    "Get shimmer (local)"

    5. To measure Breathiness:
    "To Harmonicity (cc)"
    """

    point_process = call(sound, "To PointProcess (periodic, cc)", min_f0, max_f0)

    features = {}

    """0, 0 (Time Range): Tells the engine to scan the entire audio file from start to finish.

    0.0001, 0.02 (Period Limits): The absolute shortest and longest time a human vocal cord can physically stay open. It forces Praat to ignore impossible timings.

    1.3 (The Glitch Filter): This is the most important one. It means 30%. If the distance between two waves suddenly jumps by more than 30%, Praat assumes it was a microphone glitch (or a cough) rather than a vocal tremor, and safely ignores it so it doesn't ruin the patient's score!"""

    # --- Jitter: cycle-to-cycle frequency perturbation ---
    # What it measures: Involuntary micro-tremors in the patient's pitch/timing.
    # 'local': Measures the raw timing difference between exactly 2 adjacent waves.
    # 'rap' (Relative Average Perturbation): Measures the difference across 3 adjacent 
    # waves. RAP acts as a mathematical smoother, making it much more reliable at 
    # ignoring random microphone static and identifying true Parkinson's tremors.
    # The parameters (0, 0, 0.0001, 0.02, 1.3) tell Praat to scan the whole file, 
    # ignore impossible human timings, and throw out any jumps larger than 30%.

    features["jitter_local"] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    features["jitter_rap"] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

    # --- Shimmer: cycle-to-cycle amplitude perturbation ---
    # What it measures: Involuntary micro-tremors in the patient's volume/loudness.
    # 'local': Measures the raw volume difference between exactly 2 adjacent waves.
    # 'apq3' (Amplitude Perturbation Quotient): Measures the difference across 3 
    # adjacent waves. Like RAP, it acts as a mathematical smoother to filter out 
    # random microphone pops or sudden breaths.
    # The new parameter at the end (1.6) tells Praat to completely ignore any 
    # moment where the volume suddenly jumps by more than 60%, treating it as 
    # non-human noise rather than a Parkinson's tremor.
    features["shimmer_local"] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    features["shimmer_apq3"] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # --- HNR: Harmonics-to-Noise Ratio (breathiness indicator) ---
   
    # What it measures: How much pure voice (Harmonics) is present compared 
    # to raspy, leaking air (Noise). 
    # Parkinson's Connection: Muscle rigidity often prevents the vocal cords 
    # from closing completely, allowing air to constantly leak during speech. 
    # This results in a weak, breathy voice and a mathematically low HNR score.
    # We use 'to_harmonicity' to calculate the ratio over time, and then 
    # average it ('Get mean') across the entire file (0, 0).
    harmonicity = sound.to_harmonicity()
    features["hnr_mean"] = call(harmonicity, "Get mean", 0, 0)

    # --- F0 statistics: pitch stability ---
    # f0_values = pitch.selected_array["frequency"]
    # f0_values = f0_values[f0_values != 0]
    # How: First, we extract the raw frequency numbers from the Praat object 
    # into a standard Python array. Second, we filter out all the exact '0's. 
    # Praat records a 0 anytime the vocal cords stop vibrating (like during 
    # a pause, a breath, or an 'S' sound). If we don't throw out these 0s, 
    # they will mathematically ruin the patient's average pitch score.

    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values != 0]  # remove unvoiced frames (0 = no pitch detected)

    
    
    
    if len(f0_values) > 0:
        features["f0_mean"] = float(np.mean(f0_values))
        features["f0_std"] = float(np.std(f0_values))
        features["f0_min"] = float(np.min(f0_values))
        features["f0_max"] = float(np.max(f0_values))
    else:
        # No voiced frames detected — recording may be silent/corrupted
        features["f0_mean"] = features["f0_std"] = None
        features["f0_min"] = features["f0_max"] = None

        # What it does: Calculates the final F0 statistics, safely handling empty audio.
        # How: It uses numpy to extract the Mean, Min, Max, and Standard Deviation. 
        # Standard Deviation (f0_std) is highly important, as a low score indicates 
        # the "monotone" voice characteristic of Parkinson's rigidity.
        # We wrap the numpy outputs in standard float() to ensure they can be easily 
        # converted to JSON later. The 'else' block prevents a fatal server crash 
        # if the user uploads a completely silent audio file.

    # ==========================================
    # FINAL QUALITY CONTROL
    # ==========================================
    
    clean_features = {}

    # Open the original cabinet and look at every single folder one by one
    for key, value in features.items():
        
        # If the value is a decimal number, chop it to 4 decimal places
        if isinstance(value, float):
            clean_features[key] = round(value, 4)
            
        # If the value is anything else (like 'None'), just copy it over untouched
        else:
            clean_features[key] = value

    # Ship the final, clean dictionary to Django
    return clean_features