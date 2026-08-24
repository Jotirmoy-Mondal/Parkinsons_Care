# ml_engine/voice/preprocess.py

import torch
import torchaudio
"""
WHY WE USE TORCHAUDIO:
It acts as the translator between raw audio and our AI model.
1. Loads raw .wav voice recordings into PyTorch Tensors.
2. Resamples audio to standardize inputs from different microphones.
3. Extracts features (Mel Spectrogram) to turn soundwaves into a mathematical 
"image", allowing the neural network to detect Parkinson's vocal biomarkers.
"""
import librosa
"""
WHY WE MIGHT USE LIBROSA:
Librosa is a specialized audio analysis library used for feature extraction.
While torchaudio is great for deep learning tensors, librosa excels at 
pulling out specific vocal biomarkers (like MFCCs, pitch, and tone) that 
are highly relevant for detecting Parkinson's disease.
"""

TARGET_SAMPLE_RATE = 16000

"""WHY 16,000 Hz?
16kHz is the industry standard for speech processing (used by Whisper, Wav2Vec, etc.).
It successfully captures all necessary human vocal frequencies (up to 8kHz) required 
for detecting Parkinson's biomarkers, while stripping out high-frequency noise and 
keeping our AI model computationally lightweight.
Nyquist theorem: f_sample >= 2 * f_max
Where f_sample is your sample rate and f_max is the maximum frequency you want to capture.
"""


N_MELS = 80  # was 128 — 128 is too many for 16kHz audio, causes empty filterbanks

"""
WHY N_MELS = 80?
The Mel scale mimics human hearing by focusing on lower speech frequencies. 
Setting this to 80 creates 80 distinct frequency 'buckets' per time step. 
This is the industry standard (used by OpenAI's Whisper) because it perfectly 
balances high-resolution speech detail (needed for Parkinson's tremor detection) 
while avoiding empty data tensors.
"""

N_FFT = 400 # Fast Fourier Transform window size

"""
WHY N_FFT = 400?
N_FFT defines the 'window size' for analyzing the audio. At a 16kHz sample 
rate, an N_FFT of 400 equals exactly 25 milliseconds of audio. This is the 
industry standard for speech processing because the human vocal tract holds 
a shape for ~20-30ms. It provides a perfect snapshot of the voice without 
blurring the fast micro-tremors associated with Parkinson's.
"""
HOP_LENGTH = 160      
"""
WHY HOP_LENGTH = 160?
Hop length dictates how far our 25ms window (N_FFT) slides forward. 
At 16kHz, 160 samples equals exactly 10 milliseconds. This means our 25ms 
windows overlap by 15ms. This heavy overlap ensures a smooth, continuous 
feature extraction without losing data at the edges of the frames, which 
is crucial for tracking continuous vocal tremors in Parkinson's patients.
"""


def load_and_resample(audio_path: str) -> torch.Tensor:#rorch.tensor is promised output
    """Load audio and resample to a fixed 16kHz mono waveform."""
    waveform, sample_rate = torchaudio.load(audio_path) ## output_package is exactly this: ( tensor_data, 44100[orginal sample rate] )

    # Collapse to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

        """
        STEREO TO MONO CONVERSION:
        * if waveform.shape[0] > 1: Checks the tensor's shape to see if the audio 
        has more than one channel (like Stereo).
        * waveform.mean(...): If stereo, it calculates the average (mean) between 
        the Left and Right channels at every single microsecond.
        * dim=0: Tells PyTorch to average across the channels.
        * keepdim=True: Ensures the final tensor stays 2D (e.g., [1, 44100]) 
        instead of flattening, which prevents model crashes.
        Result: Strips useless spatial data and leaves only the raw vocal data!
        """


    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform
    """EXAMPLE: 
    class MyCustomTool:
    # 1. The builder (this runs when you create the tool)
    def __init__(self, target_speed):
        self.target_speed = target_speed
        print(f"Tool built! Target is {self.target_speed}")

    # 2. The magic trigger (this lets the variable act like a function)
    def __call__(self, audio_data):
        print(f"Processing {audio_data} at {self.target_speed} Hz...")
        return "New Standardized Audio"


# Step 1: Build the object and assign it to a variable
# (This triggers the __init__ method)
my_variable = MyCustomTool(16000) 

# Step 2: Treat the variable like a function! 
# (This secretly triggers the __call__ method)
final_audio = my_variable("Raw Patient Voice")"""


def trim_silence(waveform: torch.Tensor, top_db: int = 30) -> torch.Tensor:
    #top_db: int = 30 (Type Hinting)
    """
    DECIBEL CUTOFF (NOISE GATE):
    top_db=30 defines the dynamic range of our Mel Spectrogram. It mathematically 
    locates the loudest peak in the voice recording and deletes any sound that is 
    more than 30 decibels quieter than that peak. This acts as an aggressive filter, 
    stripping away background room hiss and forcing the AI to focus exclusively on 
    the high-energy vocal cord features where Parkinson's tremors are visible.
    """

    """Trim leading/trailing silence using energy-based VAD."""

    audio_np = waveform.squeeze().numpy()
    """Step 1: Overcoming the Language Barrier
    audio_np = waveform.squeeze().numpy()

    librosa is an incredible audio scientist, but it doesn't speak PyTorch. It only understands standard Python NumPy arrays.

    .squeeze() takes your audio out of its 2D PyTorch box (changing it from [1, 44100] to just a flat list of [44100]).

    .numpy() instantly translates that list into the NumPy language so librosa can read it."""

    # librosa.effects.trim returns TWO values: the trimmed audio, and the index 
    # of the cuts. We use the underscore "_" as a throwaway variable to catch and 
    # ignore the cut indexes, keeping only the audio data in "trimmed".

    trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
    return torch.from_numpy(trimmed).unsqueeze(0)


def normalize_volume(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize so loudness differences between phones/mics don't bias the model."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform
    """
    AUDIO NORMALIZATION (PEAK MATCHING):
    Patients record at vastly different volumes. This code acts as an automatic 
    equalizer to prevent the AI from confusing volume with disease presence.
    1. waveform.abs().max(): Locates the absolute loudest peak in the recording.
    2. waveform / peak: Divides the entire tensor by that peak value. This scales 
    all audio up or down so the loudest moment perfectly hits 1.0 or -1.0. 
    Result: Every patient is standardized to the exact same maximum volume.
    """


def compute_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform to a log-scaled mel-spectrogram, shape [1, n_mels, time_frames]."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    """MelSpectrogram: Builds  a      
    constants (16kHz, 25ms windows, 10ms hops, 80 mel bands) to extract 
    human-like frequency features.
    mel_spec = mel_transform(waveform) #it transforms the data from a simple 2D line (Volume over Time) into a 3D mathematical heatmap (Frequencies and Volume over Time)."""

    # Log scale — raw mel energies span a huge range, log compresses it
    # to something the CNN trains on far more stably.
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec) # using __call__ methon in one line of code
    """
    DECIBEL CONVERSION (LOG-SCALING):
    Neural networks struggle to read raw acoustic energy because the mathematical 
    gap between quiet and loud sounds is too vast. AmplitudeToDB converts the raw 
    Mel Spectrogram into a logarithmic decibel scale. This tightly compresses the 
    data, enhancing the quiet, micro-vocal tremors associated with Parkinson's 
    so the AI can easily detect them.
    """


    return log_mel_spec


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """Full pipeline: load -> trim -> normalize -> mel-spectrogram. Ready for model input."""
    waveform = load_and_resample(audio_path)
    waveform = trim_silence(waveform)
    waveform = normalize_volume(waveform)
    mel_spec = compute_mel_spectrogram(waveform)
    return mel_spec


if __name__ == "__main__": #means "only run this block if I'm the file being directly executed, not if someone is just importing me."
    mel = preprocess_audio("data/raw/voice_sample/file_example_WAV_1MG.wav")  # relative to project root
    print(mel.shape)