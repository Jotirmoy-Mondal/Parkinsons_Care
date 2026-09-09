# =========================================================================
# OVERALL FILE EXPLANATION:
# This script is the "final exam" for your trained AI model. Now that the 
# model has learned how to compare voices, this script tests it on real audio 
# files. It checks if the model successfully pushes the mathematical representations 
# (embeddings) of Healthy and Parkinson's voices far apart, while keeping two 
# different Healthy voices clustered close together. If the PD distance is 
# significantly higher than the Healthy distance, your model works!
# =========================================================================

import torch
from ml_engine.voice.model import SiameseVoiceNet
from ml_engine.voice.preprocess import preprocess_audio
from ml_engine.voice.buildPairs import list_wav_files

model = SiameseVoiceNet(embedding_dim=128)
model.load_state_dict(torch.load("ml_engine/voice/weights/voice_model.pth"))
model.eval()
# -------------------------------------------------------------------------
# EXPLANATION FOR SETUP & MODEL LOADING:
# Takes: The saved weights file (`voice_model.pth`) from your hard drive.
# Returns: A fully loaded, trained PyTorch model in memory.
# What it does: First, it builds the empty "skeleton" of your neural network. 
# Then, it loads the "brain" (the weights you saved during training) into that 
# skeleton. Finally, `model.eval()` locks the brain so it doesn't accidentally 
# learn or change its weights while we are just testing it.
# -------------------------------------------------------------------------


def get_embedding(path):
    mel = preprocess_audio(path).unsqueeze(0)
    with torch.no_grad():
        return model.embed(mel).squeeze(0)
# -------------------------------------------------------------------------
# EXPLANATION FOR get_embedding:
# Takes: `path` (string) — the file path to a single .wav file.
# Returns: A 1D Tensor (an array of 128 numbers) representing the voice's unique "fingerprint".
# What it does: 
# 1. Converts the raw audio into a Mel-spectrogram image.
# 2. `unsqueeze(0)` adds a fake "batch" dimension because PyTorch expects 
#    groups of files, even if we are only giving it one.
# 3. `torch.no_grad()` tells PyTorch to turn off its tracking engine to save memory.
# 4. `model.embed()` pushes the audio through the network to get the 128-number fingerprint.
# 5. `squeeze(0)` removes the fake batch dimension before returning the data.
# -------------------------------------------------------------------------


healthy_files = list_wav_files("ml_engine/data/raw/healthy_voice")[:20]
pd_files = list_wav_files("ml_engine/data/raw/parkinsons_voice")[:20]

healthy_healthy_dists = []
healthy_pd_dists = []
# -------------------------------------------------------------------------
# EXPLANATION FOR DATA PREPARATION:
# Takes: The folder paths containing your raw data.
# Returns: Two lists containing exactly 20 file paths each, and two empty lists.
# What it does: It grabs a small, manageable sample of 20 healthy voices and 
# 20 Parkinson's voices to run our quick test. It also sets up two empty lists 
# (`healthy_healthy_dists` and `healthy_pd_dists`) to act as buckets to store 
# the mathematical distances we are about to calculate.
# -------------------------------------------------------------------------


for i in range(10):
    e1 = get_embedding(healthy_files[i])
    e2 = get_embedding(healthy_files[i + 10])
    healthy_healthy_dists.append(torch.norm(e1 - e2).item())

    e3 = get_embedding(pd_files[i])
    healthy_pd_dists.append(torch.norm(e1 - e3).item())
# -------------------------------------------------------------------------
# EXPLANATION FOR THE EVALUATION LOOP:
# Takes: The file paths from the lists above.
# Returns: Nothing directly, but it fills the empty lists with distance scores.
# What it does: It loops 10 times. In each loop, it:
# 1. Gets the AI's "fingerprint" for a healthy voice (e1).
# 2. Gets the fingerprint for a completely different healthy voice (e2).
# 3. Calculates the geometric distance (`torch.norm`) between them and saves it.
# 4. Gets the fingerprint for a Parkinson's voice (e3).
# 5. Calculates the distance between the first healthy voice (e1) and the PD voice (e3).
# -------------------------------------------------------------------------


print(f"Avg healthy-healthy distance: {sum(healthy_healthy_dists)/len(healthy_healthy_dists):.4f}")
print(f"Avg healthy-PD distance: {sum(healthy_pd_dists)/len(healthy_pd_dists):.4f}")
# -------------------------------------------------------------------------
# EXPLANATION FOR THE OUTPUT:
# Takes: The fully populated lists of distance scores.
# Returns: Two printed lines in your terminal.
# What it does: It adds up all 10 scores in each bucket and divides by 10 to 
# get the average. If your AI trained well, the "healthy-PD distance" should 
# be a noticeably larger number than the "healthy-healthy distance". 
# -------------------------------------------------------------------------