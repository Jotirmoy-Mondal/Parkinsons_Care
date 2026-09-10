# =========================================================================
# OVERALL FILE EXPLANATION:
# This script manages a patient's "audio anchors" — the fixed reference points 
# used by your Siamese Neural Network to measure voice degradation over time. 
# It handles fetching or creating a patient's initial healthy baseline (day one), 
# and calculating their short-term average (last 30 days). This allows your app 
# to track both long-term disease progression and short-term symptom fluctuations!
# =========================================================================

import torch
from django.utils import timezone
from datetime import timedelta

from voice_test.models import PatientVoiceBaseline, VoiceTestResult
from ml_engine.voice.inference import compute_embedding, MODEL_VERSION
# -------------------------------------------------------------------------
# EXPLANATION FOR IMPORTS:
# Takes/Returns: Nothing directly.
# What it does: Brings in PyTorch for math, Django's time tools to calculate 
# dates (like "30 days ago"), your database models, and the AI inference tools 
# you built earlier to generate the actual audio fingerprints.
# -------------------------------------------------------------------------


def get_or_create_baseline(patient, audio_path: str) -> torch.Tensor:
    """
    Fetches the patient's stored baseline embedding, or creates one
    if this is their first-ever voice test.
    """
    try:
        baseline_record = PatientVoiceBaseline.objects.get(patient=patient)
        return torch.tensor(baseline_record.embedding)
    except PatientVoiceBaseline.DoesNotExist:
        embedding = compute_embedding(audio_path)
        PatientVoiceBaseline.objects.create(
            patient=patient,
            embedding=embedding.tolist(),
            model_version=MODEL_VERSION,
        )
        return embedding
# -------------------------------------------------------------------------
# EXPLANATION FOR get_or_create_baseline:
# Takes: `patient` (the database profile) and `audio_path` (string to the .wav file).
# Returns: A `torch.Tensor` (the 128-number baseline fingerprint).
# What it does: It first asks the database, "Does this patient already have an 
# anchor?" If yes, it loads it and converts it back into a PyTorch Tensor. If the 
# database throws a "DoesNotExist" error, it knows this is the patient's very 
# first test. It then runs your AI to get the fingerprint, saves it to the DB 
# as a standard Python list, and returns the Tensor for the rest of your app to use.
# -------------------------------------------------------------------------


def get_recent_mean_embedding(patient, days: int = 30) -> torch.Tensor:
    """
    Computes the mean embedding across a patient's tests in the last
    N days, for short-term trend comparison. Returns None if there
    isn't enough recent history yet.
    """
    cutoff = timezone.now() - timedelta(days=days)
    recent_results = VoiceTestResult.objects.filter(
        patient=patient, created_at__gte=cutoff
    )

    if not recent_results.exists():
        return None

    embeddings = [torch.tensor(r.embedding) for r in recent_results]
    stacked = torch.stack(embeddings)
    return stacked.mean(dim=0)
# -------------------------------------------------------------------------
# EXPLANATION FOR get_recent_mean_embedding:
# Takes: `patient` (the database profile) and `days` (integer, default is 30).
# Returns: A `torch.Tensor` (the averaged fingerprint) or `None`.
# What it does: It calculates the exact date and time for 30 days ago, then asks 
# the database for every voice test the patient has taken since that exact moment 
# (`created_at__gte=cutoff`). If there are no recent tests, it returns None. If tests 
# do exist, it stacks all those daily fingerprints on top of each other and 
# mathematically squashes them down (`.mean`) into a single "average" fingerprint 
# for the past month.
# -------------------------------------------------------------------------