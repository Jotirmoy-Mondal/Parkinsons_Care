# ml_engine/voice/calibrate.py — full replacement

import statistics
import torch

from ml_engine.voice.model import SiameseVoiceNet
from ml_engine.voice.preprocess import preprocess_audio
from ml_engine.voice.buildPairs import list_wav_files


def get_embedding(model, path):
    mel = preprocess_audio(path).unsqueeze(0)
    with torch.no_grad():
        return model.embed(mel).squeeze(0)


def collect_distances(model, healthy_files, pd_files, num_samples=50):
    healthy_healthy_dists = []
    healthy_pd_dists = []

    for i in range(num_samples):
        e1 = get_embedding(model, healthy_files[i % len(healthy_files)])
        e2 = get_embedding(model, healthy_files[(i + num_samples) % len(healthy_files)])
        healthy_healthy_dists.append(torch.norm(e1 - e2).item())

        e3 = get_embedding(model, pd_files[i % len(pd_files)])
        healthy_pd_dists.append(torch.norm(e1 - e3).item())

    return healthy_healthy_dists, healthy_pd_dists


def calibrate_two_point(
    d_healthy: float, d_pd: float,
    target_healthy_score: float = 90.0,
    target_pd_score: float = 35.0,
):
    """
    Fits a smooth exponential curve through two known points:
    score(d_healthy) == target_healthy_score
    score(d_pd) == target_pd_score
    """
    def score_fn(distance: float) -> float:
        ratio = target_pd_score / target_healthy_score
        exponent = (distance - d_healthy) / (d_pd - d_healthy)
        score = target_healthy_score * (ratio ** exponent)
        return max(0.0, min(100.0, score))
    return score_fn


if __name__ == "__main__":
    model = SiameseVoiceNet(embedding_dim=128)
    model.load_state_dict(torch.load("ml_engine/voice/weights/voice_model.pth"))
    model.eval()

    healthy_files = list_wav_files("ml_engine/data/raw/healthy_voice")
    pd_files = list_wav_files("ml_engine/data/raw/parkinsons_voice")

    healthy_dists, pd_dists = collect_distances(model, healthy_files, pd_files, num_samples=50)

    d_healthy = statistics.median(healthy_dists)
    d_pd = statistics.median(pd_dists)

    print(f"Median healthy-healthy distance: {d_healthy:.4f}")
    print(f"Median healthy-PD distance: {d_pd:.4f}")

    stability_score = calibrate_two_point(d_healthy, d_pd, target_healthy_score=90.0, target_pd_score=35.0)

    print(f"\nScore at healthy median: {stability_score(d_healthy):.1f}%")
    print(f"Score at PD median: {stability_score(d_pd):.1f}%")
    print(f"Score at midpoint distance: {stability_score((d_healthy + d_pd) / 2):.1f}%")