# voice_test/models.py

from django.db import models
from writing_test.models import Patient  


class VoiceTestResult(models.Model):
    TASK_CHOICES = [
        ("vowel", "Sustained Vowel"),
        ("ddk", "Diadochokinetic (pa-ta-ka)"),
        ("reading", "Reading Passage"),
    ]

    patient = models.ForeignKey(
        Patient, on_delete=models.CASCADE, related_name="voice_test_results"
    )
    task_type = models.CharField(max_length=10, choices=TASK_CHOICES)

    stability_score = models.FloatField(
        help_text="Calibrated 0-100 stability score for this test"
    )
    embedding = models.JSONField(
        help_text="128-D (or model-defined) embedding vector, stored for trend recomputation"
    )
    raw_features = models.JSONField(
        help_text="Interpretable acoustic features: jitter, shimmer, HNR, F0, etc.",
        blank=True, null=True,
    )

    model_version = models.CharField(
        max_length=50,
        help_text="Which trained model/checkpoint produced this result, e.g. 'voice_cnn_v1'",
    )
    audio_duration_sec = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["patient", "task_type", "created_at"]),
        ]

    def __str__(self):
        return f"{self.patient} - {self.task_type} - {self.stability_score:.1f}% ({self.created_at.date()})"