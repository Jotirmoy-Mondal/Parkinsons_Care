# writing_test/models.py

from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="patient_profile")
    date_of_birth = models.DateField(blank=True, null=True)
    diagnosis_date = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.get_full_name() or self.user.username

class DailyWritingTest(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="writing_tests")

    TEST_TYPE_CHOICES = [
        ('spiral', 'Archimedean Spiral'),
        ('meander', 'Meander Pattern'),
    ]
    test_type = models.CharField(max_length=10, choices=TEST_TYPE_CHOICES, default='spiral')
    test_image = models.ImageField(upload_to='handwriting_samples/%Y/%m/%d/')
    stability_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient} - {self.test_type} - {self.created_at.strftime('%Y-%m-%d')}"