from django.db import models
from django.contrib.auth.models import User

class DailyWritingTest(models.Model):
    # 1. Link this test entry to a specific user/patient from Django's built-in Auth system
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name="writing_tests")
    
    # 2. Store the chosen test type (so the system knows if it's comparing spirals or meanders)
    TEST_TYPE_CHOICES = [
        ('spiral', 'Archimedean Spiral'),
        ('meander', 'Meander Pattern'),
    ]
    test_type = models.CharField(max_length=10, choices=TEST_TYPE_CHOICES, default='spiral')
    
    # 3. Store the physical image file. Django automatically uploads this to a dated path inside /media/
    test_image = models.ImageField(upload_to='handwriting_samples/%Y/%m/%d/')
    
    # 4. Store the similarity score out of 100 calculated by your Siamese network
    # We allow null=True because when the user first uploads the image, the AI calculation is still processing
    stability_score = models.FloatField(null=True, blank=True)
    
    # 5. Automatically record the exact timestamp when this specific test was completed
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at'] # Ensures queries return the newest tests first

    def __str__(self):
        return f"{self.patient.username} - {self.test_type} - {self.created_at.strftime('%Y-%m-%d')}"