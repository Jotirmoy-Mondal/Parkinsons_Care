# =========================================================================
# OVERALL FILE EXPLANATION:
# This view acts as an API endpoint. When a patient takes a voice test on 
# their device, the device sends the audio here. This script securely receives 
# the audio, temporarily saves it, runs your PyTorch Parkinson's AI, saves 
# the score to the database, and returns a JSON response to the user's screen.
# =========================================================================

import os
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

from voice_test.models import VoiceTestResult
from ml_engine.voice.inference import run_voice_inference
from ml_engine.voice.anchor import get_or_create_baseline, get_recent_mean_embedding
# -------------------------------------------------------------------------
# EXPLANATION FOR IMPORTS:
# Takes/Returns: Nothing directly.
# What it does: Gathers all the required tools. It pulls in Django's web 
# response tools, security decorators, your database models, and the AI 
# scripts you wrote earlier to actually process the audio.
# -------------------------------------------------------------------------


@csrf_exempt  # Android app posts here directly; revisit with token auth before production
@login_required
def upload_voice_test(request):
# -------------------------------------------------------------------------
# EXPLANATION FOR FUNCTION SETUP & SECURITY:
# Takes: `request` (an HTTP object containing the user's uploaded data and login info).
# Returns: Eventually returns a `JsonResponse` (a dictionary sent back to the phone/browser).
# What it does: `@login_required` is a security bouncer—it completely blocks 
# anyone who isn't logged in. `@csrf_exempt` temporarily relaxes a specific 
# web-browser security rule so mobile apps can easily send data to this URL.
# -------------------------------------------------------------------------


    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    audio_file = request.FILES.get("audio")
    task_type = request.POST.get("task_type")

    if not audio_file:
        return JsonResponse({"error": "No audio file provided"}, status=400)
    if task_type not in dict(VoiceTestResult.TASK_CHOICES):
        return JsonResponse({"error": "Invalid or missing task_type"}, status=400)
# -------------------------------------------------------------------------
# EXPLANATION FOR DATA VALIDATION:
# Takes: The incoming `request` data.
# Returns: An error message if the user messed up (HTTP 400 or 405).
# What it does: This is basic quality control. It ensures the user is actually 
# sending data (POST), checks that they attached an audio file, and verifies 
# the test type (like 'vowel' or 'reading') matches what your database expects.
# -------------------------------------------------------------------------


    patient = request.user.patient_profile

    # Save to a temp file — inference functions expect a file path, not an in-memory upload
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        temp_path = tmp.name
# -------------------------------------------------------------------------
# EXPLANATION FOR PATIENT & TEMP FILE HANDLING:
# Takes: The user's login ID and the incoming audio stream.
# Returns: A temporary physical file path (`temp_path`) on your server.
# What it does: First, it grabs the clinical Patient profile linked to the logged-in User. 
# Next, because your PyTorch AI (`librosa`) can only read physical files saved 
# on a hard drive, it takes the web upload, creates a temporary `.wav` file 
# in the cloud, and writes the audio data into it piece by piece (chunks).
# -------------------------------------------------------------------------


    try:
        baseline_embedding = get_or_create_baseline(patient, temp_path)
        recent_mean_embedding = get_recent_mean_embedding(patient, days=30)

        result = run_voice_inference(
            audio_path=temp_path,
            baseline_embedding=baseline_embedding,
            recent_mean_embedding=recent_mean_embedding,
        )
# -------------------------------------------------------------------------
# EXPLANATION FOR AI EXECUTION:
# Takes: The patient profile and the path to the temporary audio file.
# Returns: `result` (a Python dictionary containing the final AI scores and features).
# What it does: It fetches the patient's baseline (their healthy "anchor" voice), 
# then runs your AI model (`run_voice_inference`) to compare today's voice 
# test against their baseline. The `try` block ensures that if the AI crashes, 
# it doesn't crash your whole web server.
# -------------------------------------------------------------------------


        test_result = VoiceTestResult.objects.create(
            patient=patient,
            task_type=task_type,
            stability_score=result["stability_vs_baseline"],
            embedding=result["embedding"],
            raw_features=result["raw_features"],
            model_version=result["model_version"],
        )
# -------------------------------------------------------------------------
# EXPLANATION FOR DATABASE SAVE:
# Takes: The output scores from your AI model.
# Returns: `test_result` (a saved database object).
# What it does: It takes all the raw math and scores generated by your AI 
# and permanently saves them to the database under this specific patient's profile.
# -------------------------------------------------------------------------


        response_data = {
            "id": test_result.id,
            "stability_score": result["stability_vs_baseline"],
            "raw_features": result["raw_features"],
            "created_at": test_result.created_at.isoformat(),
        }

        if "stability_vs_recent" in result:
            response_data["stability_vs_recent"] = result["stability_vs_recent"]

        return JsonResponse(response_data, status=201)

    except Exception as e:
        return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)
# -------------------------------------------------------------------------
# EXPLANATION FOR RESPONSE:
# Takes: The saved test result data.
# Returns: A clean JSON dictionary to the user's screen.
# What it does: Mobile apps and frontend websites prefer talking in JSON. 
# This packages up the final percentage score and test ID, and sends it back 
# to the user with a "201 Created" success code. If the AI failed earlier, 
# the `except` block catches it and sends a "500 Server Error" instead.
# -------------------------------------------------------------------------


    finally:
        # Always clean up the temp file, even if inference failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
# -------------------------------------------------------------------------
# EXPLANATION FOR CLEANUP:
# Takes: The temporary file path.
# Returns: Nothing.
# What it does: The `finally` block is guaranteed to run no matter what happens 
# (even if the AI crashed). It deletes the temporary `.wav` file we made earlier 
# so your server's hard drive doesn't fill up with junk files over time.
# -------------------------------------------------------------------------