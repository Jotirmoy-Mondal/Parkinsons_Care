"""**`rest_framework`** (officially known as **Django REST Framework** or **DRF**) is a powerful toolkit used in Python to build Web APIs.


**Django REST Framework is the translator.** It allows your Python server to stop speaking "HTML for humans" and start speaking "JSON data for machines."

### The Three Superpowers of DRF

#### 1. Serializers (The Translators)

In your `predictor.py` script, your AI outputs a Python dictionary or a float (like `87.70`). A frontend mobile app built in React Native or Flutter cannot read Python objects.
A DRF **Serializer** automatically grabs your Python data, translates it into JSON (the universal language of the web), and sends it out.

* *Python:* `{"stability_score": 87.70, "status": "severe"}`
* *JSON (via DRF):* `{"stability_score": 87.7, "status": "severe"}`

#### 2. Views (The Bouncers)

DRF sets up specific URLs (Endpoints) and decides what actions are allowed.
For example, you could easily write a DRF View that says: *"If a user sends a POST request with an image to `/api/grade_drawing/`, run the PyTorch script. If they send a GET request, reject it."*

#### 3. Authentication (The Security Guard)

Medical data is highly sensitive. DRF has built-in systems to ensure that only logged-in, authorized users with secure API tokens can access your machine learning model. It prevents random people on the internet from uploading images and crashing your server.

### How It Fits Into this Current Project

Right now, your PyTorch script runs in a terminal. To turn it into a real product, the architecture usually looks like this:

1. **The Frontend:** A React or mobile app where the patient draws the spiral.
2. **The API (Django REST Framework):** Receives the image over the internet securely, unpacks it, and hands it to your PyTorch script.
3. **The Brain (PyTorch):** Your `predictor.py` cleans the image, calculates the `stability_score`, and hands the number back to DRF.
4. **The Delivery:** DRF translates that score into JSON and sends it back to the patient's phone so they can see their result on the screen.



What is the need for APIView?
If you tried to build a web endpoint using raw, basic Python, you would have to write hundreds of lines of code to manually read network bytes, check security headers, figure out if the user sent JSON or an image file, and handle server crashes.

APIView does all of that heavy lifting for you automatically. It provides four massive benefits out of the box:

1. HTTP Method Routing (The Traffic Cop)
When a mobile app talks to your server, it uses HTTP methods.

GET: "Give me data."

POST: "Here is new data, process it."

APIView perfectly organizes this. You just write a function named post() inside your class, and APIView guarantees that only POST requests will trigger that specific code.

2. The Magical request.data
When a patient uploads their Parkinson's drawing from a phone, that image is sent across the internet in a messy HTTP format. APIView automatically catches it, cleans it up, and hands it to you in a neat, easy-to-use Python variable called request.data. You don't have to parse anything manually.

3. The Response Object
Just like your AI needs .item() to return a normal number, a web server needs to return pure JSON. APIView allows you to use a special Response() function. You hand it a standard Python dictionary (like {"score": 87.7}), and APIView perfectly translates it into the JSON format the web browser or mobile app is expecting.

4. Built-in Security and Error Handling
If your PyTorch script crashes (for example, if the user uploads a corrupted image), APIView catches the error so your whole server doesn't go offline. It also allows you to easily attach authentication (checking if the doctor is logged in) with a single line of code.

""""
from rest_framework.views import APIView
from rest_framework.response import Response
"""The Two Things it TAKES (The Inputs)
1. The Data: The actual information you want to send back.
2. The Status Code: A standardized internet number that tells the mobile app if the request was successful or if it failed. (e.g., 200 means OK, 400 means Bad Request)."""
from rest_framework import status
""" #200 OK
return Response(my_data, status=status.HTTP_200_OK)

# 400 Bad Request
return Response(error_data, status=status.HTTP_400_BAD_REQUEST)"""


from rest_framework.parsers import MultiPartParser, FormParser
"""THE Problem it sloves: How the Internet Sends Files
When a patient takes a photo of their spiral drawing on their phone and clicks "Submit," the phone does not just beam a raw .jpg file into your Python script.

Instead, the phone wraps the image file in a massive, ugly stream of text called multipart/form-data. If your server tries to read this raw internet text, it will look like total gibberish, and your PyTorch script will crash.

The Solution: The Parsers
Parsers intercept that messy internet data, rip open the packaging, and extract the clean, usable files and text.

1. MultiPartParser (The Package Opener)
This is the most important tool for your specific project. Its only job is to look at an incoming HTTP request, find any attached files (like the patient's uploaded image), safely extract them, and store them in memory so your clean_and_threshold_image function can actually process them.

2. FormParser (The Form Reader)
While MultiPartParser handles the heavy image files, FormParser handles standard text fields. If the patient's phone sends the image and a text string that says test_type="spiral", the FormParser unpacks that text so you can use it to load the correct AI weights"""


from django.core.files.storage import default_storage

"""The Problem: Memory vs. Hard Drive
In your last step, the MultiPartParser caught the patient's drawing and handed it to you as request.FILES.get('drawing').

However, at this exact moment, that image is floating in your server's RAM (Temporary Memory).
If you remember your PyTorch script, your clean_and_threshold_image function uses OpenCV (cv2.imread(image_path)). OpenCV is strictly designed to read files from a physical hard drive path (like "C:/uploads/image.jpg").

If you try to hand OpenCV an in-memory file, it will completely crash. You must save the file to your hard drive first!

The Solution: default_storage
Instead of writing complex, messy Python code to open files, write bytes, and manage directories, Django gives you default_storage. It handles all the dirty work of saving files in one line of code.

The Two Superpowers of default_storage
1. default_storage.save()
It takes the file floating in memory and officially saves it to your hard drive.
2. default_storage.path()
Once saved, it calculates the exact absolute path on your computer (e.g., "/workspaces/Parkinsons_Care/media/patient_drawing.jpg") so you can hand it straight to OpenCV."""


from django.core.files.base import ContentFile

"""The Problem: Loose Papers (Raw Bytes)
In the previous step, you used request.FILES. This works perfectly if the patient's phone uploads the image as a standard "File." Django automatically puts it in a nice, neat folder (UploadedFile) that default_storage knows exactly how to handle.

But what if the mobile app doesn't send a file?
Very often in modern web development, mobile apps (like React Native) prefer to convert images into a massive, garbled string of text called Base64 and send it as standard JSON.
It looks like this: {"image": "iVBORw0KGgoAAAANSUhEUgAA..."}.

If your Python server receives that text, you have to decode it back into raw computer bytes. But if you try to hand raw bytes directly to default_storage, the Filing Clerk will reject it. It doesn't accept "loose papers." It only accepts official Django File objects.

The Solution: ContentFile
ContentFile takes your raw bytes (or raw text) and perfectly wraps them into a standard Django File object so the storage system can save it to your hard drive."""

import os

# Import your database model and the PyTorch toolbelt!
from .models import DailyWritingTest
from ml_engine.predictor import evaluate_stability

class ParkinsonTestAnalysisView(APIView):
    # This tells Django to expect a file upload (like a photo from a phone)
    parser_classes = [MultiPartParser, FormParser]
    #If a user sends us an image, use the first blueprint to build a worker. If they send us a text form, use the second blueprint.

    def post(self, request, *format=None):
        test_type = request.data.get('test_type')
        uploaded_file = request.data.get('uploaded_file')
        
        if not test_type or not uploaded_file:
            return Response({"error": "Missing test_type or uploaded_file"}, status=status.HTTP_400_BAD_REQUEST)
            
        # 1. Temporarily save the incoming photo to the server disk
        temp_path = default_storage.save(f"tmp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        """uploaded_file.read(): You open the file the patient sent and read 100% of its raw bytes into the server's temporary memory.

        ContentFile(...): You wrap those raw bytes inside a brand new Django "Manila Folder".

        f"tmp/{uploaded_file.name}": You tell the Filing Clerk to create a folder named tmp (if it doesn't exist) and save the file inside it using the exact name the patient gave it (e.g., tmp/spiral_test.jpg).

        temp_path: Django hands you back the relative database path (e.g., tmp/spiral_test.jpg)."""

        full_temp_path = default_storage.path(temp_path)
        """It calculates the massive, absolute hard-drive path (e.g., C:/workspaces/Parkinsons_Care/media/tmp/spiral_test.jpg)"""
        
        try:
            # 2. CALL THE PYTORCH ENGINE (Your predictor.py script!)
            calculated_stability = evaluate_stability(full_temp_path, test_type)
            
            # 3. Save the official record to your Django database
            test_record = DailyWritingTest.objects.create(
                patient=request.user, # Assumes the mobile app sends an authenticated token
                test_type=test_type,
                test_image=uploaded_file,
                stability_score=calculated_stability
            )
            
            # Clean up the temporary file
            if os.path.exists(full_temp_path):
                os.remove(full_temp_path)
            
            # 4. Return the calculated score back to the Android screen
            return Response({
                "success": True,
                "data": {
                    "test_type": test_record.test_type,
                    "stability_score": test_record.stability_score
                }
            }, status=status.HTTP_201_CREATED)


        """1. What is e?
        When you write except Exception as e:, Python takes the entire crash event (the line number, the memory state, the type of error) and packages it into a massive Python Object, which it names e.

        If you tried to send e directly over the internet like this:

        Python
        # Broken! 
        return Response({"error": e}, status=500)
        Your server would crash again. Why? Because the Response tool is trying to convert your data into JSON so the mobile app can read it. JSON only understands basic text, numbers, and lists. It has no idea how to read a complex "Python Exception Object," so it panics and fails.

        2. What str(e) does
        str() is a built-in Python function that means "String" (text).

        When you wrap the error object like this—str(e)—you are forcing Python to rip the most important, human-readable sentence out of that massive error object and turn it into standard text.

        Example 1: A Math Error

        The crash: Your code accidentally divides by zero.

        What e is: <class 'ZeroDivisionError'> object at 0x7f8...

        What str(e) becomes: "division by zero"

        Example 2: A File Error

        The crash: OpenCV tries to open an image that doesn't exist.

        What str(e) becomes: "Could not read image at path: tmp/spiral.jpg"

        3. Why it is brilliant for debugging
        By putting str(e) inside your JSON response, you are sending the exact crash reason directly to the patient's phone.

        When you (the developer) are testing the mobile app and something goes wrong, you won't just see a generic "Server Error" on the screen. The screen will literally print out:
        {"error": "Missing weight metrics for 'spiral' at /weights/siamese_spiral.pth"}

        It saves you from having to remote into your server, dig through hundreds of lines of terminal logs, and guess what went wrong. The server tells you exactly what line to fix!"""    
        except Exception as e:
            if os.path.exists(full_temp_path):
                os.remove(full_temp_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)