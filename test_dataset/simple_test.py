import face_recognition

# Load the known good test image (saved in your folder)
image = face_recognition.load_image_file("obama.jpg")

try:
    face_locations = face_recognition.face_locations(image, model="hog")
    print(f"✅ Detected {len(face_locations)} face(s).")
except Exception as e:
    print("❌ ERROR:", e)
