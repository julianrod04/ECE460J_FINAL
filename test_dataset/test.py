import numpy as np
import face_recognition
from PIL import Image

image_path = "Will_Smith_cleaned.jpg"

# Load using PIL and force RGB
pil_image = Image.open(image_path).convert("RGB")
np_image = np.array(pil_image).astype(np.uint8)
np_image = np.ascontiguousarray(np_image)

print(f"Mode: {pil_image.mode}, dtype: {np_image.dtype}, shape: {np_image.shape}")

# Try detection
try:
    face_locations = face_recognition.face_locations(np_image, model="hog")
    print(f"Detected {len(face_locations)} face(s).")
except Exception as e:
    print("ERROR:", e)
