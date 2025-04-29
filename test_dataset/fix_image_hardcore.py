from PIL import Image
import cv2
import numpy as np

# Load with Pillow and convert to RGB
image = Image.open("Will_Smith_test.jpg").convert("RGB")

# Rebuild image from raw RGB pixels (strips any profile data)
clean_array = np.array(image).astype(np.uint8)
clean_array = cv2.cvtColor(clean_array, cv2.COLOR_RGB2BGR)  # Save as BGR JPEG

# Save clean image
cv2.imwrite("Will_Smith_cleaned.jpg", clean_array)

print("✅ Saved cleaned image as 'Will_Smith_cleaned.jpg'")
