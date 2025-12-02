import cv2
import numpy as np
from skimage.feature import hog
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def process_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (128, 128))
    fd, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return fd


def detect_wanted_person(ref_path, crowd_path, threshold=0.5):
    if not os.path.exists(ref_path) or not os.path.exists(crowd_path):
        print("ERROR: Image files not found")
        return 


    ref_img = cv2.imread(ref_path)
    crowd_img = cv2.imread(crowd_path)

    ref_faces = face_cascade.detectMultiScale(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), 1.1, 3, minSize=(20, 20))
    if len(ref_faces) == 0:
        print("No face in reference image")
        return

    x, y, w, h = ref_faces[0]
    ref_features = process_face(ref_img[y:y + h, x:x + w])

    crowd_faces = face_cascade.detectMultiScale(cv2.cvtColor(crowd_img, cv2.COLOR_BGR2GRAY), 1.05, 3, minSize=(20, 20))
    print(f"Detected {len(crowd_faces)} faces")

    result = crowd_img.copy()
    matches = 0

    for i, (x, y, w, h) in enumerate(crowd_faces):
        features = process_face(crowd_img[y:y + h, x:x + w])
        distance = np.linalg.norm(ref_features - features)
        similarity = 1 / (1 + distance)

        if similarity > threshold:
            matches += 1
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{similarity:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Found {matches} matches")
    cv2.imwrite('result.jpg', result)
    cv2.imshow('Results', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


script_dir = os.path.dirname(os.path.abspath(__file__))
detect_wanted_person(os.path.join(script_dir, 'wanted.jpg'), os.path.join(script_dir, 'crowd.jpg'), threshold=0.08)
