import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
# Path to the pre-trained face detector Haar Cascade XML file.
# You can download this from the OpenCV GitHub repository or it might be included with your OpenCV installation.
# Example: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'

# Path to the pre-trained face mask detection model (.h5 file).
# This model needs to be trained to classify faces as 'with_mask' or 'without_mask'.
# A MobileNetV2-based model is commonly used for this.
# You can find pre-trained models on platforms like GitHub or Kaggle (e.g., search for "face mask detection MobileNetV2 .h5").
# For example, models named 'mask_detector.model' or 'mask-detector-model.h5' are common.
MASK_DETECTOR_MODEL_PATH = 'mask_detector.model' # Or 'mask-detector-model.h5'

# Define the input dimensions for the mask detection model.
# MobileNetV2 typically expects 224x224 input.
MODEL_INPUT_DIM = (224, 224)

# Define the confidence threshold for displaying predictions.
CONFIDENCE_THRESHOLD = 0.85

# --- Helper Functions ---

def load_face_detector(path):
    """
    Loads the pre-trained Haar Cascade classifier for face detection.

    Args:
        path (str): Path to the Haar Cascade XML file.

    Returns:
        cv2.CascadeClassifier: Loaded face detector, or None if loading fails.
    """
    if not os.path.exists(path):
        print(f"Error: Face detector XML file not found at {path}")
        print("Please download 'haarcascade_frontalface_default.xml' and place it in the same directory as the script.")
        return None
    face_detector = cv2.CascadeClassifier(path)
    if face_detector.empty():
        print(f"Error: Could not load face detector from {path}")
        return None
    return face_detector

def load_mask_detector_model(path):
    """
    Loads the pre-trained deep learning model for mask detection.

    Args:
        path (str): Path to the Keras model file (.h5).

    Returns:
        tf.keras.Model: Loaded Keras model, or None if loading fails.
    """
    if not os.path.exists(path):
        print(f"Error: Mask detector model file not found at {path}")
        print("Please ensure your pre-trained model (e.g., 'mask_detector.model' or '.h5') is in the correct path.")
        return None
    try:
        model = load_model(path)
        print(f"Successfully loaded mask detector model from {path}")
        return model
    except Exception as e:
        print(f"Error loading mask detection model from {path}: {e}")
        return None

def detect_faces(frame, face_detector):
    """
    Detects faces in a given frame using the Haar Cascade classifier.

    Args:
        frame (np.array): The input video frame.
        face_detector (cv2.CascadeClassifier): The loaded face detector.

    Returns:
        list: A list of (x, y, w, h) tuples representing detected face bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjust minSize and scaleFactor as needed for your environment.
    # minNeighbors controls false positives.
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces

def preprocess_face_roi(face_roi):
    """
    Preprocesses a face ROI for input into the mask detection model.

    Args:
        face_roi (np.array): The extracted face Region of Interest.

    Returns:
        np.array: Preprocessed face ROI ready for model prediction.
    """
    # Resize to the model's expected input dimensions
    face_resized = cv2.resize(face_roi, MODEL_INPUT_DIM)
    # Convert BGR to RGB (TensorFlow/Keras models often expect RGB)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to the range [0, 1]
    face_normalized = face_rgb.astype("float32") / 255.0
    # Add batch dimension (model expects batch of images)
    face_preprocessed = np.expand_dims(face_normalized, axis=0)
    return face_preprocessed

def classify_mask(face_roi_preprocessed, mask_detector_model):
    """
    Uses the mask detection model to classify a preprocessed face ROI.

    Args:
        face_roi_preprocessed (np.array): Preprocessed face ROI.
        mask_detector_model (tf.keras.Model): The loaded mask detection model.

    Returns:
        tuple: A tuple (label, confidence) where label is "Mask" or "No Mask",
               and confidence is the prediction probability.
    """
    # Predict probabilities (assuming a binary classification model: [mask_prob, no_mask_prob])
    predictions = mask_detector_model.predict(face_roi_preprocessed)[0]

    # The order of classes might vary. Typically, index 0 is 'Mask' and index 1 is 'No Mask'
    # if the training dataset was ordered alphabetically, but it's good to confirm with the model's training.
    # For this example, let's assume index 0 = No Mask, index 1 = Mask (common in many models).
    mask_prob = predictions[1]  # Probability of 'Mask'
    no_mask_prob = predictions[0] # Probability of 'No Mask'

    if mask_prob > no_mask_prob:
        label = "Mask"
        confidence = mask_prob
    else:
        label = "No Mask"
        confidence = no_mask_prob

    return label, confidence

def draw_results(frame, x, y, w, h, label, confidence):
    """
    Draws bounding box and label on the frame.

    Args:
        frame (np.array): The input video frame.
        x (int): x-coordinate of the bounding box top-left corner.
        y (int): y-coordinate of the bounding box top-left corner.
        w (int): Width of the bounding box.
        h (int): Height of the bounding box.
        label (str): Predicted label ("Mask" or "No Mask").
        confidence (float): Prediction confidence.
    """
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green for Mask, Red for No Mask
    text = f"{label} ({confidence * 100:.2f}%)"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    """
    Main function to run the real-time face mask detection system.
    """
    # Load face detector and mask detection model
    face_detector = load_face_detector(FACE_DETECTOR_PATH)
    mask_detector_model = load_mask_detector_model(MASK_DETECTOR_MODEL_PATH)

    if face_detector is None or mask_detector_model is None:
        print("Exiting due to model/detector loading failure.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 for default webcam, change if you have multiple cameras

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optional: Video writer for saving the output stream
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for .avi
    output_filename = 'output_mask_detection.avi'
    # Get frame width, height, and FPS from the camera
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Fallback if FPS is not properly reported by camera
        fps = 20.0
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {output_filename}")


    print("Starting webcam stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, exiting...")
            break

        # Detect faces in the current frame
        faces = detect_faces(frame, face_detector)

        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y + h, x:x + w]

            if face_roi.size == 0: # Skip if ROI is empty (e.g., face at edge of frame)
                continue

            # Preprocess the face ROI
            face_roi_preprocessed = preprocess_face_roi(face_roi)

            # Classify mask presence
            label, confidence = classify_mask(face_roi_preprocessed, mask_detector_model)

            # Draw results if confidence is above threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                draw_results(frame, x, y, w, h, label, confidence)
            else:
                # Optionally, draw a grey box for uncertain detections
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
                cv2.putText(frame, "Uncertain", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)


        # Display the output frame
        cv2.imshow("Face Mask Detection", frame)

        # Write the frame to the output video file
        out.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release() # Release the video writer
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    # Ensure TensorFlow uses eager execution (default for TF2.x)
    # tf.compat.v1.enable_eager_execution() # Uncomment if you face issues with graph mode
    main()
