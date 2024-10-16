import cv2
import numpy as np
from fer import FER
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize FER (Facial Expression Recognition) model
emotion_detector = FER()

# Load overlay images
dog_nose = cv2.imread(r"C:\Users\ragha\OneDrive\Desktop\python\images\dog_nose_transparent.png",
                      cv2.IMREAD_UNCHANGED)  # Ensure it's a transparent PNG
dog_face = cv2.imread(r"C:\Users\ragha\OneDrive\Desktop\project images\dog_tongue_transparent.png",
                      cv2.IMREAD_UNCHANGED)  # Ensure it's loaded with transparency

# Video files for specific emotions
videos = {
    "angry": r"C:\Users\ragha\Downloads\WhatsApp Video 2024-10-14 at 11.39.48 PM (2).mp4",
    "happy": r"C:\Users\ragha\Downloads\WhatsApp Video 2024-10-14 at 11.39.48 PM.mp4",
    "sad": r"C:\Users\ragha\Downloads\WhatsApp Video 2024-10-14 at 11.39.48 PM (1).mp4",
}

# Capture video from the webcam
cap = cv2.VideoCapture(0)


# Function to overlay the video with a subtle background effect
def blend_video_with_feed(video_capture, frame, alpha=0.3):
    ret, video_frame = video_capture.read()
    if not ret:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video back to start
        ret, video_frame = video_capture.read()

    # Resize the video frame to match the dimensions of the webcam frame
    video_frame_resized = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))

    # Blend the video frame in the background with a certain level of transparency
    blended_frame = cv2.addWeighted(video_frame_resized, alpha, frame, 1 - alpha, 0)

    return blended_frame


effect = 0  # Start with no effect
min_w, min_h = 100, 100
faces = []  # List to store face regions for swapping
save_dir = r"C:\Users\ragha\OneDrive\Desktop\project images\saved images"  # Directory to save images
image_count = 1  # Counter for saved images
current_emotion = None
emotion_video = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    # Apply effects based on keypress
    if effect == 1:  # Dog nose effect
        for (x, y, w, h) in detections:
            resized_dog_nose = cv2.resize(dog_nose, (w // 3, h // 3))
            nose_w, nose_h, _ = resized_dog_nose.shape
            y_nose = y + h // 2
            x_nose = x + w // 3
            if y_nose + nose_h < frame.shape[0] and x_nose + nose_w < frame.shape[1]:
                dog_nose_rgb = resized_dog_nose[:, :, :3]  # RGB (3 channels)
                dog_nose_alpha = resized_dog_nose[:, :, 3] / 255.0  # Alpha (1 channel)

                roi = frame[y_nose:y_nose + nose_h, x_nose:x_nose + nose_w]

                for c in range(0, 3):  # Loop through RGB channels
                    roi[:, :, c] = roi[:, :, c] * (1 - dog_nose_alpha) + dog_nose_rgb[:, :, c] * dog_nose_alpha

                frame[y_nose:y_nose + nose_h, x_nose:x_nose + nose_w] = roi

    elif effect == 2:  # Dog face effect
        for (x, y, w, h) in detections:
            resized_dog_face = cv2.resize(dog_face, (w, h))
            dog_face_rgb = resized_dog_face[:, :, :3]
            dog_face_alpha = resized_dog_face[:, :, 3] / 255.0

            if y + h < frame.shape[0] and x + w < frame.shape[1]:
                roi = frame[y:y + h, x:x + w]

                for c in range(0, 3):
                    roi[:, :, c] = roi[:, :, c] * (1 - dog_face_alpha) + dog_face_rgb[:, :, c] * dog_face_alpha

                frame[y:y + h, x:x + w] = roi

    elif effect == 3:  # Blur the entire frame
        frame = cv2.GaussianBlur(frame, (15, 15), 0)

    elif effect == 4:  # Blur everything except faces
        blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
        mask = np.zeros_like(frame)

        for (x, y, w, h) in detections:
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]  # Keep faces unblurred

        frame = blurred_frame
        for (x, y, w, h) in detections:
            frame[y:y + h, x:x + w] = mask[y:y + h, x:x + w]  # Keep faces unblurred

    elif effect == 5:  # Remove all filters (reset frame)
        pass  # Just do nothing, the frame remains as is

    # Apply emotion detection and video overlay when '6' is pressed
    if effect == 6:
        emotion_detected = False  # Track whether any specific emotion video should be played
        for face in detections:
            x, y, w, h = face

            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]

            # Detect the emotion using FER
            emotion = emotion_detector.top_emotion(face_roi)

            # Ensure that emotion is not None
            if emotion:
                emotion_label, score = emotion
                if emotion_label is not None and score is not None:
                    # Display the emotion on the frame
                    cv2.putText(frame, f"{emotion_label} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

                    # Check if the detected emotion has a dedicated video
                    if emotion_label in videos:
                        if current_emotion != emotion_label:
                            current_emotion = emotion_label
                            if emotion_video:
                                emotion_video.release()  # Release the previous video
                            emotion_video = cv2.VideoCapture(videos[emotion_label])  # Load the new video
                        emotion_detected = True

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # If an emotion video is detected, blend the video in the background
        if emotion_detected and emotion_video and emotion_video.isOpened():
            frame = blend_video_with_feed(emotion_video, frame, alpha=0.3)  # Adjust alpha for background effect
        else:
            current_emotion = None  # Revert to normal live feed if no emotion is detected

    # Draw a blue rectangle around each detected face
    for (x, y, w, h) in detections:
        if w >= min_w and h >= min_h:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)  # Blue border with thickness of 5

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Check for keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('1'):  # Apply dog nose
        effect = 1
    elif key == ord('2'):  # Apply dog face
        effect = 2
    elif key == ord('3'):  # Blur the whole frame
        effect = 3
    elif key == ord('4'):  # Blur everything except faces
        effect = 4
    elif key == ord('5'):  # Remove all filters (reset frame)
        effect = 5
    elif key == ord('6'):  # Apply emotion detection and video overlay
        effect = 6
    elif key == ord('7'):  # Save photo with effect
        # Create a unique filename
        save_path = os.path.join(save_dir, f"saved_image_{image_count}.png")
        cv2.imwrite(save_path, frame)  # Save the current frame
        print(f"Image saved to {save_path}")
        image_count += 1

# Release the capture and close windows
if emotion_video:
    emotion_video.release()
cap.release()
cv2.destroyAllWindows()